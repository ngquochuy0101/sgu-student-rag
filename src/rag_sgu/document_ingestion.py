from __future__ import annotations

import io
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .config import RAGSettings
from .mlops import hash_file
from .schemas import ExtractedDocument


class OCRProcessor:
    _VIETNAMESE_CORRECTIONS: dict[str, str] = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "−": "-",
        "“": '"',
        "”": '"',
        "’": "'",
    }

    def __init__(self, settings: RAGSettings):
        if fitz is None or pytesseract is None or Image is None:
            raise ImportError(
                "OCR dependencies are missing. Install requirements.txt before ingestion."
            )
        self.settings = settings
        self._preprocess_available = (
            self.settings.ocr_preprocessing_enabled
            and cv2 is not None
            and np is not None
        )
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        assert pytesseract is not None
        if self.settings.tesseract_cmd:
            candidate = Path(self.settings.tesseract_cmd)
            if candidate.exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
        if self.settings.tessdata_prefix:
            tessdata_path = Path(self.settings.tessdata_prefix)
            if tessdata_path.exists():
                # Tesseract expects this variable to point to the parent folder of tessdata files.
                os.environ["TESSDATA_PREFIX"] = str(tessdata_path) + os.sep

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_skew(self, gray_image: Any) -> float:
        if cv2 is None or np is None:
            return 0.0

        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            return 0.0

        angles: list[float] = []
        for rho, theta in lines[:, 0]:
            _ = rho
            angle = float(np.degrees(theta) - 90)
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(np.array(angles)))

    def _rotate_image(self, image: Any, angle: float) -> Any:
        if cv2 is None:
            return image
        if abs(angle) < 0.5:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _preprocess_image(self, image: Any) -> Any:
        if not self._preprocess_available:
            return image

        assert cv2 is not None
        assert np is not None

        image_array = np.array(image)
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        if self.settings.ocr_adaptive_threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )

        if self.settings.ocr_denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        if self.settings.ocr_deskew:
            angle = self._detect_skew(gray)
            if abs(angle) > 0.5:
                gray = self._rotate_image(gray, angle)

        return Image.fromarray(gray)

    def _estimate_ocr_confidence(self, image: Any, config: str) -> float:
        assert pytesseract is not None
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.settings.ocr_languages,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:  # noqa: BLE001
            return 0.0

        confidence_values = []
        for value in data.get("conf", []):
            value_str = str(value).strip()
            if not value_str or value_str == "-1":
                continue
            try:
                confidence_values.append(float(value_str))
            except ValueError:
                continue

        if not confidence_values:
            return 0.0
        return sum(confidence_values) / (len(confidence_values) * 100.0)

    def _postprocess_text(self, text: str) -> str:
        cleaned = self.clean_text(text)
        if not cleaned or not self.settings.ocr_vietnamese_correction:
            return cleaned

        for old, new in self._VIETNAMESE_CORRECTIONS.items():
            cleaned = cleaned.replace(old, new)

        if self.settings.ocr_correction_aggressive:
            cleaned = re.sub(r"(?<=\w)- (?=\w)", "", cleaned)
            cleaned = re.sub(r"([A-Za-zÀ-ỹ])\1{3,}", r"\1\1", cleaned)

        return self.clean_text(cleaned)

    def _ocr_page(self, page: Any) -> str:
        assert fitz is not None
        assert Image is not None
        assert pytesseract is not None

        pix = page.get_pixmap(matrix=fitz.Matrix(self.settings.ocr_dpi, self.settings.ocr_dpi))
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        processed_image = self._preprocess_image(image)
        base_config = f"--psm {self.settings.ocr_psm} --oem {self.settings.ocr_oem}"

        max_attempts = max(1, self.settings.ocr_max_retry_attempts)
        best_text = ""
        best_confidence = 0.0

        for attempt in range(max_attempts):
            attempt_config = base_config
            if attempt > 0 and self.settings.ocr_psm != 6:
                attempt_config = f"--psm 6 --oem {self.settings.ocr_oem}"

            raw_text = pytesseract.image_to_string(
                processed_image,
                lang=self.settings.ocr_languages,
                config=attempt_config,
            )
            candidate_text = self._postprocess_text(raw_text)
            candidate_confidence = self._estimate_ocr_confidence(processed_image, attempt_config)

            if candidate_confidence > best_confidence or not best_text:
                best_text = candidate_text
                best_confidence = candidate_confidence

            if best_confidence >= self.settings.ocr_confidence_threshold:
                break

        return best_text

    def extract_pdf(self, pdf_path: Path) -> ExtractedDocument:
        assert fitz is not None
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF file: {pdf_path}")

        source_hash = hash_file(pdf_path)
        page_texts: list[str] = []
        ocr_pages = 0

        with fitz.open(str(pdf_path)) as document:
            page_count = len(document)
            for page in document:
                extracted = self._postprocess_text(page.get_text("text"))
                if len(extracted) >= self.settings.ocr_min_text_chars:
                    page_texts.append(extracted)
                    continue

                ocr_text = self._ocr_page(page)
                page_texts.append(ocr_text)
                ocr_pages += 1

        non_empty_pages = [item for item in page_texts if item.strip()]
        full_text = self._postprocess_text("\n\n".join(non_empty_pages))
        if ocr_pages and ocr_pages < page_count:
            extraction_method = "hybrid"
        elif ocr_pages == page_count:
            extraction_method = "ocr"
        else:
            extraction_method = "text"

        return ExtractedDocument(
            source_path=pdf_path,
            text=full_text,
            page_count=page_count,
            ocr_pages=ocr_pages,
            extraction_method=extraction_method,
            source_hash=source_hash,
            page_texts=page_texts,
        )


class DocumentIngestor:
    def __init__(self, settings: RAGSettings, logger):
        self.settings = settings
        self.logger = logger
        self.ocr_processor = OCRProcessor(settings)

    def _cache_file(self, pdf_path: Path) -> Path:
        stat = pdf_path.stat()
        key = f"{pdf_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
        digest = hash_file_bytes(key.encode("utf-8"))
        return self.settings.cache_dir / f"{digest}.json"

    def _load_from_cache(self, cache_path: Path) -> ExtractedDocument | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return ExtractedDocument.from_cache_payload(payload)
        except Exception as error:  # noqa: BLE001
            self.logger.warning("Failed to read OCR cache %s: %s", cache_path.name, error)
            return None

    def _save_to_cache(self, cache_path: Path, document: ExtractedDocument) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(document.to_cache_payload(), ensure_ascii=False),
            encoding="utf-8",
        )

    def ingest_directory(self, pattern: str = "*.pdf") -> list[ExtractedDocument]:
        if not self.settings.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.settings.pdf_dir}")

        pdf_files = sorted(self.settings.pdf_dir.glob(pattern))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files matching pattern '{pattern}' in {self.settings.pdf_dir}"
            )

        ingested: list[ExtractedDocument] = []
        for pdf_file in pdf_files:
            cache_path = self._cache_file(pdf_file)
            cached = (
                self._load_from_cache(cache_path)
                if self.settings.ocr_cache_enabled
                else None
            )
            if cached is not None:
                self.logger.info("Loaded OCR cache for %s", pdf_file.name)
                ingested.append(cached)
                continue

            try:
                document = self.ocr_processor.extract_pdf(pdf_file)
            except Exception as error:  # noqa: BLE001
                self.logger.exception("Failed to ingest %s: %s", pdf_file.name, error)
                continue

            if document.text:
                ingested.append(document)
                if self.settings.ocr_cache_enabled:
                    self._save_to_cache(cache_path, document)
                self.logger.info(
                    "Ingested %s | pages=%s | ocr_pages=%s | chars=%s",
                    pdf_file.name,
                    document.page_count,
                    document.ocr_pages,
                    len(document.text),
                )

        if not ingested:
            raise RuntimeError("No documents were successfully ingested")
        return ingested


def hash_file_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def collect_pdf_files(pdf_dir: Path, pattern: str = "*.pdf") -> list[Path]:
    return sorted(path for path in pdf_dir.glob(pattern) if path.is_file())


def total_ocr_pages(documents: Iterable[ExtractedDocument]) -> int:
    return sum(item.ocr_pages for item in documents)
