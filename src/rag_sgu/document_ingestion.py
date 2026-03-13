from __future__ import annotations

import io
import json
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

from .config import RAGSettings
from .mlops import hash_file
from .schemas import ExtractedDocument


class OCRProcessor:
    def __init__(self, settings: RAGSettings):
        if fitz is None or pytesseract is None or Image is None:
            raise ImportError(
                "OCR dependencies are missing. Install requirements.txt before ingestion."
            )
        self.settings = settings
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
                import os

                os.environ["TESSDATA_PREFIX"] = str(tessdata_path) + os.sep

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _ocr_page(self, page: Any) -> str:
        assert fitz is not None
        assert Image is not None
        assert pytesseract is not None
        pix = page.get_pixmap(matrix=fitz.Matrix(self.settings.ocr_dpi, self.settings.ocr_dpi))
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_config = f"--psm {self.settings.ocr_psm} --oem {self.settings.ocr_oem}"
        text = pytesseract.image_to_string(
            image,
            lang=self.settings.ocr_languages,
            config=ocr_config,
        )
        return self.clean_text(text)

    def extract_pdf(self, pdf_path: Path) -> ExtractedDocument:
        assert fitz is not None
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF file: {pdf_path}")

        source_hash = hash_file(pdf_path)
        text_parts: list[str] = []
        ocr_pages = 0

        with fitz.open(str(pdf_path)) as document:
            page_count = len(document)
            for page in document:
                extracted = self.clean_text(page.get_text("text"))
                if len(extracted) >= self.settings.ocr_min_text_chars:
                    text_parts.append(extracted)
                    continue

                ocr_text = self._ocr_page(page)
                if ocr_text:
                    text_parts.append(ocr_text)
                ocr_pages += 1

        full_text = self.clean_text("\n\n".join(text_parts))
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
