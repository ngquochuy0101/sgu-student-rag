"""
ocr_pdf.py — OCR tối ưu cho tài liệu tiếng Việt
=================================================
Chuyển PDF scan/image-only thành PDF có text-layer searchable.

Cải tiến so với pipeline cũ:
  • Tiền xử lý ảnh (CLAHE, Adaptive Threshold, Deskew, Denoise)
  • Tham số Tesseract tối ưu cho tiếng Việt (OEM 1, PSM 6)
  • DPI cao hơn (400) để giữ dấu
  • Post-correction bằng Gemini (tùy chọn)

Yêu cầu:
  pip install pymupdf pytesseract Pillow pypdf opencv-python

Sử dụng:
  python ocr_pdf.py
  python ocr_pdf.py --input File_PDFs --output File_PDFs_OCR --dpi 400
  python ocr_pdf.py --use-llm-correction        # bật sửa dấu bằng Gemini
  python ocr_pdf.py --no-preprocessing           # tắt preprocessing (debug)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from pypdf import PdfReader, PdfWriter

# ---------------------------------------------------------------------------
# Tesseract path resolution
# ---------------------------------------------------------------------------

_COMMON_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def resolve_tesseract_cmd(tesseract_cmd: Optional[str] = None) -> str:
    """Tìm tesseract.exe từ input → PATH → vị trí cài mặc định Windows."""
    if tesseract_cmd:
        candidate = Path(tesseract_cmd)
        if not candidate.exists():
            raise FileNotFoundError(f"Không tìm thấy tesseract.exe tại: {candidate}")
        return str(candidate)

    in_path = shutil.which("tesseract")
    if in_path:
        return in_path

    for candidate in _COMMON_TESSERACT_PATHS:
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "Không tìm thấy Tesseract. Cài Tesseract OCR hoặc truyền --tesseract-cmd."
    )


# ---------------------------------------------------------------------------
# Image preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_image(img: np.ndarray, *, denoise: bool = True) -> np.ndarray:
    """
    Tiền xử lý ảnh trước khi OCR:
      1. Grayscale
      2. CLAHE — tăng tương phản cục bộ (giữ dấu tiếng Việt rõ hơn)
      3. Adaptive Gaussian Threshold — binarize tốt hơn Otsu
      4. Denoise nhẹ (morphological open) — giảm noise, giữ dấu nhỏ
    """
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. CLAHE — Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Adaptive Gaussian Threshold
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )

    # 4. Denoise nhẹ — morphological opening với kernel nhỏ
    if denoise:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Sửa ảnh bị nghiêng (deskew) dựa trên minAreaRect của contours."""
    coords = np.column_stack(np.where(img < 128))
    if len(coords) < 50:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    # Chỉnh angle về khoảng [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Chỉ deskew nếu góc nghiêng đáng kể
    if abs(angle) < 0.5:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ---------------------------------------------------------------------------
# LLM post-correction (optional)
# ---------------------------------------------------------------------------


def correct_ocr_text_with_llm(text: str, *, api_key: str, model: str = "gemini-2.5-flash") -> str:
    """Dùng Gemini để sửa lỗi OCR tiếng Việt (chủ yếu phục hồi dấu)."""
    if not text.strip():
        return text

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = (
            "Bạn là chuyên gia sửa lỗi OCR tiếng Việt. "
            "Đoạn text dưới đây được trích xuất từ PDF bằng Tesseract OCR và bị mất dấu tiếng Việt. "
            "Hãy khôi phục lại dấu tiếng Việt chính xác, giữ nguyên cấu trúc và nội dung gốc. "
            "CHỈ trả về text đã sửa, KHÔNG thêm giải thích.\n\n"
            f"TEXT OCR:\n{text}"
        )

        response = client.models.generate_content(model=model, contents=prompt)
        corrected = response.text.strip()
        return corrected if corrected else text

    except Exception as exc:
        print(f"  ⚠️ LLM correction failed: {exc}")
        return text


# ---------------------------------------------------------------------------
# OCR single PDF
# ---------------------------------------------------------------------------

# Tesseract config tối ưu cho tiếng Việt
TESSERACT_CONFIG = r"--oem 1 --psm 6"


def ocr_single_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    lang: str = "vie+eng",
    dpi: int = 400,
    tesseract_cmd: Optional[str] = None,
    enable_preprocessing: bool = True,
    enable_deskew: bool = True,
    enable_llm_correction: bool = False,
    llm_api_key: str = "",
    llm_model: str = "gemini-2.5-flash",
    verbose: bool = True,
) -> Path:
    """OCR 1 file PDF → PDF mới có text layer searchable."""
    resolved_tesseract = resolve_tesseract_cmd(tesseract_cmd)
    pytesseract.pytesseract.tesseract_cmd = resolved_tesseract

    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_pdf}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    src_doc = fitz.open(str(input_pdf))
    writer = PdfWriter()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    total_pages = len(src_doc)

    try:
        for page_index in range(total_pages):
            page = src_doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # Chuyển pixmap → numpy array cho preprocessing
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(BytesIO(img_bytes))
            img_np = np.array(pil_img)

            if enable_preprocessing:
                processed = preprocess_image(img_np)
                if enable_deskew:
                    processed = deskew_image(processed)
                ocr_input = Image.fromarray(processed)
            else:
                ocr_input = pil_img

            # OCR → PDF page có text layer
            ocr_pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                ocr_input,
                extension="pdf",
                lang=lang,
                config=TESSERACT_CONFIG,
            )
            ocr_reader = PdfReader(BytesIO(ocr_pdf_bytes))
            writer.add_page(ocr_reader.pages[0])

            if verbose:
                progress = f"[{page_index + 1}/{total_pages}]"
                print(f"  📄 {progress} Trang {page_index + 1} xong", end="\r")

        # LLM post-correction: extract text → correct → overlay
        # Note: LLM corrects the extracted text but the PDF text layer
        # is already embedded by Tesseract. This correction is applied
        # to a separate clean-text output for the RAG pipeline.
        if enable_llm_correction and llm_api_key:
            if verbose:
                print(f"\n  🤖 Đang sửa lỗi OCR bằng Gemini ({llm_model})...")

            # Extract text from OCR'd PDF for correction
            corrected_text_path = output_pdf.with_suffix(".corrected.txt")
            all_text_parts = []
            temp_reader = PdfReader(BytesIO(b""))

            # Re-read from writer pages
            temp_buffer = BytesIO()
            writer.write(temp_buffer)
            temp_buffer.seek(0)
            temp_reader = PdfReader(temp_buffer)

            for pg in temp_reader.pages:
                page_text = pg.extract_text() or ""
                if page_text.strip():
                    all_text_parts.append(page_text)

            raw_text = "\n\n".join(all_text_parts)

            # Chia text thành chunks nhỏ để Gemini xử lý tốt hơn
            chunk_size = 3000
            chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
            corrected_chunks = []

            for i, chunk in enumerate(chunks):
                if verbose:
                    print(f"  🔧 Sửa chunk {i + 1}/{len(chunks)}...", end="\r")
                corrected = correct_ocr_text_with_llm(
                    chunk, api_key=llm_api_key, model=llm_model
                )
                corrected_chunks.append(corrected)

            corrected_text = "\n\n".join(corrected_chunks)
            corrected_text_path.write_text(corrected_text, encoding="utf-8")
            if verbose:
                print(f"\n  ✅ Text đã sửa lưu tại: {corrected_text_path}")

        with output_pdf.open("wb") as f:
            writer.write(f)
    finally:
        src_doc.close()

    if verbose:
        print(f"\n  ✅ OCR xong: {input_pdf.name} → {output_pdf}")

    return output_pdf


# ---------------------------------------------------------------------------
# OCR folder
# ---------------------------------------------------------------------------


def ocr_pdf_folder(
    input_dir: str | Path = "File_PDFs",
    output_dir: str | Path = "File_PDFs_OCR",
    *,
    pattern: str = "*.pdf",
    lang: str = "vie+eng",
    dpi: int = 400,
    tesseract_cmd: Optional[str] = None,
    enable_preprocessing: bool = True,
    enable_deskew: bool = True,
    enable_llm_correction: bool = False,
    llm_api_key: str = "",
    llm_model: str = "gemini-2.5-flash",
) -> list[Path]:
    """OCR tất cả PDF trong thư mục input → thư mục output."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {input_dir}")

    resolved_tesseract = resolve_tesseract_cmd(tesseract_cmd)
    print(f"🔎 Tesseract: {resolved_tesseract}")
    print(f"📂 Input:     {input_dir}")
    print(f"📁 Output:    {output_dir}")
    print(f"⚙️  DPI: {dpi} | Preprocessing: {enable_preprocessing} | Deskew: {enable_deskew}")
    if enable_llm_correction:
        print(f"🤖 LLM correction: ON ({llm_model})")

    pdf_files = sorted(input_dir.glob(pattern))
    if not pdf_files:
        print(f"⚠️ Không tìm thấy PDF nào trong: {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    print(f"📄 Tổng số file: {len(pdf_files)}\n")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
        out_path = output_dir / f"{pdf_path.stem}_ocr.pdf"
        try:
            created = ocr_single_pdf(
                input_pdf=pdf_path,
                output_pdf=out_path,
                lang=lang,
                dpi=dpi,
                tesseract_cmd=resolved_tesseract,
                enable_preprocessing=enable_preprocessing,
                enable_deskew=enable_deskew,
                enable_llm_correction=enable_llm_correction,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                verbose=True,
            )
            created_files.append(created)
        except Exception as exc:
            print(f"  ❌ Lỗi OCR {pdf_path.name}: {exc}")

    print(f"\n🎉 Hoàn tất OCR. Tạo được {len(created_files)}/{len(pdf_files)} file.")
    return created_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR tối ưu cho tài liệu tiếng Việt — tạo PDF có text-layer searchable",
    )
    parser.add_argument("--input", default="File_PDFs", help="Thư mục chứa PDF gốc (mặc định: File_PDFs)")
    parser.add_argument("--output", default="File_PDFs_OCR", help="Thư mục lưu PDF OCR (mặc định: File_PDFs_OCR)")
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern cho PDF (mặc định: *.pdf)")
    parser.add_argument("--lang", default="vie+eng", help="Ngôn ngữ Tesseract (mặc định: vie+eng)")
    parser.add_argument("--dpi", type=int, default=400, help="DPI render (mặc định: 400)")
    parser.add_argument("--tesseract-cmd", default=None, help="Đường dẫn tesseract.exe")
    parser.add_argument("--no-preprocessing", action="store_true", help="Tắt tiền xử lý ảnh")
    parser.add_argument("--no-deskew", action="store_true", help="Tắt deskew")
    parser.add_argument("--use-llm-correction", action="store_true", help="Bật sửa lỗi OCR bằng Gemini")
    parser.add_argument("--llm-model", default="gemini-2.5-flash", help="Model Gemini cho post-correction")

    args = parser.parse_args()

    # Load API key from .env if available
    llm_api_key = ""
    if args.use_llm_correction:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        llm_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not llm_api_key or llm_api_key == "YOUR_API_KEY_HERE":
            print("⚠️ Cần GOOGLE_API_KEY trong .env để dùng LLM correction. Tắt LLM correction.")
            args.use_llm_correction = False

    created = ocr_pdf_folder(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        lang=args.lang,
        dpi=args.dpi,
        tesseract_cmd=args.tesseract_cmd,
        enable_preprocessing=not args.no_preprocessing,
        enable_deskew=not args.no_deskew,
        enable_llm_correction=args.use_llm_correction,
        llm_api_key=llm_api_key,
        llm_model=args.llm_model,
    )

    if created:
        print("\nDanh sách file OCR đã tạo:")
        for p in created:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
