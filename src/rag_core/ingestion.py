from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document


@dataclass(frozen=True)
class IngestionResult:
    documents: List[Document]
    scanned_pdf_files: List[str]
    loaded_pdf_files: List[str]


class PDFIngestionService:
    """Ingest PDF files using only DirectoryLoader + PyPDFLoader."""

    def __init__(
        self,
        pdf_dir: Path,
        pattern: str = "*.pdf",
        use_multithreading: bool = True,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.pattern = pattern
        self.use_multithreading = use_multithreading

    @staticmethod
    def clean_text(text: str) -> str:
        normalized = unicodedata.normalize("NFC", text or "")
        return re.sub(r"\s+", " ", normalized).strip()

    def _build_loader(self) -> DirectoryLoader:
        return DirectoryLoader(
            str(self.pdf_dir),
            glob=self.pattern,
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=self.use_multithreading,
            loader_kwargs={"extract_images": False},
        )

    def ingest(self) -> IngestionResult:
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục PDF: {self.pdf_dir}")

        pdf_files = sorted(self.pdf_dir.glob(self.pattern))
        if not pdf_files:
            raise FileNotFoundError(
                f"Không tìm thấy file PDF theo pattern '{self.pattern}' trong {self.pdf_dir}"
            )

        raw_docs = self._build_loader().load()
        source_stats: Dict[str, Dict[str, int]] = {}
        cleaned_docs: List[Document] = []

        for doc in raw_docs:
            metadata = dict(doc.metadata or {})
            source_value = str(metadata.get("source") or "unknown.pdf").strip()
            source_name = Path(source_value).name or "unknown.pdf"

            if source_name not in source_stats:
                source_stats[source_name] = {"total_pages": 0, "non_empty_pages": 0}

            source_stats[source_name]["total_pages"] += 1

            cleaned_content = self.clean_text(str(doc.page_content or ""))
            if not cleaned_content:
                continue

            source_stats[source_name]["non_empty_pages"] += 1
            metadata["source"] = source_value
            cleaned_docs.append(Document(page_content=cleaned_content, metadata=metadata))

        loaded_pdf_files = sorted(source_stats.keys())
        scanned_pdf_files = sorted(
            file_name
            for file_name, stats in source_stats.items()
            if stats["non_empty_pages"] == 0
        )

        if not cleaned_docs:
            details = ", ".join(scanned_pdf_files) if scanned_pdf_files else "không rõ tên file"
            raise ValueError(
                "Không trích xuất được text-layer từ PDF. "
                "Các file có thể là PDF scan/image-only và chưa được hỗ trợ OCR trong pipeline này. "
                f"Danh sách nghi vấn: {details}."
            )

        return IngestionResult(
            documents=cleaned_docs,
            scanned_pdf_files=scanned_pdf_files,
            loaded_pdf_files=loaded_pdf_files,
        )
