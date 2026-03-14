from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractedDocument:
    source_path: Path
    text: str
    page_count: int
    ocr_pages: int
    extraction_method: str
    source_hash: str
    page_texts: list[str] = field(default_factory=list)

    def to_cache_payload(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "text": self.text,
            "page_count": self.page_count,
            "ocr_pages": self.ocr_pages,
            "extraction_method": self.extraction_method,
            "source_hash": self.source_hash,
            "page_texts": self.page_texts,
        }

    @classmethod
    def from_cache_payload(cls, payload: dict[str, Any]) -> "ExtractedDocument":
        cached_page_texts = payload.get("page_texts", [])
        if not isinstance(cached_page_texts, list):
            cached_page_texts = []

        text = str(payload["text"])
        page_count = int(payload["page_count"])
        page_texts: list[str] = [str(item) for item in cached_page_texts if str(item).strip()]
        if not page_texts and text.strip():
            page_texts = [text]

        return cls(
            source_path=Path(payload["source_path"]),
            text=text,
            page_count=page_count,
            ocr_pages=int(payload["ocr_pages"]),
            extraction_method=str(payload["extraction_method"]),
            source_hash=str(payload["source_hash"]),
            page_texts=page_texts,
        )
