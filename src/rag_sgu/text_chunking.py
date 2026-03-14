from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
        def __init__(
            self,
            chunk_size: int,
            chunk_overlap: int,
            separators: list[str] | None = None,
            length_function=len,
        ):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []
            chunks: list[str] = []
            start = 0
            text_length = self.length_function(text)
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunks.append(text[start:end])
                if end >= text_length:
                    break
                start = max(0, end - self.chunk_overlap)
            return chunks

try:
    from langchain_core.documents import Document  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    try:
        from langchain.schema import Document  # type: ignore[import-not-found,no-redef]
    except ImportError:  # pragma: no cover
        @dataclass
        class Document:  # type: ignore[no-redef]
            page_content: str
            metadata: dict

from .config import RAGSettings
from .schemas import ExtractedDocument


class TextChunker:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=list(settings.separators),
            length_function=len,
        )

    def chunk_documents(self, documents: List[ExtractedDocument]) -> List[Document]:
        chunks: List[Document] = []
        for item in documents:
            try:
                source_relpath = str(item.source_path.resolve().relative_to(self.settings.base_dir))
            except ValueError:
                source_relpath = item.source_path.name

            chunk_id = 0
            page_texts = item.page_texts if item.page_texts else [item.text]

            for page_number, page_text in enumerate(page_texts, start=1):
                if not page_text.strip():
                    continue
                item_chunks = self.splitter.split_text(page_text)
                for page_chunk_id, chunk in enumerate(item_chunks):
                    metadata = {
                        "source": item.source_path.name,
                        "source_relpath": source_relpath,
                        "source_path": str(item.source_path),
                        "source_hash": item.source_hash,
                        "chunk_id": chunk_id,
                        "page_chunk_id": page_chunk_id,
                        "page_number": page_number,
                        "page_count": item.page_count,
                        "ocr_pages": item.ocr_pages,
                        "extraction_method": item.extraction_method,
                    }
                    chunks.append(Document(page_content=chunk, metadata=metadata))
                    chunk_id += 1
        return chunks
