from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Split documents into retrieval-friendly chunks."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Iterable[str],
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            raise ValueError("Không có tài liệu để chunk.")

        chunks = self._splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("chunk_id", idx)
            chunk.metadata = metadata

        return chunks
