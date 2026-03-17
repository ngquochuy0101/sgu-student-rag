from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from .chunking import TextChunker
from .config import RAGConfig
from .ingestion import IngestionResult, PDFIngestionService
from .qa_service import RAGService


@dataclass(frozen=True)
class BuildResult:
    ingestion: IngestionResult
    chunks: List[Document]


class RAGPipeline:
    """Minimal class-based pipeline for notebook and scripts."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.ingestion_service = PDFIngestionService(
            pdf_dir=config.pdf_dir,
            pattern=config.pdf_glob_pattern,
        )
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )
        self.rag_service = RAGService(config)

    def build_index(self) -> BuildResult:
        ingestion = self.ingestion_service.ingest()
        chunks = self.chunker.split_documents(ingestion.documents)
        vector_store = self.rag_service.vector_manager.build_and_save(chunks)
        self.rag_service.attach_vector_store(vector_store)
        return BuildResult(ingestion=ingestion, chunks=chunks)

    def load_index(self) -> None:
        vector_store = self.rag_service.vector_manager.load()
        self.rag_service.attach_vector_store(vector_store)

    def ensure_ready(self) -> None:
        if self.rag_service.retriever_ready:
            return
        self.load_index()

    def query(self, question: str):
        self.ensure_ready()
        return self.rag_service.query(question)
