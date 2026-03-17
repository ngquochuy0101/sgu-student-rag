from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _resolve_path(value: str, base_dir: Path) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    return (base_dir / raw_path).resolve()


@dataclass(frozen=True)
class RAGConfig:
    base_dir: Path
    pdf_dir: Path
    vector_store_dir: Path
    db_path: Path
    pdf_glob_pattern: str
    chunk_size: int
    chunk_overlap: int
    separators: Tuple[str, ...]
    embedding_model: str
    embedding_device: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_api_transport: str
    retrieval_k: int

    @staticmethod
    def from_env(base_dir: Path | None = None) -> "RAGConfig":
        root_dir = (base_dir or Path.cwd()).resolve()

        return RAGConfig(
            base_dir=root_dir,
            pdf_dir=_resolve_path(os.getenv("RAG_PDF_DIR", "File_PDFs_OCR"), root_dir),
            vector_store_dir=_resolve_path(
                os.getenv("RAG_VECTOR_STORE_DIR", "vector_store"),
                root_dir,
            ),
            db_path=_resolve_path(
                os.getenv("RAG_DEMO_DB_PATH", "artifacts/rag_demo.db"),
                root_dir,
            ),
            pdf_glob_pattern=os.getenv("RAG_PDF_GLOB", "*.pdf"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1600")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            separators=("\n\n", "\n", ". ", " ", ""),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            ),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            llm_api_transport=os.getenv("LLM_API_TRANSPORT", "rest"),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        )
