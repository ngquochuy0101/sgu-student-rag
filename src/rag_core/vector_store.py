from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import RAGConfig
from .environment import configure_runtime_environment

configure_runtime_environment()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """Handle embedding initialization and FAISS lifecycle."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._embeddings = None

    def _init_embeddings(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._init_embeddings()
        return self._embeddings

    def build(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("Không có chunks để tạo vector store.")
        return FAISS.from_documents(documents=documents, embedding=self.embeddings)

    def save(self, vector_store: FAISS, path: Optional[Path] = None) -> Path:
        target = (path or self.config.vector_store_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(target))
        return target

    def load(self, path: Optional[Path] = None) -> FAISS:
        target = (path or self.config.vector_store_dir).resolve()
        index_path = target / "index.faiss"
        store_path = target / "index.pkl"
        if not index_path.exists() or not store_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FAISS index tại {target}. Hãy build index trước khi truy vấn."
            )

        return FAISS.load_local(
            str(target),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def build_and_save(self, documents: List[Document], path: Optional[Path] = None) -> FAISS:
        vector_store = self.build(documents)
        self.save(vector_store, path=path)
        return vector_store
