from __future__ import annotations

from typing import Any, List

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-not-found]
    from langchain_community.vectorstores import FAISS  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    HuggingFaceEmbeddings = None  # type: ignore[assignment]
    FAISS = None  # type: ignore[assignment]

from .config import RAGSettings


class VectorIndexManager:
    def __init__(self, settings: RAGSettings, logger):
        if HuggingFaceEmbeddings is None or FAISS is None:
            raise ImportError(
                "Vector index dependencies are missing. Install requirements.txt before indexing."
            )
        self.settings = settings
        self.logger = logger
        self._embeddings: Any | None = None

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            self.logger.info("Loading embedding model: %s", self.settings.embedding_model)
            assert HuggingFaceEmbeddings is not None
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": self.settings.embedding_device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def build(self, documents: List[Any]) -> Any:
        if not documents:
            raise ValueError("No chunks found. Cannot build vector index.")
        self.logger.info("Building FAISS index from %s chunks", len(documents))
        assert FAISS is not None
        return FAISS.from_documents(documents=documents, embedding=self.embeddings)

    def save(self, index: Any) -> None:
        self.settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
        index.save_local(str(self.settings.vector_store_dir))
        self.logger.info("Saved FAISS index to %s", self.settings.vector_store_dir)

    def load(self) -> Any:
        assert FAISS is not None
        faiss_file = self.settings.vector_store_dir / "index.faiss"
        pkl_file = self.settings.vector_store_dir / "index.pkl"
        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                "Vector index not found. Run build-index before query/evaluate."
            )

        self.logger.info("Loading FAISS index from %s", self.settings.vector_store_dir)
        return FAISS.load_local(
            str(self.settings.vector_store_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
