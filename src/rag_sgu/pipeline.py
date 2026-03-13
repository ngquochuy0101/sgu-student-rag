from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RAGSettings
from .document_ingestion import DocumentIngestor, collect_pdf_files, total_ocr_pages
from .mlops import compute_dataset_fingerprint, read_json, utc_now_iso, write_json
from .text_chunking import TextChunker
from .vector_index import VectorIndexManager


@dataclass
class BuildResult:
    skipped: bool
    dataset_fingerprint: str
    documents_count: int
    chunks_count: int
    ocr_pages: int
    manifest_path: Path


class RAGBuildPipeline:
    def __init__(self, settings: RAGSettings, logger):
        self.settings = settings
        self.logger = logger
        self.ingestor = DocumentIngestor(settings=settings, logger=logger)
        self.chunker = TextChunker(settings=settings)
        self.vector_manager = VectorIndexManager(settings=settings, logger=logger)

    def _is_existing_index_usable(self, fingerprint: str) -> bool:
        manifest = read_json(self.settings.manifest_path)
        if not manifest:
            return False

        same_fingerprint = manifest.get("dataset_fingerprint") == fingerprint
        has_files = (self.settings.vector_store_dir / "index.faiss").exists() and (
            self.settings.vector_store_dir / "index.pkl"
        ).exists()
        return bool(same_fingerprint and has_files)

    def build_index(self, pattern: str = "*.pdf", force: bool = False) -> BuildResult:
        pdf_files = collect_pdf_files(self.settings.pdf_dir, pattern=pattern)
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.settings.pdf_dir} with pattern {pattern}"
            )

        fingerprint = compute_dataset_fingerprint(pdf_files)
        if not force and self._is_existing_index_usable(fingerprint):
            self.logger.info("Index is fresh; skipping rebuild")
            manifest = read_json(self.settings.manifest_path) or {}
            return BuildResult(
                skipped=True,
                dataset_fingerprint=fingerprint,
                documents_count=int(manifest.get("documents_count", 0)),
                chunks_count=int(manifest.get("chunks_count", 0)),
                ocr_pages=int(manifest.get("ocr_pages", 0)),
                manifest_path=self.settings.manifest_path,
            )

        documents = self.ingestor.ingest_directory(pattern=pattern)
        chunks = self.chunker.chunk_documents(documents)

        index = self.vector_manager.build(chunks)
        self.vector_manager.save(index)

        manifest = {
            "generated_at": utc_now_iso(),
            "dataset_fingerprint": fingerprint,
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "ocr_pages": total_ocr_pages(documents),
            "pattern": pattern,
            "settings": self.settings.safe_dict(),
            "sources": [
                {
                    "name": item.source_path.name,
                    "hash": item.source_hash,
                    "page_count": item.page_count,
                    "ocr_pages": item.ocr_pages,
                    "extraction_method": item.extraction_method,
                    "char_count": len(item.text),
                }
                for item in documents
            ],
        }
        write_json(self.settings.manifest_path, manifest)

        return BuildResult(
            skipped=False,
            dataset_fingerprint=fingerprint,
            documents_count=len(documents),
            chunks_count=len(chunks),
            ocr_pages=total_ocr_pages(documents),
            manifest_path=self.settings.manifest_path,
        )
