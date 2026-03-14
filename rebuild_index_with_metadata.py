import hashlib
import json
import os
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from dotenv import load_dotenv

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_fingerprint(pdf_files: List[Path]) -> str:
    rows: List[Dict[str, Any]] = []
    for path in sorted(pdf_files, key=lambda item: item.name.casefold()):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
        )
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RebuildConfig:
    base_dir: Path
    pdf_dir: Path
    vector_store_dir: Path
    ocr_dir: Path
    embedding_model: str
    embedding_device: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    pdf_loader_extract_images: bool

    @staticmethod
    def from_env() -> "RebuildConfig":
        base_dir = Path.cwd().resolve()
        pdf_dir = (base_dir / os.getenv("RAG_PDF_DIR", "File_PDFs")).resolve()
        vector_store_dir = (base_dir / os.getenv("RAG_VECTOR_STORE_DIR", "vector_store")).resolve()
        ocr_dir = (base_dir / os.getenv("RAG_OCR_DIR", "OCR")).resolve()

        return RebuildConfig(
            base_dir=base_dir,
            pdf_dir=pdf_dir,
            vector_store_dir=vector_store_dir,
            ocr_dir=ocr_dir,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            ),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
            pdf_loader_extract_images=_as_bool(os.getenv("RAG_PDFLOADER_EXTRACT_IMAGES", "0")),
        )


def _prepend_path(path_value: Path) -> None:
    path_str = str(path_value)
    if not path_value.exists():
        return

    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if path_str not in parts:
        os.environ["PATH"] = path_str + os.pathsep + current


def _discover_poppler_bin() -> Path | None:
    local_app = Path(os.getenv("LOCALAPPDATA", ""))
    if not local_app.exists():
        return None

    winget_packages = local_app / "Microsoft" / "WinGet" / "Packages"
    if not winget_packages.exists():
        return None

    for exe in winget_packages.glob("oschwartz10612.Poppler_*/**/pdfinfo.exe"):
        return exe.parent
    return None


def _discover_tesseract_dir() -> Path | None:
    program_files = Path(os.getenv("ProgramFiles", r"C:\Program Files"))
    preferred = program_files / "Tesseract-OCR" / "tesseract.exe"
    if preferred.exists():
        return preferred.parent

    local_app = Path(os.getenv("LOCALAPPDATA", ""))
    winget_packages = local_app / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.exists():
        for exe in winget_packages.glob("**/tesseract.exe"):
            return exe.parent
    return None


def setup_loader_runtime(config: RebuildConfig) -> None:
    # 1) Project-local OCR tools (if user keeps OCR folder in repo)
    tesseract_cmd = config.ocr_dir / "tesseract.exe"
    tessdata_dir = config.ocr_dir / "tessdata"
    if tesseract_cmd.exists():
        _prepend_path(tesseract_cmd.parent)
    if tessdata_dir.exists():
        os.environ.setdefault("TESSDATA_PREFIX", str(tessdata_dir) + os.sep)

    # 2) System/winget Poppler and Tesseract (works even before shell restart)
    poppler_bin = _discover_poppler_bin()
    if poppler_bin is not None:
        _prepend_path(poppler_bin)

    tesseract_dir = _discover_tesseract_dir()
    if tesseract_dir is not None:
        _prepend_path(tesseract_dir)
        tessdata_from_system = tesseract_dir / "tessdata"
        if tessdata_from_system.exists():
            os.environ.setdefault("TESSDATA_PREFIX", str(tessdata_from_system) + os.sep)


def _coerce_page_number(metadata: Dict[str, Any]) -> int:
    raw_page_number = metadata.get("page_number")
    if raw_page_number is not None:
        try:
            return max(1, int(raw_page_number))
        except (TypeError, ValueError):
            pass

    raw_page_zero_based = metadata.get("page")
    if raw_page_zero_based is not None:
        try:
            return max(1, int(raw_page_zero_based) + 1)
        except (TypeError, ValueError):
            pass

    return 1


def _load_pdf_pages(pdf_file: Path, config: RebuildConfig) -> Tuple[List[Document], str]:
    attempts: List[Tuple[str, Any]] = []

    # OCR-first loader integration for scanned PDFs.
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader

        attempts.append(
            (
                "UnstructuredPDFLoader(ocr_only)",
                UnstructuredPDFLoader(
                    str(pdf_file),
                    mode="elements",
                    strategy="ocr_only",
                ),
            )
        )
    except Exception as exc:
        attempts.append(("UnstructuredPDFLoader(ocr_only)", exc))

    attempts.append(
        (
            "PyPDFLoader",
            PyPDFLoader(
                str(pdf_file),
                extract_images=config.pdf_loader_extract_images,
            ),
        )
    )

    best_docs: List[Document] = []
    best_loader = "PyPDFLoader"
    best_chars = -1
    errors: List[str] = []

    for loader_name, candidate in attempts:
        if isinstance(candidate, Exception):
            errors.append(f"{loader_name}: {type(candidate).__name__}: {candidate}")
            continue

        try:
            docs = candidate.load()
        except Exception as exc:
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
            continue

        total_chars = sum(
            len(clean_text(str(getattr(doc, "page_content", ""))))
            for doc in docs
        )

        if total_chars > best_chars:
            best_docs = docs
            best_loader = loader_name
            best_chars = total_chars

        if total_chars > 0:
            return docs, loader_name

    if best_chars >= 0:
        if best_chars == 0 and errors:
            print(f"Warning: all loaders produced empty text for {pdf_file.name}")
            print("Loader diagnostics:")
            for item in errors:
                print(f"  - {item}")
        return best_docs, best_loader

    error_message = " | ".join(errors) if errors else "unknown loader error"
    raise RuntimeError(f"All PDFLoader integrations failed for {pdf_file.name}: {error_message}")


def build_documents(config: RebuildConfig) -> Tuple[List[Document], List[Dict[str, Any]], int, List[Path]]:
    if not config.pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {config.pdf_dir}")

    pdf_files = sorted(config.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {config.pdf_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: List[Document] = []
    sources: List[Dict[str, Any]] = []

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        file_chunk_count = 0
        file_char_count = 0
        max_page_number = 0

        page_docs, loader_name = _load_pdf_pages(pdf_file, config)
        page_count = len(page_docs)

        try:
            source_relpath = str(pdf_file.resolve().relative_to(config.base_dir))
        except ValueError:
            source_relpath = pdf_file.name

        for page_doc in page_docs:
            page_text = clean_text(str(getattr(page_doc, "page_content", "")))
            if not page_text:
                continue

            metadata = getattr(page_doc, "metadata", {}) or {}
            page_number = _coerce_page_number(metadata)
            if page_number > max_page_number:
                max_page_number = page_number
            file_char_count += len(page_text)

            page_chunks = splitter.split_text(page_text)
            for page_chunk_id, chunk in enumerate(page_chunks):
                if not chunk.strip():
                    continue

                doc_metadata = {
                    "source": pdf_file.name,
                    "source_relpath": source_relpath,
                    "source_path": str(pdf_file.resolve()),
                    "page_number": page_number,
                    "page": page_number - 1,
                    "chunk_id": file_chunk_count,
                    "page_chunk_id": page_chunk_id,
                    "extraction_method": "pdf_loader",
                    "loader": loader_name,
                }
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
                file_chunk_count += 1

        source_hash = hash_file(pdf_file)
        sources.append(
            {
                "name": pdf_file.name,
                "hash": source_hash,
                "page_count": max_page_number if max_page_number > 0 else page_count,
                "ocr_pages": 0,
                "extraction_method": "pdf_loader",
                "loader": loader_name,
                "char_count": file_char_count,
                "chunk_count": file_chunk_count,
            }
        )

    if not documents:
        raise RuntimeError(
            "No chunks created from PDFs. For scanned PDFs, install 'unstructured[pdf]' and make sure both Tesseract and Poppler are available in PATH."
        )

    return documents, sources, 0, pdf_files


def backup_existing_index(vector_store_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = vector_store_dir.parent / f"{vector_store_dir.name}_backup_{timestamp}"
    if vector_store_dir.exists() and any(vector_store_dir.iterdir()):
        shutil.copytree(vector_store_dir, backup_dir)
    return backup_dir


def write_manifest(
    config: RebuildConfig,
    sources: List[Dict[str, Any]],
    chunks_count: int,
    total_ocr_pages: int,
    dataset_fingerprint: str,
) -> None:
    manifest_path = config.vector_store_dir / "manifest.json"
    payload = {
        "generated_at": utc_now_iso(),
        "dataset_fingerprint": dataset_fingerprint,
        "documents_count": len(sources),
        "chunks_count": chunks_count,
        "ocr_pages": total_ocr_pages,
        "pattern": "*.pdf",
        "settings": {
            "base_dir": str(config.base_dir),
            "pdf_dir": str(config.pdf_dir),
            "vector_store_dir": str(config.vector_store_dir),
            "ocr_dir": str(config.ocr_dir),
            "embedding_model": config.embedding_model,
            "embedding_device": config.embedding_device,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "retrieval_k": config.retrieval_k,
            "pdf_loader_primary": "UnstructuredPDFLoader(ocr_only)",
            "pdf_loader_fallback": "PyPDFLoader",
            "pdf_loader_extract_images": config.pdf_loader_extract_images,
        },
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    config = RebuildConfig.from_env()
    setup_loader_runtime(config)

    print("=" * 80)
    print("REBUILDING VECTOR INDEX WITH LANGCHAIN PDFLOADER INTEGRATION")
    print("=" * 80)
    print(f"PDF dir: {config.pdf_dir}")
    print(f"Vector store dir: {config.vector_store_dir}")

    documents, sources, total_ocr_pages, pdf_files = build_documents(config)
    print(f"Total chunks: {len(documents)}")

    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if config.vector_store_dir.exists():
        backup_dir = backup_existing_index(config.vector_store_dir)
        if backup_dir.exists():
            print(f"Backup created: {backup_dir}")

    config.vector_store_dir.mkdir(parents=True, exist_ok=True)
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    vector_store.save_local(str(config.vector_store_dir))

    dataset_fingerprint = compute_dataset_fingerprint(pdf_files)
    write_manifest(
        config=config,
        sources=sources,
        chunks_count=len(documents),
        total_ocr_pages=total_ocr_pages,
        dataset_fingerprint=dataset_fingerprint,
    )

    print("=" * 80)
    print("REBUILD COMPLETED")
    print("=" * 80)
    print(f"Index saved to: {config.vector_store_dir}")
    print(f"Manifest: {config.vector_store_dir / 'manifest.json'}")
    print("Sample source metadata from first chunk:")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()