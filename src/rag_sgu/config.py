from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _resolve_path(base_dir: Path, env_name: str, fallback_relative: str) -> Path:
    raw_value = os.getenv(env_name)
    if raw_value:
        raw_path = Path(raw_value)
        return raw_path if raw_path.is_absolute() else (base_dir / raw_path)
    return base_dir / fallback_relative


@dataclass
class RAGSettings:
    base_dir: Path
    pdf_dir: Path
    vector_store_dir: Path
    ocr_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    cache_dir: Path
    eval_dir: Path

    tesseract_cmd: str | None
    tessdata_prefix: str | None
    ocr_languages: str
    ocr_dpi: int
    ocr_psm: int
    ocr_oem: int
    ocr_min_text_chars: int
    ocr_cache_enabled: bool

    chunk_size: int
    chunk_overlap: int
    separators: tuple[str, ...]

    embedding_model: str
    embedding_device: str
    retrieval_k: int

    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    google_api_key: str | None

    run_name: str

    @property
    def manifest_path(self) -> Path:
        return self.vector_store_dir / "manifest.json"

    def ensure_directories(self) -> None:
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in [
            "base_dir",
            "pdf_dir",
            "vector_store_dir",
            "ocr_dir",
            "artifacts_dir",
            "logs_dir",
            "cache_dir",
            "eval_dir",
        ]:
            data[key] = str(data[key])
        data["google_api_key"] = "***" if self.google_api_key else None
        data["separators"] = list(self.separators)
        return data


def load_settings(base_dir: str | Path | None = None, env_file: str | Path | None = None) -> RAGSettings:
    resolved_base = Path(base_dir or os.getenv("RAG_BASE_DIR") or Path.cwd()).resolve()

    if env_file is not None:
        env_path = Path(env_file)
        env_path = env_path if env_path.is_absolute() else (resolved_base / env_path)
    else:
        env_path = resolved_base / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=False)

    pdf_dir = _resolve_path(resolved_base, "RAG_PDF_DIR", "File_PDFs")
    vector_store_dir = _resolve_path(resolved_base, "RAG_VECTOR_STORE_DIR", "vector_store")
    ocr_dir = _resolve_path(resolved_base, "RAG_OCR_DIR", "OCR")
    artifacts_dir = _resolve_path(resolved_base, "RAG_ARTIFACTS_DIR", "artifacts")

    logs_dir = artifacts_dir / "logs"
    cache_dir = artifacts_dir / "cache" / "ocr"
    eval_dir = artifacts_dir / "evaluations"

    tesseract_path = os.getenv("TESSERACT_CMD")
    if not tesseract_path:
        local_tesseract = ocr_dir / "tesseract.exe"
        tesseract_path = str(local_tesseract) if local_tesseract.exists() else None

    tessdata_prefix = os.getenv("TESSDATA_PREFIX")
    if not tessdata_prefix:
        local_tessdata = ocr_dir / "tessdata"
        tessdata_prefix = str(local_tessdata) if local_tessdata.exists() else None

    settings = RAGSettings(
        base_dir=resolved_base,
        pdf_dir=pdf_dir,
        vector_store_dir=vector_store_dir,
        ocr_dir=ocr_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        cache_dir=cache_dir,
        eval_dir=eval_dir,
        tesseract_cmd=tesseract_path,
        tessdata_prefix=tessdata_prefix,
        ocr_languages=os.getenv("OCR_LANGUAGES", "vie+eng"),
        ocr_dpi=_env_int("OCR_DPI", 2),
        ocr_psm=_env_int("OCR_PSM", 3),
        ocr_oem=_env_int("OCR_OEM", 1),
        ocr_min_text_chars=_env_int("OCR_MIN_TEXT_CHARS", 60),
        ocr_cache_enabled=_env_bool("OCR_CACHE_ENABLED", True),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        separators=("\n\n", "\n", ". ", " ", ""),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        retrieval_k=_env_int("RETRIEVAL_K", 4),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        run_name=os.getenv("RAG_RUN_NAME", "default"),
    )
    settings.ensure_directories()
    return settings
