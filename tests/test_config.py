from pathlib import Path

from rag_sgu.config import load_settings


def test_load_settings_reads_env_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHUNK_SIZE=1000\n"
        "CHUNK_OVERLAP=200\n"
        "RETRIEVAL_K=3\n"
        "RAG_PDF_DIR=data/pdfs\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("RETRIEVAL_K", raising=False)
    monkeypatch.delenv("RAG_PDF_DIR", raising=False)

    settings = load_settings(base_dir=tmp_path)

    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_k == 3
    assert settings.pdf_dir == tmp_path / "data" / "pdfs"
    assert settings.vector_store_dir == tmp_path / "vector_store"


def test_safe_dict_masks_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")
    settings = load_settings(base_dir=tmp_path)

    payload = settings.safe_dict()

    assert payload["google_api_key"] == "***"
    assert payload["base_dir"] == str(Path(tmp_path).resolve())


def test_load_settings_notebook_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("RETRIEVAL_K", raising=False)
    monkeypatch.delenv("OCR_PREPROCESSING_ENABLED", raising=False)
    monkeypatch.delenv("OCR_ADAPTIVE_THRESHOLD", raising=False)
    monkeypatch.delenv("OCR_DENOISE", raising=False)
    monkeypatch.delenv("OCR_DESKEW", raising=False)
    monkeypatch.delenv("OCR_VIETNAMESE_CORRECTION", raising=False)
    monkeypatch.delenv("OCR_CORRECTION_AGGRESSIVE", raising=False)
    monkeypatch.delenv("OCR_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("OCR_MAX_RETRY_ATTEMPTS", raising=False)

    settings = load_settings(base_dir=tmp_path)

    assert settings.retrieval_k == 3
    assert settings.ocr_preprocessing_enabled is True
    assert settings.ocr_adaptive_threshold is True
    assert settings.ocr_denoise is True
    assert settings.ocr_deskew is True
    assert settings.ocr_vietnamese_correction is True
    assert settings.ocr_correction_aggressive is False
    assert settings.ocr_confidence_threshold == 0.60
    assert settings.ocr_max_retry_attempts == 2
