from pathlib import Path

from rag_sgu.config import load_settings


def test_load_settings_reads_env_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHUNK_SIZE=512\n"
        "CHUNK_OVERLAP=64\n"
        "RETRIEVAL_K=7\n"
        "RAG_PDF_DIR=data/pdfs\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("RETRIEVAL_K", raising=False)
    monkeypatch.delenv("RAG_PDF_DIR", raising=False)

    settings = load_settings(base_dir=tmp_path)

    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.retrieval_k == 7
    assert settings.pdf_dir == tmp_path / "data" / "pdfs"
    assert settings.vector_store_dir == tmp_path / "vector_store"


def test_safe_dict_masks_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")
    settings = load_settings(base_dir=tmp_path)

    payload = settings.safe_dict()

    assert payload["google_api_key"] == "***"
    assert payload["base_dir"] == str(Path(tmp_path).resolve())
