import logging
import time

import pytest

from rag_sgu.config import load_settings
from rag_sgu.pipeline import BuildResult
from rag_sgu.web.service import WebRAGService
import rag_sgu.web.service as web_service_module


def test_load_manifest_returns_none_when_missing(tmp_path):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    assert service.load_manifest() is None


def test_build_index_resets_runtime_cache(tmp_path, monkeypatch):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    service._vector_store = object()
    service._qa_system = object()  # type: ignore[assignment]

    expected_result = BuildResult(
        skipped=False,
        dataset_fingerprint="abc123",
        documents_count=2,
        chunks_count=8,
        ocr_pages=1,
        manifest_path=settings.manifest_path,
    )

    class FakePipeline:
        def __init__(self, settings, logger):
            self.settings = settings
            self.logger = logger

        def build_index(self, pattern: str = "*.pdf", force: bool = False) -> BuildResult:
            assert pattern == "*.pdf"
            assert force is True
            return expected_result

    monkeypatch.setattr(web_service_module, "RAGBuildPipeline", FakePipeline)

    result = service.build_index(pattern="*.pdf", force=True)

    assert result == expected_result
    assert service._vector_store is None
    assert service._qa_system is None


def test_query_uses_cached_runtime_components(tmp_path, monkeypatch):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    counters = {"load": 0, "qa_init": 0, "query": 0}

    class FakeVectorIndexManager:
        def __init__(self, settings, logger):
            self.settings = settings
            self.logger = logger

        def load(self):
            counters["load"] += 1
            return "vector-store"

    class FakeQASystem:
        def __init__(self, settings, logger):
            self.settings = settings
            self.logger = logger
            counters["qa_init"] += 1

        def query(self, question: str, vector_store, top_k: int | None = None):
            counters["query"] += 1
            return {
                "question": question,
                "answer": f"answer::{question}",
                "sources": [{"source": "sample.pdf", "chunk_id": 0, "preview": "..."}],
                "vector_store": vector_store,
                "top_k": top_k,
            }

        def batch_query(self, questions: list[str], vector_store, top_k: int | None = None):
            return [
                {"question": q, "answer": "ok", "sources": [], "vector_store": vector_store}
                for q in questions
            ]

    monkeypatch.setattr(web_service_module, "VectorIndexManager", FakeVectorIndexManager)
    monkeypatch.setattr(web_service_module, "RAGQASystem", FakeQASystem)

    first = service.query("What is SGU?", top_k=3)
    second = service.query("What is OCR?", top_k=3)

    assert first["vector_store"] == "vector-store"
    assert second["vector_store"] == "vector-store"
    assert counters["load"] == 1
    assert counters["qa_init"] == 1
    assert counters["query"] == 2


def test_query_rejects_blank_question(tmp_path):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    with pytest.raises(ValueError):
        service.query("   ")


def test_batch_query_rejects_empty_questions(tmp_path):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    with pytest.raises(ValueError):
        service.batch_query(["", "   "])


def test_validate_email_respects_allowed_domain(tmp_path, monkeypatch):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    monkeypatch.setenv("WEB_ALLOWED_EMAIL_DOMAINS", "sgu.edu.vn")

    is_valid, _ = service.validate_email("student@sgu.edu.vn")
    is_invalid, _ = service.validate_email("outsider@gmail.com")

    assert is_valid is True
    assert is_invalid is False


def test_check_rate_limit_blocks_after_limit(tmp_path, monkeypatch):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    monkeypatch.setenv("WEB_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("WEB_RATE_LIMIT_WINDOW_SECONDS", "3600")

    first = service.check_rate_limit("student@sgu.edu.vn")
    second = service.check_rate_limit("student@sgu.edu.vn")
    third = service.check_rate_limit("student@sgu.edu.vn")

    assert first[0] is True
    assert second[0] is True
    assert third[0] is False
    assert third[2] > 0


def test_start_query_returns_cache_hit_when_semantic_match(tmp_path, monkeypatch):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    (settings.vector_store_dir / "index.faiss").write_bytes(b"")
    (settings.vector_store_dir / "index.pkl").write_bytes(b"")

    monkeypatch.setenv("WEB_SEMANTIC_CACHE_THRESHOLD", "0.95")
    monkeypatch.setattr(service, "_embed_text", lambda _text: [1.0, 0.0, 0.0])

    service._semantic_cache = [
        {
            "question": "cached question",
            "answer": "cached answer",
            "sources": [{"source": "sample.pdf", "chunk_id": 0, "preview": "..."}],
            "embedding": [1.0, 0.0, 0.0],
            "user_email": "student@sgu.edu.vn",
            "collection": "Kho chung",
            "created_unix": time.time(),
            "created_at": "now",
        }
    ]

    handle = service.start_query(
        question="cached question",
        user_email="student@sgu.edu.vn",
        collection="Kho chung",
        conversation_key="student@sgu.edu.vn::Kho chung",
        top_k=3,
    )

    assert handle.cache_hit is True
    assert handle.cached_answer == "cached answer"
    assert handle.stream_session is None


def test_memory_summary_keeps_recent_window(tmp_path, monkeypatch):
    settings = load_settings(base_dir=tmp_path)
    service = WebRAGService(settings=settings, logger=logging.getLogger("test"))

    class FakeQASystem:
        def summarize_expired_turns(self, existing_summary: str, expired_turns):
            return f"{existing_summary}|expired={len(expired_turns)}"

    monkeypatch.setenv("WEB_MEMORY_WINDOW", "2")
    monkeypatch.setattr(service, "_get_qa_system", lambda: FakeQASystem())

    service._append_turn_to_memory("conv", "Q1", "A1")
    service._append_turn_to_memory("conv", "Q2", "A2")
    service._append_turn_to_memory("conv", "Q3", "A3")

    summary, recent_turns = service.get_memory_context("conv")

    assert len(recent_turns) == 2
    assert recent_turns[0]["question"] == "Q2"
    assert recent_turns[1]["question"] == "Q3"
    assert "expired=1" in summary
