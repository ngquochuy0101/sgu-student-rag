from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    import redis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    redis = None  # type: ignore[assignment]

from ..config import RAGSettings, load_settings
from ..document_ingestion import OCRProcessor
from ..logging_utils import configure_logging
from ..mlops import read_json, utc_now_iso
from ..pipeline import BuildResult, RAGBuildPipeline
from ..qa_service import RAGQASystem, StreamingAnswerSession
from ..vector_index import VectorIndexManager


def _normalize_path_key(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    try:
        return str(Path(raw_path).resolve()).casefold()
    except Exception:  # noqa: BLE001
        return str(raw_path).casefold()


def _safe_slug(value: str, fallback: str = "default") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return normalized or fallback


def _sanitize_filename(value: str) -> str:
    file_name = Path(value).name.strip()
    if not file_name:
        file_name = "uploaded.pdf"
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", file_name)
    if not sanitized.lower().endswith(".pdf"):
        sanitized = f"{sanitized}.pdf"
    return sanitized


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for idx in range(len(a)):
        va = float(a[idx])
        vb = float(b[idx])
        dot_product += va * vb
        norm_a += va * va
        norm_b += vb * vb

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass
class ConversationMemoryState:
    summary: str = ""
    recent_turns: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SemanticCacheMatch:
    similarity: float
    answer: str
    sources: list[dict[str, Any]]


@dataclass
class LiveQueryHandle:
    interaction_id: str
    question: str
    user_email: str
    collection: str
    conversation_key: str
    cache_hit: bool
    cache_similarity: float | None
    cached_answer: str | None
    cached_sources: list[dict[str, Any]]
    stream_session: StreamingAnswerSession | None
    started_at: float
    question_embedding: list[float]


class WebRAGService:
    def __init__(self, settings: RAGSettings, logger: logging.Logger):
        self.settings = settings
        self.logger = logger
        self._vector_store: Any | None = None
        self._qa_system: RAGQASystem | None = None

        web_cache_dir = self.settings.artifacts_dir / "cache" / "web"
        web_cache_dir.mkdir(parents=True, exist_ok=True)

        self._collections_path = web_cache_dir / "collections.json"
        self._semantic_cache_path = web_cache_dir / "semantic_cache.json"
        self._feedback_path = self.settings.eval_dir / "web_feedback.jsonl"
        self._trace_path = self.settings.logs_dir / "web_trace.jsonl"
        self._semantic_cache_redis_key = "rag_sgu:semantic_cache"

        redis_url = os.getenv("WEB_SEMANTIC_CACHE_REDIS_URL", "").strip()
        self._redis_client: Any | None = None
        if redis is not None and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                self._redis_client.ping()
                self.logger.info("Semantic cache backend: redis")
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Redis semantic cache disabled: %s", error)
                self._redis_client = None

        self._collections: dict[str, dict[str, list[str]]] = self._load_collections()
        self._semantic_cache: list[dict[str, Any]] = self._load_semantic_cache()
        self._conversation_states: dict[str, ConversationMemoryState] = {}
        self._rate_limit_state: dict[str, list[float]] = {}

    @property
    def index_ready(self) -> bool:
        index_path = self.settings.vector_store_dir / "index.faiss"
        metadata_path = self.settings.vector_store_dir / "index.pkl"
        return index_path.exists() and metadata_path.exists()

    @property
    def tracing_provider(self) -> str:
        if os.getenv("LANGSMITH_API_KEY"):
            return "langsmith"
        if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
            return "arize-phoenix"
        return "local-jsonl"

    def load_manifest(self) -> dict[str, Any] | None:
        payload = read_json(self.settings.manifest_path)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("Manifest payload must be a JSON object")
        return payload

    def build_index(self, pattern: str = "*.pdf", force: bool = False) -> BuildResult:
        pipeline = RAGBuildPipeline(settings=self.settings, logger=self.logger)
        result = pipeline.build_index(pattern=pattern, force=force)
        self.reset_runtime_cache()
        self.trace_event(
            "build_index",
            {
                "pattern": pattern,
                "force": force,
                "skipped": result.skipped,
                "documents_count": result.documents_count,
                "chunks_count": result.chunks_count,
                "ocr_pages": result.ocr_pages,
            },
        )
        return result

    def _get_vector_store(self, force_reload: bool = False) -> Any:
        if self._vector_store is None or force_reload:
            manager = VectorIndexManager(settings=self.settings, logger=self.logger)
            self._vector_store = manager.load()
        return self._vector_store

    def _get_qa_system(self) -> RAGQASystem:
        if self._qa_system is None:
            self._qa_system = RAGQASystem(settings=self.settings, logger=self.logger)
        return self._qa_system

    @staticmethod
    def _read_json_payload(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _write_json_payload(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_collections(self) -> dict[str, dict[str, list[str]]]:
        payload = self._read_json_payload(self._collections_path, default={})
        if not isinstance(payload, dict):
            return {}

        normalized: dict[str, dict[str, list[str]]] = {}
        for raw_email, raw_collections in payload.items():
            if not isinstance(raw_email, str) or not isinstance(raw_collections, dict):
                continue
            email_key = raw_email.casefold()
            normalized[email_key] = {}
            for collection_name, sources in raw_collections.items():
                if not isinstance(collection_name, str):
                    continue
                source_values = [str(item) for item in sources] if isinstance(sources, list) else []
                normalized[email_key][collection_name] = sorted(set(source_values))
        return normalized

    def _persist_collections(self) -> None:
        self._write_json_payload(self._collections_path, self._collections)

    def list_collections(self, user_email: str) -> list[str]:
        user_key = user_email.strip().casefold()
        collections = self._collections.get(user_key, {})
        return sorted(collections.keys(), key=lambda item: item.casefold())

    def create_collection(self, user_email: str, collection_name: str) -> None:
        normalized_name = collection_name.strip()
        if not normalized_name:
            raise ValueError("Collection name must not be empty")

        user_key = user_email.strip().casefold()
        self._collections.setdefault(user_key, {})
        self._collections[user_key].setdefault(normalized_name, [])
        self._persist_collections()

    def delete_collection(self, user_email: str, collection_name: str) -> None:
        user_key = user_email.strip().casefold()
        user_collections = self._collections.get(user_key, {})
        if collection_name in user_collections:
            del user_collections[collection_name]
            self._persist_collections()

    def get_collection_sources(self, user_email: str, collection_name: str | None) -> list[str]:
        if not collection_name:
            return []
        user_key = user_email.strip().casefold()
        user_collections = self._collections.get(user_key, {})
        values = user_collections.get(collection_name, [])
        return [str(item) for item in values]

    def set_collection_sources(
        self,
        user_email: str,
        collection_name: str,
        source_paths: Sequence[str],
    ) -> None:
        normalized_name = collection_name.strip()
        if not normalized_name:
            raise ValueError("Collection name must not be empty")

        user_key = user_email.strip().casefold()
        self._collections.setdefault(user_key, {})
        values = sorted({str(item).strip() for item in source_paths if str(item).strip()})
        self._collections[user_key][normalized_name] = values
        self._persist_collections()

    def add_source_to_collection(self, user_email: str, collection_name: str, source_path: str) -> None:
        existing = self.get_collection_sources(user_email, collection_name)
        updated = sorted(set(existing + [source_path]))
        self.set_collection_sources(user_email, collection_name, updated)

    def list_indexed_sources(self) -> list[dict[str, str]]:
        if not self.index_ready:
            return []

        vector_store = self._get_vector_store()
        docstore = getattr(vector_store, "docstore", None)
        doc_map = getattr(docstore, "_dict", {})
        if not isinstance(doc_map, dict):
            return []

        merged: dict[str, dict[str, str]] = {}
        for item in doc_map.values():
            metadata = item.metadata if isinstance(getattr(item, "metadata", {}), dict) else {}
            source_path = str(metadata.get("source_path", "")).strip()
            source_name = str(metadata.get("source", "")).strip() or "unknown"
            source_relpath = str(metadata.get("source_relpath", "")).strip()
            if not source_path:
                continue

            key = _normalize_path_key(source_path)
            if key in merged:
                continue
            merged[key] = {
                "source": source_name,
                "source_path": source_path,
                "source_relpath": source_relpath,
            }

        return sorted(merged.values(), key=lambda row: row["source"].casefold())

    def validate_email(self, email: str) -> tuple[bool, str]:
        normalized = email.strip().casefold()
        if not normalized:
            return False, "Email khong duoc de trong"

        if not re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+", normalized):
            return False, "Email khong hop le"

        allowed_domains = [
            item.strip().casefold()
            for item in os.getenv("WEB_ALLOWED_EMAIL_DOMAINS", "sgu.edu.vn").split(",")
            if item.strip()
        ]
        if "*" in allowed_domains:
            return True, "ok"

        domain = normalized.split("@", maxsplit=1)[-1]
        if domain not in allowed_domains:
            return False, f"Chi chap nhan email thuoc cac domain: {', '.join(allowed_domains)}"

        return True, "ok"

    def check_rate_limit(self, user_email: str) -> tuple[bool, int, int]:
        limit = max(1, int(os.getenv("WEB_RATE_LIMIT_REQUESTS", "20")))
        window_seconds = max(10, int(os.getenv("WEB_RATE_LIMIT_WINDOW_SECONDS", "60")))

        user_key = user_email.strip().casefold()
        now = time.time()
        events = self._rate_limit_state.setdefault(user_key, [])
        events[:] = [item for item in events if now - item < window_seconds]

        if len(events) >= limit:
            retry_after = max(1, int(window_seconds - (now - events[0])))
            return False, 0, retry_after

        events.append(now)
        remaining = max(0, limit - len(events))
        return True, remaining, 0

    @staticmethod
    def _fallback_embedding(text: str, dimensions: int = 128) -> list[float]:
        vector = [0.0] * dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.casefold())
        for token in tokens:
            vector[hash(token) % dimensions] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return vector
        return [value / norm for value in vector]

    def _embed_text(self, text: str) -> list[float]:
        try:
            vector_store = self._get_vector_store()
            embedding_function = getattr(vector_store, "embedding_function", None)

            if embedding_function is None:
                manager = VectorIndexManager(settings=self.settings, logger=self.logger)
                embedding_function = manager.embeddings

            values: Any
            if hasattr(embedding_function, "embed_query"):
                values = embedding_function.embed_query(text)
            elif callable(embedding_function):
                values = embedding_function(text)
            else:
                return self._fallback_embedding(text)

            if isinstance(values, list) and values and isinstance(values[0], list):
                values = values[0]
            if not isinstance(values, list):
                return self._fallback_embedding(text)

            return [float(item) for item in values]
        except Exception:  # noqa: BLE001
            self.logger.warning("Falling back to hash embedding for semantic cache", exc_info=True)
            return self._fallback_embedding(text)

    def _load_semantic_cache(self) -> list[dict[str, Any]]:
        if self._redis_client is not None:
            try:
                rows = self._redis_client.lrange(self._semantic_cache_redis_key, 0, -1)
                payload = [json.loads(item) for item in rows]
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Failed to read semantic cache from redis: %s", error)
                payload = []
        else:
            payload = self._read_json_payload(self._semantic_cache_path, default=[])

        if not isinstance(payload, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("question"), str):
                continue
            if not isinstance(item.get("answer"), str):
                continue
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue

            normalized.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "sources": item.get("sources", []),
                    "embedding": [float(value) for value in embedding],
                    "user_email": str(item.get("user_email", "")).casefold(),
                    "collection": str(item.get("collection", "")),
                    "created_unix": float(item.get("created_unix", 0.0)),
                    "created_at": str(item.get("created_at", "")),
                }
            )
        return normalized

    def _persist_semantic_cache(self) -> None:
        max_entries = max(20, int(os.getenv("WEB_SEMANTIC_CACHE_MAX_ENTRIES", "300")))
        self._semantic_cache = self._semantic_cache[-max_entries:]

        if self._redis_client is not None:
            try:
                ttl_seconds = max(3600, int(os.getenv("WEB_SEMANTIC_CACHE_TTL_SECONDS", "259200")))
                pipeline = self._redis_client.pipeline(transaction=True)
                pipeline.delete(self._semantic_cache_redis_key)
                if self._semantic_cache:
                    serialized = [json.dumps(item, ensure_ascii=False) for item in self._semantic_cache]
                    pipeline.rpush(self._semantic_cache_redis_key, *serialized)
                    pipeline.expire(self._semantic_cache_redis_key, ttl_seconds)
                pipeline.execute()
                return
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Failed to persist semantic cache to redis: %s", error)

        self._write_json_payload(self._semantic_cache_path, self._semantic_cache)

    def _lookup_semantic_cache(
        self,
        question: str,
        question_embedding: Sequence[float],
        user_email: str,
        collection: str,
    ) -> SemanticCacheMatch | None:
        threshold = float(os.getenv("WEB_SEMANTIC_CACHE_THRESHOLD", "0.95"))
        max_age_hours = max(1, int(os.getenv("WEB_SEMANTIC_CACHE_MAX_AGE_HOURS", "72")))
        max_age_seconds = float(max_age_hours * 3600)
        now = time.time()

        email_key = user_email.casefold()
        collection_key = collection.strip().casefold()

        best_similarity = 0.0
        best_match: SemanticCacheMatch | None = None

        for entry in reversed(self._semantic_cache):
            entry_email = str(entry.get("user_email", "")).casefold()
            entry_collection = str(entry.get("collection", "")).strip().casefold()
            if entry_email != email_key or entry_collection != collection_key:
                continue

            created_unix = float(entry.get("created_unix", 0.0))
            if created_unix and now - created_unix > max_age_seconds:
                continue

            similarity = _cosine_similarity(question_embedding, entry.get("embedding", []))
            if similarity < threshold or similarity < best_similarity:
                continue

            best_similarity = similarity
            best_match = SemanticCacheMatch(
                similarity=similarity,
                answer=str(entry.get("answer", "")),
                sources=entry.get("sources", []) if isinstance(entry.get("sources", []), list) else [],
            )

        return best_match

    def _store_semantic_cache(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        question_embedding: Sequence[float],
        user_email: str,
        collection: str,
    ) -> None:
        if not answer.strip():
            return

        self._semantic_cache.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
                "embedding": [float(value) for value in question_embedding],
                "user_email": user_email.casefold(),
                "collection": collection,
                "created_unix": time.time(),
                "created_at": utc_now_iso(),
            }
        )
        self._persist_semantic_cache()

    def _get_memory_state(self, conversation_key: str) -> ConversationMemoryState:
        if conversation_key not in self._conversation_states:
            self._conversation_states[conversation_key] = ConversationMemoryState()
        return self._conversation_states[conversation_key]

    def get_memory_context(self, conversation_key: str) -> tuple[str, list[dict[str, str]]]:
        state = self._get_memory_state(conversation_key)
        return state.summary, list(state.recent_turns)

    def clear_memory_context(self, conversation_key: str) -> None:
        if conversation_key in self._conversation_states:
            del self._conversation_states[conversation_key]

    def _append_turn_to_memory(self, conversation_key: str, question: str, answer: str) -> None:
        state = self._get_memory_state(conversation_key)
        state.recent_turns.append({"question": question, "answer": answer})

        window_size = max(1, int(os.getenv("WEB_MEMORY_WINDOW", "5")))
        if len(state.recent_turns) <= window_size:
            return

        expired_turns = state.recent_turns[:-window_size]
        state.recent_turns = state.recent_turns[-window_size:]

        try:
            qa_system = self._get_qa_system()
            state.summary = qa_system.summarize_expired_turns(
                existing_summary=state.summary,
                expired_turns=expired_turns,
            )
        except Exception:  # noqa: BLE001
            fallback = " ".join(
                [
                    state.summary,
                    " ".join(
                        f"Q: {item['question']} A: {item['answer']}" for item in expired_turns
                    ),
                ]
            ).strip()
            state.summary = fallback[:700]

    def start_query(
        self,
        question: str,
        user_email: str,
        collection: str,
        conversation_key: str,
        top_k: int | None = None,
    ) -> LiveQueryHandle:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty")

        if not self.index_ready:
            raise FileNotFoundError("Vector index not found. Build index before querying.")

        is_allowed, _, retry_after = self.check_rate_limit(user_email)
        if not is_allowed:
            raise RuntimeError(f"Rate limit exceeded. Thu lai sau {retry_after} giay.")

        question_embedding = self._embed_text(normalized_question)
        cache_match = self._lookup_semantic_cache(
            question=normalized_question,
            question_embedding=question_embedding,
            user_email=user_email,
            collection=collection,
        )
        if cache_match is not None:
            self.trace_event(
                "query_cache_hit",
                {
                    "question": normalized_question,
                    "user_email": user_email,
                    "collection": collection,
                    "similarity": cache_match.similarity,
                },
            )
            return LiveQueryHandle(
                interaction_id=str(uuid.uuid4()),
                question=normalized_question,
                user_email=user_email,
                collection=collection,
                conversation_key=conversation_key,
                cache_hit=True,
                cache_similarity=cache_match.similarity,
                cached_answer=cache_match.answer,
                cached_sources=cache_match.sources,
                stream_session=None,
                started_at=time.perf_counter(),
                question_embedding=question_embedding,
            )

        conversation_summary, recent_turns = self.get_memory_context(conversation_key)
        collection_sources = self.get_collection_sources(user_email, collection)

        qa_system = self._get_qa_system()
        vector_store = self._get_vector_store()
        stream_session = qa_system.stream_query(
            question=normalized_question,
            vector_store=vector_store,
            top_k=top_k,
            conversation_summary=conversation_summary,
            recent_turns=recent_turns,
            allowed_source_paths=collection_sources or None,
        )

        handle = LiveQueryHandle(
            interaction_id=str(uuid.uuid4()),
            question=normalized_question,
            user_email=user_email,
            collection=collection,
            conversation_key=conversation_key,
            cache_hit=False,
            cache_similarity=None,
            cached_answer=None,
            cached_sources=[],
            stream_session=stream_session,
            started_at=time.perf_counter(),
            question_embedding=question_embedding,
        )
        self.trace_event(
            "query_started",
            {
                "interaction_id": handle.interaction_id,
                "question": normalized_question,
                "user_email": user_email,
                "collection": collection,
                "retrieval_ms": stream_session.retrieval_ms,
            },
        )
        return handle

    @staticmethod
    def stream_cached_answer(answer: str):
        for token in re.findall(r"\S+\s*", answer):
            yield token

    def finalize_query(self, handle: LiveQueryHandle, rendered_answer: str | None = None) -> dict[str, Any]:
        if handle.cache_hit:
            answer = (rendered_answer or handle.cached_answer or RAGQASystem.NOT_FOUND_MESSAGE).strip()
            sources = handle.cached_sources
            retrieval_ms = 0.0
            generation_ms = 0.0
            ttft_ms = 0.0
        else:
            if handle.stream_session is None:
                raise RuntimeError("Missing streaming session for non-cached query")

            answer = (rendered_answer or "").strip() or handle.stream_session.answer
            sources = handle.stream_session.sources
            retrieval_ms = float(handle.stream_session.retrieval_ms)
            generation_ms = float(handle.stream_session.generation_ms)
            ttft_ms = float(handle.stream_session.ttft_ms or 0.0)

            self._store_semantic_cache(
                question=handle.question,
                answer=answer,
                sources=sources,
                question_embedding=handle.question_embedding,
                user_email=handle.user_email,
                collection=handle.collection,
            )

        total_ms = (time.perf_counter() - handle.started_at) * 1000.0
        result = {
            "interaction_id": handle.interaction_id,
            "question": handle.question,
            "answer": answer,
            "sources": sources,
            "cache_hit": handle.cache_hit,
            "cache_similarity": handle.cache_similarity,
            "timings": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
            },
        }

        self._append_turn_to_memory(handle.conversation_key, handle.question, answer)
        self.trace_event(
            "query_completed",
            {
                "interaction_id": handle.interaction_id,
                "user_email": handle.user_email,
                "collection": handle.collection,
                "cache_hit": handle.cache_hit,
                "cache_similarity": handle.cache_similarity,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
            },
        )
        return result

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty")

        qa_system = self._get_qa_system()
        vector_store = self._get_vector_store()
        return qa_system.query(
            question=normalized_question,
            vector_store=vector_store,
            top_k=top_k,
        )

    def batch_query(self, questions: list[str], top_k: int | None = None) -> list[dict[str, Any]]:
        normalized_questions = [item.strip() for item in questions if item.strip()]
        if not normalized_questions:
            raise ValueError("At least one non-empty question is required")

        qa_system = self._get_qa_system()
        vector_store = self._get_vector_store()
        return qa_system.batch_query(
            questions=normalized_questions,
            vector_store=vector_store,
            top_k=top_k,
        )

    def suggest_queries(
        self,
        draft_query: str,
        user_email: str,
        collection: str,
        limit: int = 5,
    ) -> list[str]:
        normalized_query = draft_query.strip()
        if len(normalized_query) < 2 or not self.index_ready:
            return []

        qa_system = self._get_qa_system()
        vector_store = self._get_vector_store()
        collection_sources = self.get_collection_sources(user_email, collection)
        return qa_system.suggest_questions(
            draft_query=normalized_query,
            vector_store=vector_store,
            limit=limit,
            allowed_source_paths=collection_sources or None,
        )

    def save_uploaded_pdf(
        self,
        user_email: str,
        collection: str,
        file_name: str,
        payload: bytes,
    ) -> Path:
        safe_email = _safe_slug(user_email.split("@", maxsplit=1)[0], fallback="user")
        safe_collection = _safe_slug(collection, fallback="default")
        safe_name = _sanitize_filename(file_name)

        destination = self.settings.pdf_dir / "uploads" / safe_email / safe_collection / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

        self.add_source_to_collection(user_email, collection, str(destination.resolve()))
        self.trace_event(
            "document_uploaded",
            {
                "user_email": user_email,
                "collection": collection,
                "destination": str(destination),
                "bytes": len(payload),
            },
        )
        return destination

    def summarize_uploaded_pdf(self, pdf_path: Path) -> str:
        try:
            processor = OCRProcessor(self.settings)
            extracted = processor.extract_pdf(pdf_path)
            qa_system = self._get_qa_system()
            summary = qa_system.summarize_document(extracted.text)
            return summary
        except Exception as error:  # noqa: BLE001
            self.logger.warning("Unable to summarize uploaded document: %s", error)
            return "Khong the tao tom tat tu dong cho tai lieu nay."

    def record_feedback(
        self,
        interaction_id: str,
        user_email: str,
        collection: str,
        score: str,
        question: str,
        answer: str,
        cache_hit: bool,
    ) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "interaction_id": interaction_id,
            "user_email": user_email,
            "collection": collection,
            "score": score,
            "question": question,
            "answer": answer,
            "cache_hit": cache_hit,
        }
        self._append_jsonl(self._feedback_path, payload)

    def trace_event(self, event_name: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": utc_now_iso(),
            "event": event_name,
            "provider": self.tracing_provider,
            "payload": payload,
        }
        self._append_jsonl(self._trace_path, row)

    def reset_runtime_cache(self) -> None:
        self._vector_store = None
        self._qa_system = None


def build_web_service(base_dir: str | Path, run_name: str = "streamlit") -> WebRAGService:
    settings = load_settings(base_dir=base_dir)
    settings.run_name = run_name
    logger = configure_logging(settings.logs_dir, settings.run_name, verbose=False)
    return WebRAGService(settings=settings, logger=logger)
