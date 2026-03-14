from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

try:
    from langchain.prompts import PromptTemplate  # type: ignore[import-not-found]
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    PromptTemplate = None  # type: ignore[assignment]
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]

from .config import RAGSettings


def _extract_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text

    return str(chunk)


def _normalize_path_key(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    try:
        return str(Path(raw_path).resolve()).casefold()
    except Exception:  # noqa: BLE001
        return str(raw_path).casefold()


@dataclass
class QueryPreparation:
    question: str
    prompt_text: str
    sources: list[dict[str, Any]]
    retrieval_ms: float
    effective_k: int


@dataclass
class StreamingAnswerSession:
    llm: Any
    prompt_text: str
    not_found_message: str
    sources: list[dict[str, Any]]
    retrieval_ms: float
    question: str
    effective_k: int
    _tokens: list[str] = field(default_factory=list)
    ttft_ms: float | None = None
    generation_ms: float = 0.0
    _completed: bool = False

    def token_generator(self) -> Iterator[str]:
        started_at = time.perf_counter()
        for chunk in self.llm.stream(self.prompt_text):
            token = _extract_chunk_text(chunk)
            if not token:
                continue

            if self.ttft_ms is None:
                self.ttft_ms = (time.perf_counter() - started_at) * 1000.0

            self._tokens.append(token)
            yield token

        self.generation_ms = (time.perf_counter() - started_at) * 1000.0
        self._completed = True

    @property
    def answer(self) -> str:
        rendered = "".join(self._tokens).strip()
        return rendered if rendered else self.not_found_message

    @property
    def completed(self) -> bool:
        return self._completed


class RAGQASystem:
    NOT_FOUND_MESSAGE = "Toi khong tim thay thong tin nay trong tai lieu"

    def __init__(self, settings: RAGSettings, logger):
        if PromptTemplate is None or ChatGoogleGenerativeAI is None:
            raise ImportError(
                "QA dependencies are missing. Install requirements.txt before running queries."
            )

        self.settings = settings
        self.logger = logger
        self.prompt = self._build_prompt()
        self.summary_prompt = self._build_summary_prompt()
        self.document_summary_prompt = self._build_document_summary_prompt()
        self._llm: Any | None = None

    def _build_prompt(self) -> Any:
        assert PromptTemplate is not None
        template = """Ban la tro ly AI hoc thuat cho sinh vien SGU.

MUC TIEU:
- Tra loi dua hoan toan vao ngu canh trich tu tai lieu.
- Neu cau tra loi su dung nguon nao thi phai gan so trich dan [n] dung voi khoi ngu canh [n].
- Ho tro Markdown, bang, code block va cong thuc toan hoc LaTeX khi phu hop.

TOM TAT NGU CANH CUOC HOI THOAI:
{conversation_summary}

5 LUOT GAN NHAT:
{recent_turns}

KHOI TRI THUC TRUY XUAT:
{context}

CAU HOI:
{question}

QUY TAC:
1. Khong biet thi tra loi chinh xac cau: \"Toi khong tim thay thong tin nay trong tai lieu\".
2. Khong bia, khong chen thong tin ngoai tai lieu.
3. Van ban tra loi bang tieng Viet ngan gon va ro rang.
4. Uu tien dang bullet neu can liet ke.

TRA LOI:"""
        return PromptTemplate(
            template=template,
            input_variables=["conversation_summary", "recent_turns", "context", "question"],
        )

    def _build_summary_prompt(self) -> Any:
        assert PromptTemplate is not None
        template = """Ban dang quan ly bo nho hoi thoai cho he thong RAG.

TOM TAT HIEN CO:
{existing_summary}

LUOT HOI DAP CAN GOM LAI:
{expired_turns}

Hay cap nhat lai tom tat (toi da 120 tu), giu du thong tin can thiet de tra loi tiep theo.
Chi tra ve doan tom tat duy nhat."""
        return PromptTemplate(
            template=template,
            input_variables=["existing_summary", "expired_turns"],
        )

    def _build_document_summary_prompt(self) -> Any:
        assert PromptTemplate is not None
        template = """Ban la tro ly hoc tap. Hay tom tat tai lieu sau cho sinh vien.

TAI LIEU:
{document_text}

YEU CAU:
- 3 den 5 bullet points
- Neu co cong thuc toan, giu nguyen ky hieu LaTeX
- Ngan gon, de doc
"""
        return PromptTemplate(template=template, input_variables=["document_text"])

    def _create_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Add it to .env before running query.")

        assert ChatGoogleGenerativeAI is not None
        self._llm = ChatGoogleGenerativeAI(
            model=self.settings.llm_model,
            google_api_key=self.settings.google_api_key,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_tokens,
            convert_system_message_to_human=True,
        )
        return self._llm

    @staticmethod
    def _format_recent_turns(turns: Sequence[dict[str, Any]] | None) -> str:
        if not turns:
            return "(khong co)"

        lines: list[str] = []
        for index, turn in enumerate(turns[-5:], start=1):
            question = str(turn.get("question", "")).strip()
            answer = str(turn.get("answer", "")).strip()
            if not question or not answer:
                continue
            lines.append(f"{index}. Q: {question}")
            lines.append(f"   A: {answer}")

        return "\n".join(lines) if lines else "(khong co)"

    @staticmethod
    def _truncate_preview(text: str, limit: int = 240) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _is_doc_allowed(
        self,
        metadata: dict[str, Any],
        allowed_source_paths: set[str] | None,
    ) -> bool:
        if not allowed_source_paths:
            return True

        source_path = str(metadata.get("source_path", "")).strip()
        if not source_path:
            return False
        return _normalize_path_key(source_path) in allowed_source_paths

    def _retrieve_documents(
        self,
        question: str,
        vector_store: Any,
        top_k: int,
        allowed_source_paths: set[str] | None = None,
        depth_multiplier: int = 4,
    ) -> list[Any]:
        search_depth = max(top_k, top_k * max(1, depth_multiplier))
        candidates = vector_store.similarity_search(question, k=search_depth)

        filtered: list[Any] = []
        for item in candidates:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            if self._is_doc_allowed(metadata, allowed_source_paths):
                filtered.append(item)
            if len(filtered) >= top_k:
                break

        return filtered

    def _documents_to_sources(self, docs: list[Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, item in enumerate(docs, start=1):
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            page_number_raw = metadata.get("page_number")
            page_number = int(page_number_raw) if isinstance(page_number_raw, int) else None

            sources.append(
                {
                    "citation": index,
                    "source": str(metadata.get("source", "unknown")),
                    "source_relpath": str(metadata.get("source_relpath", "")),
                    "source_path": str(metadata.get("source_path", "")),
                    "chunk_id": int(metadata.get("chunk_id", -1)),
                    "page_number": page_number,
                    "preview": self._truncate_preview(item.page_content),
                }
            )

        return sources

    def _build_context(self, docs: list[Any]) -> str:
        if not docs:
            return "(khong tim thay ngu canh phu hop)"

        context_blocks: list[str] = []
        for index, item in enumerate(docs, start=1):
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            source_name = str(metadata.get("source", "unknown"))
            page_number = metadata.get("page_number")
            page_hint = f" | page={page_number}" if isinstance(page_number, int) else ""
            context_text = item.page_content.strip()
            context_blocks.append(f"[{index}] source={source_name}{page_hint}\n{context_text}")

        return "\n\n".join(context_blocks)

    def prepare_query(
        self,
        question: str,
        vector_store: Any,
        top_k: int | None = None,
        conversation_summary: str | None = None,
        recent_turns: Sequence[dict[str, Any]] | None = None,
        allowed_source_paths: Sequence[str] | None = None,
    ) -> QueryPreparation:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty")

        effective_k = top_k or self.settings.retrieval_k
        allowed_paths = {
            _normalize_path_key(item) for item in (allowed_source_paths or []) if str(item).strip()
        }
        start_retrieval = time.perf_counter()
        docs = self._retrieve_documents(
            question=normalized_question,
            vector_store=vector_store,
            top_k=effective_k,
            allowed_source_paths=allowed_paths if allowed_paths else None,
        )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

        context = self._build_context(docs)
        prompt_text = self.prompt.format(
            conversation_summary=(conversation_summary or "(khong co)").strip() or "(khong co)",
            recent_turns=self._format_recent_turns(recent_turns),
            context=context,
            question=normalized_question,
        )

        return QueryPreparation(
            question=normalized_question,
            prompt_text=prompt_text,
            sources=self._documents_to_sources(docs),
            retrieval_ms=retrieval_ms,
            effective_k=effective_k,
        )

    def query(
        self,
        question: str,
        vector_store: Any,
        top_k: int | None = None,
        conversation_summary: str | None = None,
        recent_turns: Sequence[dict[str, Any]] | None = None,
        allowed_source_paths: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        preparation = self.prepare_query(
            question=question,
            vector_store=vector_store,
            top_k=top_k,
            conversation_summary=conversation_summary,
            recent_turns=recent_turns,
            allowed_source_paths=allowed_source_paths,
        )

        llm = self._create_llm()
        self.logger.info(
            "Running QA query | top_k=%s | question=%s",
            preparation.effective_k,
            preparation.question,
        )

        generation_started = time.perf_counter()
        response = llm.invoke(preparation.prompt_text)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        answer = _extract_chunk_text(response).strip()
        if not answer:
            answer = self.NOT_FOUND_MESSAGE

        return {
            "question": preparation.question,
            "answer": answer,
            "sources": preparation.sources,
            "timings": {
                "retrieval_ms": preparation.retrieval_ms,
                "generation_ms": generation_ms,
                "ttft_ms": None,
            },
            "cache_hit": False,
        }

    def stream_query(
        self,
        question: str,
        vector_store: Any,
        top_k: int | None = None,
        conversation_summary: str | None = None,
        recent_turns: Sequence[dict[str, Any]] | None = None,
        allowed_source_paths: Sequence[str] | None = None,
    ) -> StreamingAnswerSession:
        preparation = self.prepare_query(
            question=question,
            vector_store=vector_store,
            top_k=top_k,
            conversation_summary=conversation_summary,
            recent_turns=recent_turns,
            allowed_source_paths=allowed_source_paths,
        )
        llm = self._create_llm()

        self.logger.info(
            "Running streaming QA query | top_k=%s | question=%s",
            preparation.effective_k,
            preparation.question,
        )

        return StreamingAnswerSession(
            llm=llm,
            prompt_text=preparation.prompt_text,
            not_found_message=self.NOT_FOUND_MESSAGE,
            sources=preparation.sources,
            retrieval_ms=preparation.retrieval_ms,
            question=preparation.question,
            effective_k=preparation.effective_k,
        )

    def suggest_questions(
        self,
        draft_query: str,
        vector_store: Any,
        limit: int = 5,
        allowed_source_paths: Sequence[str] | None = None,
    ) -> list[str]:
        normalized = draft_query.strip()
        if len(normalized) < 2:
            return []

        allowed_paths = {
            _normalize_path_key(item) for item in (allowed_source_paths or []) if str(item).strip()
        }
        docs = self._retrieve_documents(
            question=normalized,
            vector_store=vector_store,
            top_k=max(limit * 3, limit),
            allowed_source_paths=allowed_paths if allowed_paths else None,
            depth_multiplier=2,
        )

        suggestions: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            preview = self._truncate_preview(str(doc.page_content), limit=90)
            if not preview:
                continue

            topic = preview.split(".")[0].strip(" :,-")
            topic = re.sub(r"\s+", " ", topic)
            if not topic:
                continue

            if normalized.endswith("?"):
                candidate = normalized
            else:
                candidate = f"{normalized} lien quan den {topic} nhu the nao?"

            lowered = candidate.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            suggestions.append(candidate)
            if len(suggestions) >= limit:
                break

        return suggestions

    def summarize_document(self, raw_text: str) -> str:
        normalized = re.sub(r"\s+", " ", raw_text).strip()
        if not normalized:
            return "Khong co noi dung van ban de tom tat."

        if not self.settings.google_api_key:
            return self._truncate_preview(normalized, limit=500)

        llm = self._create_llm()
        prompt_text = self.document_summary_prompt.format(document_text=normalized[:7000])
        response = llm.invoke(prompt_text)
        answer = _extract_chunk_text(response).strip()
        if not answer:
            return self._truncate_preview(normalized, limit=500)
        return answer

    def summarize_expired_turns(
        self,
        existing_summary: str,
        expired_turns: Sequence[dict[str, Any]],
    ) -> str:
        serialized_turns: list[str] = []
        for item in expired_turns:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                serialized_turns.append(f"Q: {question}\nA: {answer}")

        if not serialized_turns:
            return existing_summary.strip()

        fallback = " ".join([existing_summary.strip(), " ".join(serialized_turns)]).strip()
        fallback = self._truncate_preview(fallback, limit=700)

        if not self.settings.google_api_key:
            return fallback

        try:
            llm = self._create_llm()
            prompt_text = self.summary_prompt.format(
                existing_summary=existing_summary.strip() or "(khong co)",
                expired_turns="\n\n".join(serialized_turns),
            )
            response = llm.invoke(prompt_text)
            summary = _extract_chunk_text(response).strip()
            if summary:
                return summary
        except Exception:  # noqa: BLE001
            self.logger.warning("Failed to update conversation summary with LLM", exc_info=True)

        return fallback

    def batch_query(
        self,
        questions: list[str],
        vector_store: Any,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized_questions = [item.strip() for item in questions if item.strip()]
        if not normalized_questions:
            raise ValueError("At least one non-empty question is required")
        return [self.query(question=item, vector_store=vector_store, top_k=top_k) for item in normalized_questions]
