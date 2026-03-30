from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .environment import configure_runtime_environment

# Must be applied before importing Gemini-related modules.
configure_runtime_environment()

from .config import RAGConfig
from .vector_store import VectorStoreManager

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp, chuyên gia về tài liệu đào tạo.

NHIỆM VỤ: Trả lời câu hỏi của người dùng dựa trên thông tin từ tài liệu được cung cấp.
    
THÔNG TIN TỪ TÀI LIỆU:
{context}

CÂU HỎI: {question}

YÊU CẦU:
1. Trả lời chính xác, dựa hoàn toàn vào thông tin được cung cấp
2. Trả lời rõ ràng, đầy đủ các ý liên quan bằng tiếng Việt (không rút gọn quá mức)
3. Nếu không tìm thấy thông tin, hãy trả lời: "Tôi không tìm thấy thông tin này trong tài liệu"
4. Không bịa đặt thông tin không có trong tài liệu
5. Sử dụng bullet points nếu cần liệt kê
6. Nếu câu hỏi không liên quan đến tài liệu, trả lời: "Tôi không tìm thấy thông tin này trong tài liệu"
7. Với câu hỏi dạng "mục tiêu", "chuẩn đầu ra", "nội dung gồm những gì", hãy liệt kê đầy đủ các ý tìm thấy trong ngữ cảnh
8. Đảm bảo câu trả lời kết thúc trọn ý, không bỏ dở giữa câu

TRẢ LỜI:
"""

NOT_FOUND_FALLBACK = "Tôi không tìm thấy thông tin này trong tài liệu"


class RAGService:
    """Shared QA service used by notebook and Streamlit app."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_manager = VectorStoreManager(config)
        self.vector_store = None
        self.llm = None
        self.llm_unavailable_reason = ""
        self._retriever_ready = False
        self._retriever_lock = threading.Lock()
        self._preload_started = False
        self._preload_error = ""
        self._preload_thread: Optional[threading.Thread] = None
        self._init_llm_if_available()

    @property
    def retriever_ready(self) -> bool:
        return self._retriever_ready

    @property
    def retriever_status(self) -> str:
        if self._retriever_ready:
            return "Đã nạp index"
        if self._preload_started:
            return "Đang preload nền..."
        if self._preload_error:
            return "Preload lỗi, sẽ lazy-load khi hỏi"
        return "Lazy-load (nạp ở câu hỏi đầu tiên)"

    @property
    def retriever_last_error(self) -> str:
        return self._preload_error

    def attach_vector_store(self, vector_store: Any) -> None:
        with self._retriever_lock:
            self.vector_store = vector_store
            self._retriever_ready = True
            self._preload_started = False
            self._preload_error = ""

    def _load_vector_store(self) -> None:
        self.vector_store = self.vector_manager.load()

    def _ensure_retriever_ready(self) -> None:
        if self._retriever_ready:
            return

        with self._retriever_lock:
            if self._retriever_ready:
                return

            self._load_vector_store()
            self._retriever_ready = True
            self._preload_error = ""

    def _preload_worker(self) -> None:
        try:
            self._ensure_retriever_ready()
        except Exception as exc:  # pragma: no cover - defensive path
            with self._retriever_lock:
                self._preload_error = f"{type(exc).__name__}: {exc}"
                self._preload_started = False
            return

        with self._retriever_lock:
            self._preload_started = False

    def start_preload(self) -> None:
        if self._retriever_ready:
            return

        with self._retriever_lock:
            if self._retriever_ready or self._preload_started:
                return

            self._preload_started = True
            self._preload_error = ""
            self._preload_thread = threading.Thread(
                target=self._preload_worker,
                name="rag-retriever-preload",
                daemon=True,
            )
            self._preload_thread.start()

    def _init_llm_if_available(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            self.llm = None
            self.llm_unavailable_reason = "Chưa cấu hình GOOGLE_API_KEY"
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=api_key,
                temperature=self.config.llm_temperature,
                max_output_tokens=self.config.llm_max_tokens,
                # model_kwargs={
                #     "transport": self.config.llm_api_transport
                # },
                # retries=1,
                # convert_system_message_to_human=True,
            )
            self.llm_unavailable_reason = ""
        except Exception as exc:
            self.llm = None
            message = str(exc)
            if "Descriptors cannot be created directly" in message:
                self.llm_unavailable_reason = (
                    "Gemini không khả dụng do protobuf không tương thích. "
                    "Hãy cài protobuf 3.20.x hoặc dùng PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python."
                )
            else:
                self.llm_unavailable_reason = f"Gemini không khả dụng ({type(exc).__name__})"

    @staticmethod
    def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".casefold()
        markers = [
            "toomanyrequests",
            "resourceexhausted",
            "rate limit",
            "quota",
            "429",
        ]
        return any(marker in text for marker in markers)

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.casefold()
        without_marks = "".join(
            ch for ch in unicodedata.normalize("NFD", lowered) if unicodedata.category(ch) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", without_marks).strip()

    @staticmethod
    def _short_source_name(source: str) -> str:
        raw = Path(str(source).strip()).name
        if not raw:
            return "Tài liệu"

        cleaned = raw
        copy_prefixes = ("Bản sao của ", "Ban sao cua ")
        changed = True
        while changed:
            changed = False
            for prefix in copy_prefixes:
                if cleaned.casefold().startswith(prefix.casefold()):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True
                    break

        return cleaned or raw

    @staticmethod
    def _coerce_page_number(metadata: Dict[str, Any]) -> Optional[int]:
        if metadata.get("page_number") is not None:
            raw_page = metadata.get("page_number")
            offset = 0
        elif metadata.get("page") is not None:
            raw_page = metadata.get("page")
            offset = 1
        else:
            return None

        try:
            return int(raw_page) + offset
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_source_label(doc: Any, idx: int) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        source_keys = ["source", "source_relpath", "source_path", "file_name", "document", "doc_id"]
        source = ""
        for key in source_keys:
            value = metadata.get(key)
            if value is not None and str(value).strip():
                source = str(value).strip()
                break

        source_name = (
            RAGService._short_source_name(source)
            if source
            else f"Tài liệu {idx} (thiếu metadata nguồn)"
        )
        page = RAGService._coerce_page_number(metadata)
        if page is not None:
            return f"{source_name} - trang {page}"
        return source_name

    @staticmethod
    def _dedupe_sources(labels: List[str]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for label in labels:
            key = label.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(label)
        return unique

    @staticmethod
    def _build_fallback_snippets(docs: List[Any], max_items: int = 3) -> List[str]:
        snippets: List[str] = []
        for idx, doc in enumerate(docs[:max_items], start=1):
            content = str(getattr(doc, "page_content", "")).replace(chr(10), " ").strip()
            if not content:
                continue

            source_label = RAGService._extract_source_label(doc, idx)
            short_content = content[:220].rstrip()
            if len(content) > 220:
                short_content += "..."

            snippets.append(f"- **{source_label}**: {short_content}")
        return snippets

    @staticmethod
    def _build_context(docs: List[Any]) -> str:
        return "\n\n".join(
            [
                f"[Nguồn {idx + 1}: {RAGService._extract_source_label(doc, idx + 1)}]\n{doc.page_content}"
                for idx, doc in enumerate(docs)
                if getattr(doc, "page_content", "").strip()
            ]
        )

    def _build_retrieval_only_answer(
        self,
        docs: List[Any],
        exc: Optional[Exception] = None,
    ) -> str:
        if exc is not None and self._is_quota_or_rate_limit_error(exc):
            self.llm = None
            self.llm_unavailable_reason = "Gemini vượt giới hạn truy cập (quota/rate limit)"
            reason = self.llm_unavailable_reason
        elif exc is not None:
            reason = f"Gemini tạm thời không khả dụng ({type(exc).__name__})"
        else:
            reason = self.llm_unavailable_reason or "LLM không khả dụng"

        mode_text = (
            "hệ thống chuyển sang chế độ chỉ truy xuất."
            if exc is not None
            else "hệ thống đang ở chế độ chỉ truy xuất."
        )

        fallback_snippets = self._build_fallback_snippets(docs)
        lines = [
            f"{reason}, {mode_text}",
            "Thông tin liên quan nhất (kèm nguồn):",
        ]
        if fallback_snippets:
            lines.extend(fallback_snippets)
        else:
            lines.append("- Không có đoạn nội dung phù hợp để hiển thị.")
        return "\n".join(lines)

    def _generate_answer(self, question: str, docs: List[Any]) -> str:
        context = self._build_context(docs)
        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = str(getattr(response, "content", response)).strip()
        if not answer:
            return NOT_FOUND_FALLBACK
        return answer

    def _retrieve_docs(self, question: str, k: int) -> List[Any]:
        return self.vector_store.similarity_search(question, k=k)

    @classmethod
    def _should_expand_retrieval(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        broad_markers = (
            "muc tieu",
            "chuan dau ra",
            "noi dung",
            "bao gom",
            "gom nhung gi",
            "liet ke",
        )
        return any(marker in normalized for marker in broad_markers)

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_retriever_ready()

        k = max(1, int(top_k or self.config.retrieval_k))
        if self._should_expand_retrieval(question):
            k = max(k, self.config.retrieval_k + 4)
        docs = self._retrieve_docs(question, k=k)

        if not docs:
            return {
                "answer": NOT_FOUND_FALLBACK,
                "sources": [],
                "docs": [],
            }

        sources = self._dedupe_sources(
            [self._extract_source_label(doc, idx + 1) for idx, doc in enumerate(docs)]
        )

        if self.llm is None:
            answer = self._build_retrieval_only_answer(docs)
            return {"answer": answer, "sources": sources, "docs": docs}

        try:
            answer = self._generate_answer(question, docs)
        except Exception as exc:
            answer = self._build_retrieval_only_answer(docs, exc=exc)

        return {"answer": answer, "sources": sources, "docs": docs}
