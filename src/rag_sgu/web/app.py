from __future__ import annotations

import base64
import html
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from .service import WebRAGService, build_web_service
from .student_db import (
    StudentAuthResult,
    authenticate_student,
    clear_chat_session,
    count_students,
    ensure_chat_session,
    get_db_path,
    get_seed_path,
    init_db,
    load_chat_history,
    record_login,
    save_chat_turn,
    save_feedback,
    seed_students_from_csv,
    update_student_password,
)


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

:root {
  --gemini-bg: #FFFFFF;
  --gemini-sidebar: #F0F4F9;
  --gemini-text: #1F1F1F;
  --gemini-subtext: #444746;
  --gemini-user-msg: #F0F4F9;
  --gemini-input-bg: #F0F4F9;
  --gemini-border: #E3E3E3;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--gemini-bg);
  color: var(--gemini-text);
  font-family: 'Be Vietnam Pro', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
  background-color: var(--gemini-sidebar);
  border-right: none;
}
[data-testid="stSidebar"] hr {
    border-color: #D3E3FD;
}

.block-container {
  max-width: 1200px;
  padding-top: 2rem;
  padding-bottom: 6rem;
}

/* Gemini Greeting Styling */
.gemini-greeting-container {
    padding: 2rem 0 3rem 0;
    margin-bottom: 1rem;
}
.gemini-greeting {
    font-size: 3.5rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.05rem;
    background: linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%, #9b72cb 50%, #d96570 56%, #131314 75%, #131314 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 400% 100%;
}
.gemini-sub-greeting {
    font-size: 3.5rem;
    font-weight: 600;
    color: #C4C7C5;
    line-height: 1.2;
    letter-spacing: -0.05rem;
}

/* Login Box (Minimalist) */
.sgu-login-box {
  background: #FFFFFF;
  border-radius: 24px;
  padding: 3rem 2.5rem;
  margin: 10vh auto 2rem auto;
  max-width: 450px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  border: 1px solid var(--gemini-border);
  text-align: center;
}

/* Chat Messages Styling */
[data-testid="stChatMessage"] {
  background: transparent;
  padding: 1.5rem 1rem;
  border-radius: 12px;
}

/* User Message specific styling */
[data-testid="stChatMessage"][data-baseweb="list-item"]:has([data-testid="userAvatar"]) {
    background-color: var(--gemini-user-msg);
    border-radius: 24px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    max-width: 90%;
    margin-left: auto;
}

/* Assistant Message specific styling */
[data-testid="stChatMessage"][data-baseweb="list-item"]:has([data-testid="botAvatar"]) {
    padding: 1rem 0;
}

/* Chat Input Styling */
[data-testid="stChatInput"] {
    background: var(--gemini-bg);
    padding-bottom: 2rem;
}
[data-testid="stChatInput"] textarea {
    background-color: var(--gemini-input-bg);
    border-radius: 32px;
    border: none;
    padding: 1rem 1.5rem;
    font-size: 1rem;
    box-shadow: none;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 0 1px #4285f4;
    background-color: #FFFFFF;
}

/* Feedback & Citations Chips */
.rag-feedback-chip, button[kind="secondary"] {
  border: 1px solid var(--gemini-border);
  border-radius: 16px;
  background: #FFFFFF;
  color: var(--gemini-subtext);
  font-size: 0.85rem;
  transition: all 0.2s;
}
.rag-feedback-chip:hover {
    background: #F8F9FA;
}

/* Citation Panel */
.rag-preview {
  border-left: 3px solid #4285f4;
  background: #F8F9FA;
  border-radius: 0 12px 12px 0;
  padding: 1rem 1.2rem;
  color: var(--gemini-subtext);
  margin-top: 0.5rem;
  font-size: 0.95rem;
}

/* Skeleton loader (Gemini Shimmer) */
.sgu-skeleton-card { padding: 0.5rem 0; }
.sgu-skeleton-line {
  height: 14px;
  border-radius: 8px;
  margin-bottom: 0.8rem;
  background: linear-gradient(90deg, #F0F4F9 25%, #E3E3E3 50%, #F0F4F9 75%);
  background-size: 400% 100%;
  animation: sgu-shimmer 1.5s infinite linear;
}
.w-90 { width: 95%; }
.w-75 { width: 80%; }
.w-60 { width: 60%; }

@keyframes sgu-shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}

.sgu-caption-metric {
  color: #a8c7fa;
  font-size: 0.8rem;
  font-weight: 500;
}
</style>
"""


def _inject_custom_css(st_module: Any) -> None:
    st_module.markdown(APP_CSS, unsafe_allow_html=True)


def _skeleton_markup() -> str:
    return """
<div class="sgu-skeleton-card">
  <div class="sgu-skeleton-line w-90"></div>
  <div class="sgu-skeleton-line w-75"></div>
  <div class="sgu-skeleton-line w-60"></div>
</div>
"""


@lru_cache(maxsize=32)
def _load_pdf_base64(pdf_path: str) -> str:
    payload = Path(pdf_path).read_bytes()
    return base64.b64encode(payload).decode("utf-8")


def _ensure_service(st_module: Any, base_dir: str) -> WebRAGService:
    current_base_dir = st_module.session_state.get("base_dir")
    if "web_service" not in st_module.session_state or current_base_dir != base_dir:
        st_module.session_state["web_service"] = build_web_service(base_dir=base_dir)
        st_module.session_state["base_dir"] = base_dir
        st_module.session_state["chat_history"] = []
        st_module.session_state["selected_citation"] = None
        st_module.session_state["pending_question"] = ""
        st_module.session_state["suggestion_cache_query"] = ""
        st_module.session_state["suggestion_cache_results"] = []
    return st_module.session_state["web_service"]


def _ensure_db(base_dir: Path) -> Path:
    db_path = get_db_path(base_dir)
    init_db(db_path)
    seed_path = get_seed_path(base_dir)
    if seed_path.exists():
        seed_students_from_csv(db_path, seed_path)
    return db_path


def _render_answer_metrics(st_module: Any, item: dict[str, Any]) -> None:
    timings = item.get("timings", {}) if isinstance(item.get("timings"), dict) else {}
    total_ms = float(timings.get("total_ms", 0.0) or 0.0)
    cache_hit = bool(item.get("cache_hit", False))

    if cache_hit:
        st_module.markdown("<span class='sgu-caption-metric'>✨ Phản hồi nhanh (Cached)</span>", unsafe_allow_html=True)
    elif total_ms > 0:
        st_module.markdown(f"<span class='sgu-caption-metric'>⏱️ Đã xử lý trong {total_ms/1000:.2f}s</span>", unsafe_allow_html=True)


def _build_citation_label(source: dict[str, Any], index: int) -> str:
    citation = int(source.get("citation", index))
    source_name = str(source.get("source", "unknown"))
    page_number = source.get("page_number")
    if isinstance(page_number, int):
        return f"[{citation}] {source_name} (tr.{page_number})"
    return f"[{citation}] {source_name}"


def _render_feedback_controls(
    st_module: Any,
    service: WebRAGService,
    history: list[dict[str, Any]],
    item: dict[str, Any],
    user_email: str,
    collection: str,
    db_path: Path,
) -> None:
    interaction_id = str(item.get("interaction_id", ""))
    feedback_value = str(item.get("feedback", "")).strip()
    feedback_comment = str(item.get("feedback_comment", "")).strip()
    turn_id = item.get("db_turn_id")

    if feedback_value:
        display_text = "Câu trả lời tốt 👍" if feedback_value == "like" else "Chưa chính xác 👎"
        st_module.markdown(
            f"<span class='rag-feedback-chip'>{display_text}</span>",
            unsafe_allow_html=True,
        )

        if isinstance(turn_id, int):
            with st_module.expander("📝 Góp ý thêm cho hệ thống"):
                comment_input = st_module.text_area(
                    "Nhập chi tiết góp ý của bạn",
                    value=feedback_comment,
                    key=f"feedback-comment-{turn_id}",
                    height=80,
                    label_visibility="collapsed"
                )
                if st_module.button("Gửi", key=f"feedback-save-{turn_id}"):
                    save_feedback(db_path, turn_id, feedback_value, comment_input.strip() or None)
                    item["feedback_comment"] = comment_input.strip()
                    st_module.session_state["chat_history"] = history
                    st_module.rerun()
        return

    st_module.write("") # small gap
    like_col, dislike_col, empty = st_module.columns([1, 1, 4])
    if like_col.button("👍", key=f"feedback-like-{interaction_id}", help="Câu trả lời hữu ích"):
        service.record_feedback(
            interaction_id=interaction_id,
            user_email=user_email,
            collection=collection,
            score="like",
            question=str(item.get("question", "")),
            answer=str(item.get("answer", "")),
            cache_hit=bool(item.get("cache_hit", False)),
        )
        if isinstance(turn_id, int):
            save_feedback(db_path, turn_id, "like", None)
        item["feedback"] = "like"
        st_module.session_state["chat_history"] = history
        st_module.rerun()

    if dislike_col.button("👎", key=f"feedback-dislike-{interaction_id}", help="Câu trả lời chưa tốt"):
        service.record_feedback(
            interaction_id=interaction_id,
            user_email=user_email,
            collection=collection,
            score="dislike",
            question=str(item.get("question", "")),
            answer=str(item.get("answer", "")),
            cache_hit=bool(item.get("cache_hit", False)),
        )
        if isinstance(turn_id, int):
            save_feedback(db_path, turn_id, "dislike", None)
        item["feedback"] = "dislike"
        st_module.session_state["chat_history"] = history
        st_module.rerun()


def _render_source_buttons(st_module: Any, item: dict[str, Any]) -> None:
    sources = item.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return

    interaction_id = str(item.get("interaction_id", ""))
    st_module.caption("📚 Nguồn tham khảo:")
    
    # Render sources in a neat row
    cols = st_module.columns(len(sources) if len(sources) < 4 else 3)
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        label = _build_citation_label(source, index)
        col = cols[(index - 1) % len(cols)]
        if col.button(label, key=f"citation-{interaction_id}-{index}", use_container_width=True):
            st_module.session_state["selected_citation"] = source
            st_module.rerun()


def _render_chat_history(
    st_module: Any,
    service: WebRAGService,
    history: list[dict[str, Any]],
    user_email: str,
    collection: str,
    db_path: Path,
) -> None:
    for item in history:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        with st_module.chat_message("user"):
            st_module.write(question)

        # Using a distinct icon for assistant to mimic Gemini (sparkles)
        with st_module.chat_message("assistant", avatar="✨"):
            st_module.markdown(answer)
            _render_answer_metrics(st_module, item)
            _render_source_buttons(st_module, item)
            _render_feedback_controls(st_module, service, history, item, user_email, collection, db_path)


def _render_pdf_panel(st_module: Any, components_module: Any, selected_citation: dict[str, Any] | None) -> None:
    if not selected_citation:
        st_module.markdown(
            """
            <div style='text-align: center; color: var(--gemini-subtext); margin-top: 50%;'>
                <p style='font-size: 2rem;'>📑</p>
                <p>Nhấp vào các thẻ tham khảo bên cạnh <br>để xem chi tiết tài liệu tại đây.</p>
            </div>
            """, unsafe_allow_html=True
        )
        return

    source_name = str(selected_citation.get("source", "unknown"))
    source_path = str(selected_citation.get("source_path", "")).strip()
    page_number = selected_citation.get("page_number")
    preview = str(selected_citation.get("preview", "")).strip()

    st_module.markdown(f"### {source_name}")
    if isinstance(page_number, int):
        st_module.caption(f"📍 Hiển thị trang: {page_number}")

    if preview:
        st_module.markdown(
            f"<div class='rag-preview'><i>\"{html.escape(preview)}\"</i></div>",
            unsafe_allow_html=True,
        )

    path = Path(source_path)
    if not source_path or not path.exists():
        st_module.warning("Tài liệu này không có sẵn PDF để xem trực tiếp.")
        return

    try:
        pdf_base64 = _load_pdf_base64(str(path.resolve()))
    except Exception as error:  # noqa: BLE001
        st_module.error(f"Lỗi tải tài liệu: {error}")
        return

    safe_page = page_number if isinstance(page_number, int) and page_number > 0 else 1
    search_query = quote_plus(" ".join(preview.split()[:8])) if preview else ""
    iframe_src = f"data:application/pdf;base64,{pdf_base64}#page={safe_page}&search={search_query}"
    st_module.markdown("<br>", unsafe_allow_html=True)
    components_module.html(
        f"<iframe src='{iframe_src}' width='100%' height='700' style='border:1px solid #E3E3E3; border-radius:12px; background: white;'></iframe>",
        height=710,
    )


def _finalize_login(
    st_module: Any,
    db_path: Path,
    auth: StudentAuthResult,
) -> None:
    assert auth.student_id is not None
    assert auth.mssv is not None
    st_module.session_state["logged_in"] = True
    st_module.session_state["user_id"] = auth.mssv
    st_module.session_state["student_id"] = auth.student_id
    chat_session_id = ensure_chat_session(db_path, int(auth.student_id))
    st_module.session_state["chat_session_id"] = chat_session_id
    st_module.session_state["chat_history"] = load_chat_history(db_path, chat_session_id)
    record_login(db_path, int(auth.student_id))


def _render_login_screen(st_module: Any, db_path: Path) -> bool:
    if st_module.session_state.get("logged_in", False):
        return True

    reset_student_id = st_module.session_state.get("password_reset_student_id")
    reset_mssv = st_module.session_state.get("password_reset_mssv")
    if isinstance(reset_student_id, int) and reset_mssv:
        st_module.markdown(
            """
            <div class="sgu-login-box">
                <h2 style="margin-bottom: 0.5rem; font-weight: 500;">Đổi mật khẩu</h2>
                <p style="color: var(--gemini-subtext); margin-bottom: 2rem; font-size: 0.9rem;">
                    Vì lý do bảo mật, vui lòng đổi mật khẩu trong lần đăng nhập đầu tiên.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st_module.columns([1, 1.5, 1])
        with col2:
            with st_module.form("password_reset_form"):
                new_password = st_module.text_input("Mật khẩu mới", type="password")
                confirm_password = st_module.text_input("Xác nhận mật khẩu", type="password")
                st_module.markdown("<br>", unsafe_allow_html=True)
                submitted = st_module.form_submit_button("Cập nhật & Đăng nhập", use_container_width=True)

                if submitted:
                    if not new_password.strip():
                        st_module.error("Mật khẩu mới không được để trống.")
                    elif new_password != confirm_password:
                        st_module.error("Mật khẩu xác nhận không khớp.")
                    else:
                        update_student_password(db_path, int(reset_student_id), new_password)
                        auth = StudentAuthResult(
                            True,
                            int(reset_student_id),
                            str(reset_mssv),
                            None,
                            False,
                            None,
                        )
                        st_module.session_state.pop("password_reset_student_id", None)
                        st_module.session_state.pop("password_reset_mssv", None)
                        _finalize_login(st_module, db_path, auth)
                        st_module.rerun()

        return False

    st_module.markdown(
        """
        <div class="sgu-login-box">
            <h1 style="background: linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%, #9b72cb 50%, #d96570 56%, #131314 75%, #131314 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600; font-size: 2.2rem; margin-bottom: 0.5rem;">SGU RAG SYSTEM</h1>
            <p style="color: var(--gemini-subtext); margin-bottom: 2rem;">Đăng nhập để bắt đầu trò chuyện</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st_module.columns([1, 1.5, 1])
    with col2:
        with st_module.form("login_form"):
            mssv = st_module.text_input("Mã số sinh viên (MSSV)", placeholder="Ví dụ: 3120...")
            password = st_module.text_input("Mật khẩu", type="password", placeholder="Mặc định: Ngày sinh (vd: 01/01/2000)")
            st_module.markdown("<br>", unsafe_allow_html=True)
            submitted = st_module.form_submit_button("Đăng nhập", use_container_width=True)

            if submitted:
                if not mssv.strip() or not password.strip():
                    st_module.warning("Vui lòng nhập đầy đủ MSSV và mật khẩu.")
                else:
                    auth = authenticate_student(db_path, mssv, password)
                    if not auth.ok:
                        error_map = {
                            "missing_mssv": "Vui lòng nhập MSSV.",
                            "not_found": "Không tìm thấy MSSV này trong hệ thống.",
                            "bad_password": "Mật khẩu không chính xác.",
                        }
                        error_key = auth.error or ""
                        st_module.error(error_map.get(error_key, "Đăng nhập thất bại"))
                    elif auth.must_change_password:
                        st_module.session_state["password_reset_student_id"] = auth.student_id
                        st_module.session_state["password_reset_mssv"] = auth.mssv
                        st_module.rerun()
                    else:
                        _finalize_login(st_module, db_path, auth)
                        st_module.rerun()

        if count_students(db_path) == 0:
            st_module.caption("Hệ thống chưa có dữ liệu sinh viên. Cần seed từ `artifacts/data/students_seed.csv`.")

    return st_module.session_state.get("logged_in", False)


def main() -> None:
    import streamlit as st  # type: ignore[import-not-found]
    import streamlit.components.v1 as components  # type: ignore[import-not-found]

    st.set_page_config(
        page_title="SGU QA System",
        page_icon="✨",
        layout="wide",
    )

    _inject_custom_css(st)

    base_dir = str(Path.cwd())
    db_path = _ensure_db(Path(base_dir))

    if not _render_login_screen(st, db_path):
        st.stop()
        return

    # Sidebar: Minimalist like Google
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://sgu.edu.vn/wp-content/uploads/2021/04/Logo-SGU-2.png", width=120)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**👤 Xin chào, {st.session_state.get('user_id')}**")
        
        st.divider()
        
        if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
            session_id = st.session_state.get("chat_session_id")
            if isinstance(session_id, int):
                clear_chat_session(db_path, session_id)
            st.session_state["chat_history"] = []
            service = _ensure_service(st, base_dir)
            user_email = f"{st.session_state.get('user_id', 'student')}@sgu.edu.vn"
            service.clear_memory_context(f"{user_email}::All Documents")
            st.rerun()
            
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("Đăng xuất", type="secondary", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state.pop("chat_session_id", None)
            st.session_state.pop("chat_history", None)
            st.rerun()

    try:
        service = _ensure_service(st, base_dir)
    except Exception as error:  # noqa: BLE001
        st.error(f"Lỗi khởi tạo hệ thống: {error}")
        if st.button("Thử lại"):
            st.rerun()
        st.stop()
        return

    user_email = f"{st.session_state.get('user_id', 'student')}@sgu.edu.vn"
    collection = "All Documents"
    conversation_key = f"{user_email}::{collection}"

    if not service.index_ready:
        st.error("Chưa có cơ sở dữ liệu học thuật. Vui lòng liên hệ Quản trị viên.")
        st.stop()
        return

    student_id = st.session_state.get("student_id")
    chat_session_id = st.session_state.get("chat_session_id")
    if isinstance(student_id, int) and not isinstance(chat_session_id, int):
        chat_session_id = ensure_chat_session(db_path, student_id)
        st.session_state["chat_session_id"] = chat_session_id

    if "chat_history" not in st.session_state:
        if isinstance(chat_session_id, int):
            st.session_state["chat_history"] = load_chat_history(db_path, chat_session_id)
        else:
            st.session_state["chat_history"] = []
    
    history = st.session_state["chat_history"]
    
    chat_col, gap, citation_col = st.columns([1.8, 0.1, 1.2])

    with chat_col:
        # Show Gemini-like greeting only if there's no chat history
        if not history and not st.session_state.get("pending_question"):
            st.markdown(
                f"""
                <div class="gemini-greeting-container">
                    <div class="gemini-greeting">Xin chào, {st.session_state.get('user_id')}</div>
                    <div class="gemini-sub-greeting">Tôi có thể giúp gì cho bạn hôm nay?</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        msg_container = st.container(height=650, border=False)
        
        pending_question = str(st.session_state.pop("pending_question", "")).strip()

        with msg_container:
            _render_chat_history(
                st_module=st,
                service=service,
                history=history,
                user_email=user_email,
                collection=collection,
                db_path=db_path,
            )

            if pending_question:
                with st.chat_message("user"):
                    st.write(pending_question)

                with st.chat_message("assistant", avatar="✨"):
                    skeleton_placeholder = st.empty()
                    skeleton_placeholder.markdown(_skeleton_markup(), unsafe_allow_html=True)
                    answer_placeholder = st.empty()

                    try:
                        top_k = int(os.getenv("WEB_RETRIEVAL_K", str(service.settings.retrieval_k)))
                        top_k = max(1, min(top_k, 8))
                        query_handle = service.start_query(
                            question=pending_question,
                            user_email=user_email,
                            collection=collection,
                            conversation_key=conversation_key,
                            top_k=top_k,
                        )
                    except Exception as error:  # noqa: BLE001
                        skeleton_placeholder.empty()
                        st.error(f"Đã xảy ra lỗi: {error}")
                    else:
                        skeleton_placeholder.empty()
                        
                        if query_handle.cache_hit:
                            stream_obj = service.stream_cached_answer(query_handle.cached_answer or "")
                        else:
                            assert query_handle.stream_session is not None
                            stream_obj = query_handle.stream_session.token_generator()

                        rendered_answer_text = answer_placeholder.write_stream(stream_obj)

                        if not isinstance(rendered_answer_text, str):
                            rendered_answer_text = "".join(str(item) for item in rendered_answer_text)

                        result = service.finalize_query(
                            query_handle,
                            rendered_answer=rendered_answer_text,
                        )
                        if isinstance(chat_session_id, int):
                            sources_json = json.dumps(result.get("sources", []), ensure_ascii=False)
                            timings_json = json.dumps(result.get("timings", {}), ensure_ascii=False)
                            turn_id = save_chat_turn(
                                db_path,
                                chat_session_id,
                                result.get("interaction_id"),
                                pending_question,
                                rendered_answer_text,
                                sources_json,
                                timings_json,
                                bool(result.get("cache_hit", False)),
                            )
                            result["db_turn_id"] = turn_id
                        history.append(result)
                        st.session_state["chat_history"] = history
                        st.rerun()

        prompt = st.chat_input("Hỏi trợ lý SGU bất cứ điều gì...")
        if prompt:
            st.session_state["pending_question"] = prompt.strip()
            st.rerun() 

    with citation_col:
        _render_pdf_panel(
            st_module=st,
            components_module=components,
            selected_citation=st.session_state.get("selected_citation"),
        )