import hashlib
import hmac
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_core.config import RAGConfig
from rag_core.environment import configure_runtime_environment

configure_runtime_environment()

from rag_core.pipeline import RAGPipeline
from rag_core.qa_service import RAGService

BIRTH_DATE_FORMAT = "%d/%m/%Y"
PASSWORD_HASH_ITERATIONS = 120_000

ROLE_USER = "user"
ROLE_ADMIN = "admin"
VALID_ROLES = {ROLE_USER, ROLE_ADMIN}
ROLE_LABELS = {
    ROLE_USER: "Người dùng",
    ROLE_ADMIN: "Quản trị viên",
}

MENU_CHAT = "Trò chuyện"
MENU_USER_MANAGEMENT = "Quản lý người dùng"
MENU_CHAT_LOGS = "Nhật ký hỏi đáp"
MENU_MY_HISTORY = "Lịch sử của tôi"

ADMIN_MENU_OPTIONS = [MENU_CHAT, MENU_USER_MANAGEMENT, MENU_CHAT_LOGS]
USER_MENU_OPTIONS = [MENU_CHAT, MENU_MY_HISTORY]

DEFAULT_ADMIN_MSSV = "admin"
DEFAULT_ADMIN_BIRTH_DATE = "01/01/2000"

CHAT_LOG_MIN_LIMIT = 1
CHAT_LOG_MAX_LIMIT = 500


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PASSWORD_HASH_ITERATIONS,
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def normalize_birth_date(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None

    try:
        parsed = datetime.strptime(raw, BIRTH_DATE_FORMAT)
    except ValueError:
        return None

    return parsed.strftime(BIRTH_DATE_FORMAT)


def get_admin_credentials() -> Tuple[str, str]:
    admin_mssv = os.getenv("RAG_ADMIN_MSSV", DEFAULT_ADMIN_MSSV)
    admin_birth_date = os.getenv("RAG_ADMIN_BIRTH_DATE", DEFAULT_ADMIN_BIRTH_DATE)
    return admin_mssv, admin_birth_date


def result_ok(message: str) -> Dict[str, str]:
    return {"status": "ok", "message": message}


def result_error(message: str) -> Dict[str, str]:
    return {"status": "error", "message": message}


def write_sources(sources: List[str]) -> None:
    for source in sources:
        st.write(f"- {source}")


def build_retrieved_passage_previews(
    rag: RAGService,
    docs: List[Any],
    max_items: int = 5,
    max_chars: int = 180,
) -> List[str]:
    previews: List[str] = []
    for idx, doc in enumerate(docs[:max_items], start=1):
        content = str(getattr(doc, "page_content", "")).replace("\n", " ").strip()
        if not content:
            continue

        label = rag._extract_source_label(doc, idx)
        short_content = content[:max_chars].rstrip()
        if len(content) > max_chars:
            short_content += "..."
        previews.append(f"{label}: {short_content}")
    return previews


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    mssv TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    birth_date TEXT,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                )
                """
            )

            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "birth_date" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN birth_date TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mssv TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(mssv) REFERENCES users(mssv)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_logs_mssv_time ON chat_logs(mssv, created_at)"
            )
            # One-time role normalization for existing records.
            conn.execute("UPDATE users SET role = ? WHERE role = 'student'", (ROLE_USER,))
            conn.commit()

    def ensure_admin(self, mssv: str, birth_date: str, full_name: str = "System Admin") -> None:
        normalized_birth_date = normalize_birth_date(birth_date)
        if normalized_birth_date is None:
            raise ValueError("Ngày sinh admin mặc định không đúng định dạng dd/mm/yyyy.")

        user = self.get_user(mssv)
        if user is None:
            self.create_user(
                mssv=mssv,
                full_name=full_name,
                birth_date=normalized_birth_date,
                role=ROLE_ADMIN,
            )
            return

        updates: List[str] = []
        values: List[str] = []

        if str(user.get("role") or "") != ROLE_ADMIN:
            updates.append("role = ?")
            values.append(ROLE_ADMIN)

        # Migrate legacy admin accounts that were password-based.
        if not str(user.get("birth_date") or "").strip():
            updates.append("birth_date = ?")
            values.append(normalized_birth_date)
            updates.append("password_hash = ?")
            values.append(hash_password(normalized_birth_date))

        if updates:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE mssv = ?",
                    (*values, mssv),
                )
                conn.commit()

    def get_user(self, mssv: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE mssv = ?", (mssv,)).fetchone()
            return dict(row) if row else None

    def create_user(
        self,
        mssv: str,
        full_name: str,
        birth_date: str,
        role: str = ROLE_USER,
    ) -> Dict[str, str]:
        mssv = mssv.strip()
        full_name = full_name.strip()
        normalized_birth_date = normalize_birth_date(birth_date)

        if not mssv or not full_name or normalized_birth_date is None:
            return result_error(
                "Thông tin tạo tài khoản không hợp lệ. Ngày sinh phải theo định dạng dd/mm/yyyy."
            )
        if role not in VALID_ROLES:
            return result_error("Vai trò không hợp lệ.")
        if self.get_user(mssv):
            return result_error(f"MSSV {mssv} đã tồn tại.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (mssv, full_name, birth_date, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    mssv,
                    full_name,
                    normalized_birth_date,
                    hash_password(normalized_birth_date),
                    role,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

        return result_ok(f"Đã tạo tài khoản {mssv}.")

    def delete_user(self, mssv: str, actor_mssv: str) -> Dict[str, str]:
        mssv = mssv.strip()
        if not mssv:
            return result_error("MSSV không hợp lệ.")
        if mssv == actor_mssv:
            return result_error("Không thể xóa tài khoản đang đăng nhập.")

        with self._connect() as conn:
            found = conn.execute("SELECT mssv FROM users WHERE mssv = ?", (mssv,)).fetchone()
            if not found:
                return result_error(f"Không tìm thấy MSSV {mssv}.")

            conn.execute("DELETE FROM users WHERE mssv = ?", (mssv,))
            conn.execute("DELETE FROM chat_logs WHERE mssv = ?", (mssv,))
            conn.commit()

        return result_ok(f"Đã xóa tài khoản {mssv}.")

    def authenticate(self, mssv: str, birth_date: str) -> Optional[Dict[str, Any]]:
        user = self.get_user(mssv.strip())
        if not user:
            return None

        stored_birth_date = str(user.get("birth_date") or "").strip()

        if stored_birth_date:
            normalized_birth_date = normalize_birth_date(birth_date)
            if normalized_birth_date is None:
                return None
            if not hmac.compare_digest(stored_birth_date, normalized_birth_date):
                return None
            if not verify_password(normalized_birth_date, str(user["password_hash"])):
                return None
        else:
            # Backward compatibility for legacy password-based records.
            if not verify_password(birth_date, str(user["password_hash"])):
                return None

        user.pop("password_hash", None)
        return user

    def list_users(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT mssv, full_name, birth_date, role, created_at FROM users ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def save_chat_log(self, mssv: str, question: str, answer: str, sources: List[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_logs (mssv, question, answer, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    mssv,
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_chat_logs(self, limit: int = 100, mssv: Optional[str] = None) -> List[Dict[str, Any]]:
        safe_limit = max(CHAT_LOG_MIN_LIMIT, min(limit, CHAT_LOG_MAX_LIMIT))
        with self._connect() as conn:
            if mssv:
                rows = conn.execute(
                    """
                    SELECT id, mssv, question, answer, sources, created_at
                    FROM chat_logs
                    WHERE mssv = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (mssv, safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, mssv, question, answer, sources, created_at
                    FROM chat_logs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()

        logs: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["sources"] = json.loads(item.get("sources") or "[]")
            except json.JSONDecodeError:
                item["sources"] = []
            logs.append(item)
        return logs


@st.cache_resource(show_spinner=False)
def get_database_manager(db_path: Path) -> DatabaseManager:
    return DatabaseManager(db_path)


@st.cache_resource(show_spinner=False)
def get_rag_service(config: RAGConfig) -> RAGService:
    return RAGService(config)


def build_rag_index(config: RAGConfig) -> Tuple[RAGService, Optional[str]]:
    """Build index from PDFs (like notebook), fallback to load existing index.

    Returns (rag_service, status_message).
    """
    pipeline = RAGPipeline(config)
    try:
        result = pipeline.build_index()
        pdf_count = len(result.ingestion.loaded_pdf_files)
        doc_count = len(result.ingestion.documents)
        chunk_count = len(result.chunks)
        msg = (
            f"Đã build index thành công từ PDF.\n"
            f"- PDF đã nạp: {pdf_count}\n"
            f"- Tài liệu trích xuất: {doc_count}\n"
            f"- Chunks tạo ra: {chunk_count}"
        )
        if result.ingestion.scanned_pdf_files:
            skipped = ", ".join(result.ingestion.scanned_pdf_files)
            msg += f"\n- PDF bỏ qua (scan/không có text): {skipped}"
        return pipeline.rag_service, msg
    except Exception as build_exc:
        try:
            pipeline.load_index()
            return pipeline.rag_service, (
                f"Không thể build từ PDF ({build_exc}). "
                "Đã nạp index có sẵn từ disk."
            )
        except Exception as load_exc:
            return pipeline.rag_service, (
                f"Không thể build ({build_exc}) "
                f"và không thể load index ({load_exc})."
            )


def init_session_state() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("rag_reloaded_notice", False)


def logout() -> None:
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.session_state["chat_messages"] = []


def render_login(db: DatabaseManager) -> None:
    st.title("Hệ thống Hỏi đáp Tài liệu SGU")
    st.caption("Đăng nhập bằng MSSV và ngày sinh để sử dụng")

    with st.form("login_form"):
        mssv = st.text_input("MSSV", key="login_mssv")
        birth_date = st.text_input(
            "Ngày sinh (dd/mm/yyyy)",
            placeholder="Ví dụ: 02/09/2004",
            key="login_birth_date",
        )
        submitted = st.form_submit_button("Đăng nhập")

    if submitted:
        user = db.authenticate(mssv, birth_date)
        if user is None:
            st.error("MSSV hoặc ngày sinh không đúng. Ngày sinh phải theo định dạng dd/mm/yyyy.")
        else:
            st.session_state["logged_in"] = True
            st.session_state["user"] = user
            st.success(f"Xin chào {user['full_name']} ({user['mssv']})")
            st.rerun()

    admin_mssv, admin_birth_date = get_admin_credentials()
    st.info(
        f"Tài khoản admin mặc định: mssv={admin_mssv}, ngày sinh={admin_birth_date}. "
        "Ngày sinh phải theo định dạng dd/mm/yyyy."
    )


def render_sidebar(user: Dict[str, Any], config: RAGConfig) -> str:
    current_role_label = role_label(str(user.get("role", "")))

    st.sidebar.header("Tài khoản")
    st.sidebar.write(f"MSSV: {user['mssv']}")
    st.sidebar.write(f"Họ tên: {user['full_name']}")
    st.sidebar.write(f"Vai trò: {current_role_label}")

    if st.sidebar.button("Nạp lại RAG index", key="sidebar_reload_rag"):
        # Giống notebook: thử build_index() từ PDF trước, fallback load_index().
        with st.sidebar.status("Đang build lại index...", expanded=True) as status:
            get_rag_service.clear()
            new_rag, build_msg = build_rag_index(config)
            st.session_state["_rebuilt_rag"] = new_rag
            status.update(label=build_msg, state="complete")
        st.session_state["chat_messages"] = []
        st.session_state["rag_reloaded_notice"] = True
        st.rerun()

    if st.sidebar.button("Đăng xuất", key="sidebar_logout"):
        logout()
        st.rerun()

    if user["role"] == ROLE_ADMIN:
        return st.sidebar.radio("Menu", ADMIN_MENU_OPTIONS, index=0, key="sidebar_menu")
    return st.sidebar.radio("Menu", USER_MENU_OPTIONS, index=0, key="sidebar_menu")


def render_chat_page(db: DatabaseManager, rag: RAGService, user: Dict[str, Any], config: RAGConfig) -> None:
    st.title("Trò chuyện với Tài liệu")

    with st.sidebar.expander("Tham số trò chuyện", expanded=False):
        ui_top_k = st.slider(
            "Số đoạn truy xuất (top-k)",
            min_value=1,
            max_value=20,
            value=max(1, int(config.retrieval_k)),
            step=1,
            key="chat_top_k",
        )
        ui_show_passage_previews = st.checkbox(
            "Hiển thị đoạn truy xuất",
            value=True,
            key="chat_show_passage_previews",
        )
        ui_preview_items = st.slider(
            "Số đoạn hiển thị",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="chat_preview_items",
            disabled=not ui_show_passage_previews,
        )
        ui_preview_chars = st.slider(
            "Số ký tự mỗi đoạn",
            min_value=80,
            max_value=500,
            value=180,
            step=20,
            key="chat_preview_chars",
            disabled=not ui_show_passage_previews,
        )

    status = "Có Gemini" if rag.llm is not None else "Chỉ truy xuất"
    status_detail = f" ({rag.llm_unavailable_reason})" if rag.llm is None and rag.llm_unavailable_reason else ""
    retriever_status = rag.retriever_status
    if rag.retriever_last_error:
        retriever_status = f"{retriever_status} ({rag.retriever_last_error})"
    st.caption(
        f"Embedding: {config.embedding_model} | LLM: {status}{status_detail} | "
        f"Temperature: {config.llm_temperature} | Truy xuất: {retriever_status} | Top-k: {ui_top_k} | "
        f"Thư mục PDF: {config.pdf_dir}"
    )

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Nguồn tham khảo"):
                    write_sources(message["sources"])
            if ui_show_passage_previews and message.get("passage_previews"):
                with st.expander("Top đoạn truy xuất"):
                    write_sources(message["passage_previews"])

    question = st.chat_input("Nhập câu hỏi về tài liệu...", key="chat_question_input")
    if not question:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy vấn hệ thống RAG..."):
            result = rag.query(question, top_k=ui_top_k)
            answer = result["answer"]
            sources = result["sources"]
            if ui_show_passage_previews:
                passage_previews = build_retrieved_passage_previews(
                    rag,
                    result.get("docs", []),
                    max_items=ui_preview_items,
                    max_chars=ui_preview_chars,
                )
            else:
                passage_previews = []

        st.markdown(answer)
        if sources:
            with st.expander("Nguồn tham khảo"):
                write_sources(sources)
        if passage_previews:
            with st.expander("Top đoạn truy xuất"):
                write_sources(passage_previews)

    db.save_chat_log(user["mssv"], question, answer, sources)
    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "passage_previews": passage_previews,
        }
    )


def render_user_management(db: DatabaseManager, user: Dict[str, Any]) -> None:
    st.title("Quản lý người dùng")

    create_tab, list_tab, delete_tab = st.tabs(["Tạo tài khoản", "Danh sách tài khoản", "Xóa tài khoản"])

    with create_tab:
        with st.form("create_user_form"):
            role_options = {"Người dùng": ROLE_USER, "Quản trị viên": ROLE_ADMIN}

            new_mssv = st.text_input("MSSV mới", key="create_user_mssv")
            new_name = st.text_input("Họ tên", key="create_user_name")
            new_birth_date = st.text_input(
                "Ngày sinh (dd/mm/yyyy)",
                placeholder="Ví dụ: 02/09/2004",
                key="create_user_birth_date",
            )
            new_role_label = st.selectbox(
                "Vai trò",
                list(role_options.keys()),
                index=0,
                key="create_user_role",
            )
            create_submitted = st.form_submit_button("Tạo tài khoản")

        if create_submitted:
            outcome = db.create_user(
                mssv=new_mssv,
                full_name=new_name,
                birth_date=new_birth_date,
                role=role_options[new_role_label],
            )
            if outcome["status"] == "ok":
                st.success(outcome["message"])
            else:
                st.error(outcome["message"])

    with list_tab:
        users = db.list_users()
        st.write(f"Tổng số tài khoản: {len(users)}")

        display_users: List[Dict[str, Any]] = []
        for item in users:
            display_users.append(
                {
                    "MSSV": item.get("mssv", ""),
                    "Họ tên": item.get("full_name", ""),
                    "Ngày sinh": item.get("birth_date", ""),
                    "Vai trò": role_label(str(item.get("role", ""))),
                    "Ngày tạo": item.get("created_at", ""),
                }
            )

        st.dataframe(display_users, use_container_width=True, hide_index=True)

    with delete_tab:
        users = db.list_users()
        all_mssv = [item["mssv"] for item in users if item["mssv"] != user["mssv"]]
        if not all_mssv:
            st.warning("Không có tài khoản nào để xóa.")
            return

        target = st.selectbox("Chọn MSSV cần xóa", all_mssv, key="delete_user_target")
        if st.button("Xác nhận xóa", type="primary", key="delete_user_submit"):
            outcome = db.delete_user(target, actor_mssv=user["mssv"])
            if outcome["status"] == "ok":
                st.success(outcome["message"])
                st.rerun()
            else:
                st.error(outcome["message"])


def render_logs(db: DatabaseManager, user: Dict[str, Any], admin_mode: bool) -> None:
    st.title("Nhật ký hỏi đáp")
    limit = st.slider(
        "Số bản ghi",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        key="chat_logs_limit",
    )

    mssv_filter = None if admin_mode else user["mssv"]
    logs = db.get_chat_logs(limit=limit, mssv=mssv_filter)

    if not logs:
        st.info("Chưa có dữ liệu chat.")
        return

    for item in logs:
        title = f"[{item['created_at']}] {item['mssv']} - Q: {item['question'][:80]}"
        with st.expander(title):
            st.markdown("**Câu hỏi**")
            st.write(item["question"])
            st.markdown("**Trả lời**")
            st.write(item["answer"])
            if item.get("sources"):
                st.markdown("**Nguồn tham khảo**")
                write_sources(item["sources"])


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Hỏi đáp Tài liệu SGU", page_icon=":books:", layout="wide")

    config = RAGConfig.from_env(base_dir=BASE_DIR)
    db = get_database_manager(config.db_path)

    admin_mssv, admin_birth_date = get_admin_credentials()
    db.ensure_admin(admin_mssv, admin_birth_date)

    init_session_state()

    if not st.session_state["logged_in"]:
        render_login(db)
        return

    user = st.session_state["user"]

    # Nếu vừa rebuild index, sử dụng RAGService mới; nếu không, lazy-load.
    rebuilt_rag = st.session_state.pop("_rebuilt_rag", None)
    if rebuilt_rag is not None:
        rag = rebuilt_rag
    else:
        rag = get_rag_service(config)
        # Start retriever warm-up after login so first query is faster.
        rag.start_preload()

    menu = render_sidebar(user, config)

    if st.session_state.get("rag_reloaded_notice"):
        st.success("Đã nạp lại RAG index theo dữ liệu mới nhất.")
        st.session_state["rag_reloaded_notice"] = False

    if menu == MENU_CHAT:
        try:
            render_chat_page(db, rag, user, config)
        except Exception as exc:
            st.error(f"Không thể khởi tạo dịch vụ RAG: {exc}")
            st.info(
                "Hãy đảm bảo đã có FAISS index trong thư mục vector_store "
                "và đã cài đúng dependencies trong requirements.txt"
            )

    elif menu == MENU_USER_MANAGEMENT:
        render_user_management(db, user)

    elif menu == MENU_CHAT_LOGS:
        render_logs(db, user, admin_mode=True)

    elif menu == MENU_MY_HISTORY:
        render_logs(db, user, admin_mode=False)


if __name__ == "__main__":
    main()