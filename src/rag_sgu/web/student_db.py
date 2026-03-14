from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_NAME = "sgu_app.db"
DEFAULT_SEED_FILE = "students_seed.csv"
PASSWORD_ITERATIONS = 150_000


@dataclass
class StudentAuthResult:
    ok: bool
    student_id: int | None
    mssv: str | None
    full_name: str | None
    must_change_password: bool
    error: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db_path(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "data" / DEFAULT_DB_NAME


def get_seed_path(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "data" / DEFAULT_SEED_FILE


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mssv TEXT UNIQUE NOT NULL,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                must_change_password INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                title TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                interaction_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT,
                timings_json TEXT,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                score TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(turn_id) REFERENCES chat_turns(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_students_mssv ON students(mssv);
            CREATE INDEX IF NOT EXISTS idx_sessions_student ON chat_sessions(student_id);
            CREATE INDEX IF NOT EXISTS idx_turns_session ON chat_turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_turn ON feedback(turn_id);
            """
        )
        conn.commit()


def _hash_password(password: str, *, iterations: int = PASSWORD_ITERATIONS) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "{}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("utf-8"),
        base64.b64encode(dk).decode("utf-8"),
    )


def _verify_password(password: str, hashed: str) -> bool:
    try:
        parts = hashed.split("$")
        if len(parts) != 3:
            return False
        iterations = int(parts[0])
        salt = base64.b64decode(parts[1].encode("utf-8"))
        expected = base64.b64decode(parts[2].encode("utf-8"))
    except (ValueError, OSError):
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def upsert_student(
    db_path: Path,
    mssv: str,
    password: str,
    full_name: str | None = None,
    must_change_password: bool = True,
) -> int:
    normalized = mssv.strip()
    if not normalized:
        raise ValueError("MSSV must not be empty")

    password_hash = _hash_password(password)
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM students WHERE mssv = ?",
            (normalized,),
        ).fetchone()
        if existing:
            return int(existing["id"])

        now = _utc_now()
        cursor = conn.execute(
            """
            INSERT INTO students (mssv, full_name, password_hash, must_change_password, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalized, full_name, password_hash, 1 if must_change_password else 0, now),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Failed to create student")
        return int(cursor.lastrowid)


def seed_students_from_csv(db_path: Path, csv_path: Path) -> int:
    if not csv_path.exists():
        return 0

    inserted = 0
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mssv = str(row.get("mssv", "")).strip()
            password = str(row.get("password", "")).strip()
            full_name = str(row.get("full_name", "")).strip() or None
            must_change_raw = str(row.get("must_change_password", "1")).strip()
            must_change = must_change_raw not in {"0", "false", "False", "no", "NO"}
            if not mssv or not password:
                continue
            upsert_student(db_path, mssv, password, full_name=full_name, must_change_password=must_change)
            inserted += 1

    return inserted


def count_students(db_path: Path) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(1) as total FROM students").fetchone()
        return int(row["total"]) if row else 0


def authenticate_student(db_path: Path, mssv: str, password: str) -> StudentAuthResult:
    normalized = mssv.strip()
    if not normalized:
        return StudentAuthResult(False, None, None, None, False, "missing_mssv")

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, mssv, full_name, password_hash, must_change_password
            FROM students
            WHERE mssv = ?
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        return StudentAuthResult(False, None, None, None, False, "not_found")

    if not _verify_password(password, str(row["password_hash"])):
        return StudentAuthResult(False, None, None, None, False, "bad_password")

    return StudentAuthResult(
        True,
        int(row["id"]),
        str(row["mssv"]),
        str(row["full_name"]) if row["full_name"] else None,
        bool(row["must_change_password"]),
        None,
    )


def update_student_password(db_path: Path, student_id: int, new_password: str) -> None:
    password_hash = _hash_password(new_password)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE students
            SET password_hash = ?, must_change_password = 0
            WHERE id = ?
            """,
            (password_hash, student_id),
        )
        conn.commit()


def record_login(db_path: Path, student_id: int) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE students SET last_login_at = ? WHERE id = ?",
            (_utc_now(), student_id),
        )
        conn.commit()


def create_chat_session(db_path: Path, student_id: int, title: str | None = None) -> int:
    now = _utc_now()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_sessions (student_id, started_at, last_activity_at, title)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, now, now, title),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Failed to create chat session")
        return int(cursor.lastrowid)


def ensure_chat_session(db_path: Path, student_id: int, force_new: bool = False) -> int:
    if force_new:
        return create_chat_session(db_path, student_id)

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id FROM chat_sessions
            WHERE student_id = ?
            ORDER BY last_activity_at DESC
            LIMIT 1
            """,
            (student_id,),
        ).fetchone()

    if row is None:
        return create_chat_session(db_path, student_id)

    return int(row["id"])


def clear_chat_session(db_path: Path, session_id: int) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM feedback WHERE turn_id IN (SELECT id FROM chat_turns WHERE session_id = ?)", (session_id,))
        conn.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
        conn.execute(
            "UPDATE chat_sessions SET last_activity_at = ? WHERE id = ?",
            (_utc_now(), session_id),
        )
        conn.commit()


def save_chat_turn(
    db_path: Path,
    session_id: int,
    interaction_id: str | None,
    question: str,
    answer: str,
    sources_json: str | None,
    timings_json: str | None,
    cache_hit: bool,
) -> int:
    now = _utc_now()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_turns (
                session_id, interaction_id, question, answer, sources_json, timings_json, cache_hit, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, interaction_id, question, answer, sources_json, timings_json, 1 if cache_hit else 0, now),
        )
        conn.execute(
            "UPDATE chat_sessions SET last_activity_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Failed to save chat turn")
        return int(cursor.lastrowid)


def load_chat_history(db_path: Path, session_id: int) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT t.id as turn_id, t.interaction_id, t.question, t.answer, t.sources_json,
                   t.timings_json, t.cache_hit, f.score as feedback_score, f.comment as feedback_comment
            FROM chat_turns t
            LEFT JOIN feedback f ON f.turn_id = t.id
            WHERE t.session_id = ?
            ORDER BY t.id ASC
            """,
            (session_id,),
        ).fetchall()

    history: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "interaction_id": row["interaction_id"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": [],
            "timings": {},
            "cache_hit": bool(row["cache_hit"]),
            "db_turn_id": int(row["turn_id"]),
        }
        sources_json = row["sources_json"]
        timings_json = row["timings_json"]
        if sources_json:
            try:
                import json

                entry["sources"] = json.loads(sources_json)
            except (ValueError, TypeError):
                entry["sources"] = []
        if timings_json:
            try:
                import json

                entry["timings"] = json.loads(timings_json)
            except (ValueError, TypeError):
                entry["timings"] = {}

        feedback_score = row["feedback_score"]
        feedback_comment = row["feedback_comment"]
        if feedback_score:
            entry["feedback"] = str(feedback_score)
        if feedback_comment:
            entry["feedback_comment"] = str(feedback_comment)

        history.append(entry)

    return history


def save_feedback(db_path: Path, turn_id: int, score: str, comment: str | None = None) -> None:
    now = _utc_now()
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM feedback WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE feedback
                SET score = ?, comment = ?, created_at = ?
                WHERE turn_id = ?
                """,
                (score, comment, now, turn_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO feedback (turn_id, score, comment, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (turn_id, score, comment, now),
            )
        conn.commit()
