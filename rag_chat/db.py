"""
db.py — SQLite 账号 / 每日额度 / 问答审计库
========================================
单文件数据库，默认 rag_chat/data/rag_app.db（APP_DB_PATH 可覆盖）。
按 user_id 做逻辑隔离；不按用户拆分物理文件，方便迁移、备份和查询。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "rag_app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL,
    username_norm   TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    last_login_at   TEXT,
    session_version INTEGER NOT NULL DEFAULT 1,
    disabled        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_daily (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day     TEXT    NOT NULL,
    count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS qa_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    username        TEXT    NOT NULL,
    request_id      TEXT,
    conversation_id TEXT,
    question        TEXT    NOT NULL,
    answer          TEXT,
    thinking_json   TEXT,
    tool_calls_json TEXT,
    sources_json    TEXT,
    history_json    TEXT,
    error           TEXT,
    mode            TEXT    NOT NULL DEFAULT 'live',
    status          TEXT    NOT NULL DEFAULT 'aborted',
    latency_ms      INTEGER,
    model           TEXT,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qa_audit_user_time ON qa_audit(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_audit_time ON qa_audit(created_at DESC);
"""


def get_db_path() -> Path:
    """数据库路径。APP_DB_PATH 为相对路径时相对于 rag_chat/ 解析。"""
    raw = (os.getenv("APP_DB_PATH") or "").strip()
    if not raw:
        return _DEFAULT_DB
    path = Path(raw)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def connect_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """建表，幂等。应用启动和测试 fixture 都会调用。"""
    conn = connect_db(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def local_day(now: datetime | None = None) -> str:
    """按服务器本地时区生成 YYYY-MM-DD，额度在本地零点重置。"""
    return (now or datetime.now()).strftime("%Y-%m-%d")


def try_consume_daily_quota(user_id: int, limit: int, day: str | None = None) -> bool:
    """
    原子扣减当日额度。返回 True 表示放行，False 表示已超限。
    limit <= 0 表示不限制。
    """
    if limit <= 0:
        return True

    day = day or local_day()
    conn = connect_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO usage_daily(user_id, day, count) VALUES (?, ?, 1)
            ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1
            WHERE usage_daily.count < ?
            """,
            (user_id, day, limit),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_daily_usage(user_id: int, day: str | None = None) -> int:
    day = day or local_day()
    conn = connect_db()
    try:
        row = conn.execute(
            "SELECT count FROM usage_daily WHERE user_id = ? AND day = ?",
            (user_id, day),
        ).fetchone()
        return int(row["count"]) if row else 0
    finally:
        conn.close()
