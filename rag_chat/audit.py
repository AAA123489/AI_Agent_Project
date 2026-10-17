"""
audit.py — 按用户的问答审计
==========================
QATrace 在 run_agent_sse 里边流式边收集：
问题、答案、思考步骤、工具调用（参数/结果/耗时）、来源、错误、耗时。

设计要点：
- 一条问答一行，存在 users 同一个 SQLite 里，按 user_id 逻辑隔离
- 工具结果最多保留 20 KB，避免检索结果把库撑大
- 落库失败只记日志，绝不影响对话
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Iterable, Iterator

from db import connect_db

logger = logging.getLogger("audit")

MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_TOOL_ARGS_CHARS = 4_000
MAX_THINKING_STEPS = 200
MAX_HISTORY_CHARS = 16_000
MAX_ANSWER_CHARS = 64_000
MAX_FIELD_CHARS = 512_000


def _dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _cap(value, limit: int) -> str:
    text = _dumps(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _loads(value):
    if not value:
        return []
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


class QATrace:
    """一次问答的收集器。线程/协程内单实例使用。"""

    def __init__(
        self,
        *,
        user_id: int,
        username: str,
        question: str,
        request_id: str = "",
        conversation_id: str | None = None,
        mode: str = "live",
        history: list | None = None,
    ):
        self.user_id = int(user_id)
        self.username = username
        self.question = question
        self.request_id = request_id
        self.conversation_id = conversation_id
        self.mode = mode
        self.model = os.getenv("MODEL_NAME", "")
        self.history = history or []

        self.answer_parts: list[str] = []
        self.thinking: list[str] = []
        self.sources: list = []
        self.tool_calls: list[dict] = []
        self._pending: list[dict] = []
        self.error: str | None = None
        self.status = "aborted"

        self.started = time.perf_counter()
        self.persisted = False

    # ── 事件 ──

    def on_thinking(self, step: str) -> None:
        if step and len(self.thinking) < MAX_THINKING_STEPS:
            self.thinking.append(str(step))

    def on_tool_call(self, tool: str, args: dict) -> None:
        self._pending.append(
            {
                "tool": tool,
                "args": _cap(args, MAX_TOOL_ARGS_CHARS),
                "started": time.perf_counter(),
            }
        )

    def on_tool_result(self, tool: str, success: bool, output: str, sources=None) -> None:
        pending = None
        for index, item in enumerate(self._pending):
            if item["tool"] == tool:
                pending = self._pending.pop(index)
                break
        if pending is None:
            pending = {"tool": tool, "args": "{}", "started": time.perf_counter()}

        raw_output = output or ""
        truncated = len(raw_output) > MAX_TOOL_OUTPUT_CHARS
        stored_output = raw_output
        if truncated:
            stored_output = (
                raw_output[:MAX_TOOL_OUTPUT_CHARS]
                + f"\n[truncated {len(raw_output) - MAX_TOOL_OUTPUT_CHARS} chars]"
            )

        self.tool_calls.append(
            {
                "seq": len(self.tool_calls) + 1,
                "tool": tool,
                "args": pending.get("args", "{}"),
                "success": bool(success),
                "output": stored_output,
                "duration_ms": int((time.perf_counter() - pending["started"]) * 1000),
                "truncated": truncated,
            }
        )
        if sources:
            self.sources = list(sources)

    def on_text(self, content: str) -> None:
        if content:
            self.answer_parts.append(content)

    def on_done(self, thinking: list, sources: list) -> None:
        if thinking:
            self.thinking = [str(step) for step in thinking][:MAX_THINKING_STEPS]
        if sources is not None:
            self.sources = list(sources)
        self.status = "done"

    def on_error(self, message: str) -> None:
        self.error = str(message)[:2000]
        self.status = "error"

    # ── 行数据 ──

    def to_row(self) -> dict:
        answer = "".join(self.answer_parts)[:MAX_ANSWER_CHARS]
        row = {
            "user_id": self.user_id,
            "username": self.username,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "question": self.question[:8000],
            "answer": answer,
            "thinking_json": _cap(self.thinking, MAX_FIELD_CHARS),
            "tool_calls_json": _cap(self.tool_calls, MAX_FIELD_CHARS),
            "sources_json": _cap(self.sources, MAX_FIELD_CHARS),
            "history_json": _cap(self.history, MAX_HISTORY_CHARS),
            "error": self.error,
            "mode": self.mode,
            "status": self.status,
            "latency_ms": int((time.perf_counter() - self.started) * 1000),
            "model": self.model,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return row


def insert_qa_audit(row: dict) -> None:
    conn = connect_db()
    try:
        conn.execute(
            """
            INSERT INTO qa_audit(
                user_id, username, request_id, conversation_id,
                question, answer, thinking_json, tool_calls_json, sources_json,
                history_json, error, mode, status, latency_ms, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["user_id"],
                row["username"],
                row["request_id"],
                row["conversation_id"],
                row["question"],
                row["answer"],
                row["thinking_json"],
                row["tool_calls_json"],
                row["sources_json"],
                row["history_json"],
                row["error"],
                row["mode"],
                row["status"],
                row["latency_ms"],
                row["model"],
                row["created_at"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_qa_trace(trace: QATrace | None) -> None:
    """幂等落库；任何异常都吞掉，避免影响聊天主流程。"""
    if trace is None or trace.persisted:
        return
    trace.persisted = True
    try:
        insert_qa_audit(trace.to_row())
    except Exception:
        logger.exception("写入问答审计失败（不影响对话）")


# ── 保留期限 ──

def audit_retention_days() -> int:
    """审计保留天数。0 = 永久保留；非法值回落到 90。"""
    try:
        return max(0, int(os.getenv("AUDIT_RETENTION_DAYS", "90")))
    except ValueError:
        return 90


def purge_old_audit(days: int, now: datetime | None = None) -> int:
    """删除早于 days 天的审计行，返回删除条数。days <= 0 表示不清理。"""
    if days <= 0:
        return 0

    cutoff = (now or datetime.now()) - timedelta(days=days)
    cutoff_str = cutoff.isoformat(timespec="seconds")

    conn = connect_db()
    try:
        cursor = conn.execute(
            "DELETE FROM qa_audit WHERE created_at < ?", (cutoff_str,)
        )
        conn.commit()
        removed = int(cursor.rowcount or 0)
        if removed:
            # 更新查询计划统计；真正的空间回收需要 VACUUM，日常不做
            conn.execute("PRAGMA optimize")
        return removed
    finally:
        conn.close()


# ── 查询 / 导出 ──

def _where(user_id=None, date_from=None, date_to=None, status=None):
    clauses: list[str] = []
    params: list = []
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(int(user_id))
    if date_from:
        clauses.append("created_at >= ?")
        params.append(str(date_from))
    if date_to:
        end = str(date_to)
        if len(end) == 10:
            end += "T23:59:59"
        clauses.append("created_at <= ?")
        params.append(end)
    if status:
        clauses.append("status = ?")
        params.append(str(status))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def iter_audit_rows(
    user_id=None,
    date_from=None,
    date_to=None,
    status=None,
    batch_size: int = 200,
) -> Iterator[dict]:
    clause, params = _where(user_id, date_from, date_to, status)
    conn = connect_db()
    try:
        cursor = conn.execute(
            "SELECT * FROM qa_audit" + clause + " ORDER BY id DESC", params
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                item = dict(row)
                item["thinking"] = _loads(item.pop("thinking_json"))
                item["tool_calls"] = _loads(item.pop("tool_calls_json"))
                item["sources"] = _loads(item.pop("sources_json"))
                item["history"] = _loads(item.pop("history_json"))
                yield item
    finally:
        conn.close()


def query_audit(
    user_id=None,
    limit: int = 50,
    offset: int = 0,
    date_from=None,
    date_to=None,
    status=None,
) -> tuple[int, list[dict]]:
    limit = min(500, max(1, int(limit)))
    offset = max(0, int(offset))
    clause, params = _where(user_id, date_from, date_to, status)

    conn = connect_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM qa_audit" + clause, params).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM qa_audit" + clause + " ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    finally:
        conn.close()

    items = []
    for row in rows:
        item = dict(row)
        item["thinking"] = _loads(item.pop("thinking_json"))
        item["tool_calls"] = _loads(item.pop("tool_calls_json"))
        item["sources"] = _loads(item.pop("sources_json"))
        item["history"] = _loads(item.pop("history_json"))
        items.append(item)
    return int(total), items


def export_audit_rows(fmt: str = "jsonl", **filters) -> Iterable[str]:
    """按行产出导出内容：jsonl 或 csv。"""
    fmt = (fmt or "jsonl").lower()

    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "id", "user_id", "username", "request_id", "conversation_id",
                "question", "answer", "thinking", "tool_calls", "sources",
                "history", "error", "mode", "status", "latency_ms", "model", "created_at",
            ]
        )
        # BOM：让 Excel 正确识别 UTF-8 中文
        yield "﻿" + buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

        for item in iter_audit_rows(**filters):
            writer.writerow(
                [
                    item.get("id"),
                    item.get("user_id"),
                    item.get("username"),
                    item.get("request_id"),
                    item.get("conversation_id"),
                    item.get("question"),
                    item.get("answer"),
                    _dumps(item.get("thinking")),
                    _dumps(item.get("tool_calls")),
                    _dumps(item.get("sources")),
                    _dumps(item.get("history")),
                    item.get("error"),
                    item.get("mode"),
                    item.get("status"),
                    item.get("latency_ms"),
                    item.get("model"),
                    item.get("created_at"),
                ]
            )
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
        return

    for item in iter_audit_rows(**filters):
        yield json.dumps(item, ensure_ascii=False, default=str) + "\n"
