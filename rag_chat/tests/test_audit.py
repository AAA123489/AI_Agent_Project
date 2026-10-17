"""问答审计：采集、落库、查询、导出、失败不影响主流程。"""

import json

import pytest

from audit import (
    MAX_TOOL_OUTPUT_CHARS,
    QATrace,
    export_audit_rows,
    query_audit,
    record_qa_trace,
)


def make_trace(**overrides):
    params = {
        "user_id": 1,
        "username": "alice",
        "question": "河南工学院有哪些专业？",
        "request_id": "req-1",
        "conversation_id": "conv-1",
        "history": [{"role": "user", "content": "你好"}],
    }
    params.update(overrides)
    return QATrace(**params)


def test_正常问答完整落库(app_env):
    trace = make_trace()
    trace.on_thinking("先检索知识库")
    trace.on_tool_call("search_kb", {"query": "专业"})
    trace.on_tool_result(
        "search_kb",
        True,
        "检索结果原文",
        sources=[{"url": "https://example.edu/a", "similarity": 0.87}],
    )
    trace.on_text("学校开设了")
    trace.on_text("多个专业。")
    trace.on_done(
        thinking=["先检索知识库", "再组织答案"],
        sources=[{"url": "https://example.edu/a", "similarity": 0.87}],
    )

    record_qa_trace(trace)

    total, items = query_audit(user_id=1)
    assert total == 1
    item = items[0]
    assert item["username"] == "alice"
    assert item["question"] == "河南工学院有哪些专业？"
    assert item["answer"] == "学校开设了多个专业。"
    assert item["thinking"] == ["先检索知识库", "再组织答案"]
    assert item["status"] == "done"
    assert item["mode"] == "live"
    assert item["conversation_id"] == "conv-1"
    assert item["history"] == [{"role": "user", "content": "你好"}]
    assert item["sources"][0]["similarity"] == 0.87

    call = item["tool_calls"][0]
    assert call["tool"] == "search_kb"
    assert call["success"] is True
    assert call["output"] == "检索结果原文"
    assert json.loads(call["args"]) == {"query": "专业"}
    assert call["duration_ms"] >= 0


def test_落库幂等(app_env):
    trace = make_trace()
    trace.on_text("答案")
    trace.on_done([], [])

    record_qa_trace(trace)
    record_qa_trace(trace)

    assert query_audit()[0] == 1


def test_异常与缓存模式(app_env):
    failed = make_trace(request_id="req-error")
    failed.on_error("工具调用超时")
    record_qa_trace(failed)

    cached = make_trace(question="缓存问题", request_id="req-cache", mode="cache")
    cached.on_text("缓存答案")
    cached.on_done([], [])
    record_qa_trace(cached)

    total, items = query_audit(status="error")
    assert total == 1
    assert items[0]["error"] == "工具调用超时"

    total, items = query_audit(status="done")
    assert total == 1
    assert items[0]["mode"] == "cache"


def test_工具输出超长被截断(app_env):
    trace = make_trace()
    trace.on_tool_call("search_kb", {})
    trace.on_tool_result("search_kb", True, "x" * (MAX_TOOL_OUTPUT_CHARS + 500))
    trace.on_text("答案")
    trace.on_done([], [])
    record_qa_trace(trace)

    call = query_audit()[1][0]["tool_calls"][0]
    assert call["truncated"] is True
    assert len(call["output"]) <= MAX_TOOL_OUTPUT_CHARS + 100
    assert "[truncated" in call["output"]


def test_工具结果与调用按名称配对(app_env):
    trace = make_trace()
    trace.on_tool_call("first", {"n": 1})
    trace.on_tool_call("second", {"n": 2})
    trace.on_tool_result("second", True, "第二")
    trace.on_tool_result("first", False, "第一")
    trace.on_text("答案")
    trace.on_done([], [])
    record_qa_trace(trace)

    calls = query_audit()[1][0]["tool_calls"]
    assert [(c["tool"], c["args"]) for c in calls] == [
        ("second", json.dumps({"n": 2}, ensure_ascii=False)),
        ("first", json.dumps({"n": 1}, ensure_ascii=False)),
    ]
    assert calls[0]["success"] is True
    assert calls[1]["success"] is False


def test_落库失败不影响主流程(app_env, monkeypatch):
    import audit

    def boom(_row):
        raise RuntimeError("disk full")

    monkeypatch.setattr(audit, "insert_qa_audit", boom)

    trace = make_trace()
    trace.on_text("答案")
    trace.on_done([], [])

    # 不抛异常
    record_qa_trace(trace)
    assert query_audit()[0] == 0


def test_导出_jsonl与csv(app_env):
    trace = make_trace()
    trace.on_text("答案")
    trace.on_done([], [])
    record_qa_trace(trace)

    jsonl = "".join(export_audit_rows(fmt="jsonl", user_id=1))
    rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["question"] == "河南工学院有哪些专业？"

    csv_text = "".join(export_audit_rows(fmt="csv", user_id=1))
    assert csv_text.startswith("﻿" + "id,")
    assert "河南工学院有哪些专业？" in csv_text


def test_查询过滤(app_env):
    for user_id, username in [(1, "alice"), (2, "bob")]:
        trace = make_trace(user_id=user_id, username=username)
        trace.on_text("答案")
        trace.on_done([], [])
        record_qa_trace(trace)

    assert query_audit(user_id=2)[0] == 1
    assert query_audit(user_id=2)[1][0]["username"] == "bob"
    assert query_audit()[0] == 2


@pytest.mark.parametrize("limit", [0, -5, 999])
def test_查询limit被限制在合法范围(app_env, limit):
    trace = make_trace()
    trace.on_text("答案")
    trace.on_done([], [])
    record_qa_trace(trace)

    total, items = query_audit(limit=limit)
    assert total == 1
    assert len(items) == 1
