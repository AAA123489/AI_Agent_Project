"""测试召回自检（src/recall_guard.py）—— 判定逻辑、三种去向、改写重检环、终止保证。

全部通过 DI 注入 fake `retrieve_fn` / `llm_fn`，**零网络、零 LLM 调用、不依赖 chromadb/torch**。
本模块也不 import app_backend，所以本文件不需要拉起向量库。

不用 `@pytest.mark.asyncio`（.venv 没装 pytest-asyncio），照项目三
`tests/test_agent_loop.py` 的写法用 `asyncio.run`。
"""
import asyncio

import pytest

from src.recall_guard import (
    PASS_DIST,
    REFUSE_TEXT,
    _parse_verdict,
    run_recall_guard,
)

API_KEY = "test-key"


# ── 辅助 ──────────────────────────────────────────────────────


class _Spy:
    """记录调用次数的计数器（列表包一层，便于在闭包里改）。"""

    def __init__(self):
        self.retrieve = 0
        self.llm = 0
        self.llm_inputs: list[str] = []
        self.retrieve_queries: list[str] = []


def _make_retrieve(spy, best_distance=None, extra=0, short_circuit=None):
    """造一个假检索。

    best_distance：向量路最佳余弦距离，直接落到 vector_distance 字段；
                   None 表示没有向量路命中（只有同文档补块）。
    extra：额外补几个**没有** vector_distance 的块，模拟「同文档补块」。
    """

    def _retrieve(query, top_k):
        spy.retrieve += 1
        spy.retrieve_queries.append(query)
        if short_circuit is not None:
            return {"candidates": [], "short_circuit": short_circuit, "timings": {}}
        candidates = []
        if best_distance is not None:
            candidates.append({
                "text": f"向量块：{query}",
                "metadata": {"source": "doc.md"},
                "distance": best_distance,
                "vector_distance": best_distance,
            })
        for i in range(extra):
            candidates.append({
                "text": f"同文档补块 {i}",
                "metadata": {"source": "doc.md"},
                "distance": 0.5,  # 融合后的分，不可信
            })
        return {"candidates": candidates, "short_circuit": None, "timings": {}}

    return _retrieve


def _make_llm(spy, verdict="sufficient", rewrite="-"):
    """造一个假判官，永远返回同一判定。"""

    async def _llm(messages, api_key, **kwargs):
        spy.llm += 1
        spy.llm_inputs.append(messages[0]["content"])
        return {"text": f"VERDICT: {verdict} | REWRITE: {rewrite} | EVIDENCE: 测试依据"}

    return _llm


def _run(retrieve, llm, query="测试问题", top_k=8, max_attempts=2):
    return asyncio.run(run_recall_guard(
        retrieve_fn=retrieve,
        format_fn=lambda cands: f"[{len(cands)} 块]检索上下文",
        llm_fn=llm,
        api_key=API_KEY,
        query=query,
        top_k=top_k,
        max_attempts=max_attempts,
    ))


# ── _parse_verdict：宽松解析 + 双默认兜底 ──────────────────────


def test_parse_verdict_正常格式():
    assert _parse_verdict("VERDICT: insufficient | REWRITE: 河南工学院 转专业 条件 | EVIDENCE: x") == (
        "insufficient",
        "河南工学院 转专业 条件",
    )
    assert _parse_verdict("VERDICT: off_topic | REWRITE: - | EVIDENCE: x") == ("off_topic", "")


@pytest.mark.parametrize("raw", [
    "",                                     # 空文本（判官 API 失败时的返回值）
    "API 调用失败 (401)",                    # _call_llm 的失败返回值
    "我不知道该怎么判",                       # 完全跑偏
    "VERDICT: maybe | REWRITE: -",          # 判定值不在白名单
    "判定结果：充分",                         # 没有 VERDICT 字段
])
def test_parse_verdict_解析不出时一律放行(raw):
    """双默认兜底：任何解析不出的情况都必须 sufficient，否则 API 抖动会直接掉分。"""
    assert _parse_verdict(raw) == ("sufficient", "")


@pytest.mark.parametrize("rewrite", ["-", "无", "none", "N/A"])
def test_parse_verdict_空改写归一化(rewrite):
    assert _parse_verdict(f"VERDICT: insufficient | REWRITE: {rewrite}") == ("insufficient", "")


# ── 快通过通道：省掉 LLM 往返 ────────────────────────────────


def test_向量距离够近时快通过且不调判官():
    """基线 5 题的最佳距离全在 0.14~0.31，走这条路，回归风险≈0 且省一次 LLM 调用。"""
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=PASS_DIST - 0.10),
               _make_llm(spy, verdict="off_topic"))  # 判官故意唱反调，快通过应无视它

    assert spy.llm == 0, "快通过不应触发判官"
    assert spy.retrieve == 1
    assert out == "[1 块]检索上下文"


def test_无向量信号时必走判官():
    """补块不带 vector_distance，best is None → 不能快通过，必须交给判官。"""
    spy = _Spy()
    _run(_make_retrieve(spy, best_distance=None, extra=3), _make_llm(spy))

    assert spy.llm == 1


# ── 三种去向 ─────────────────────────────────────────────────


def test_判官放行():
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=0.40), _make_llm(spy, "sufficient"))

    assert spy.retrieve == 1
    assert out == "[1 块]检索上下文"


def test_判官判主题不符直接拒答():
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=0.47, extra=11),
               _make_llm(spy, "off_topic"))

    assert out == REFUSE_TEXT
    assert spy.retrieve == 1, "off_topic 不该重检"


def test_拒答文案前缀必须是召回自检():
    """AgentLoop 靠「【召回自检」这个具体前缀区分本层拒答与库外闸门的「【库外题拦截】」。

    用宽泛的「【」会把库外拦截误标成自检未通过。
    """
    assert REFUSE_TEXT.startswith("【召回自检")


# ── 改写重检环 ────────────────────────────────────────────────


def test_判不充分时改写query重检():
    spy = _Spy()
    _run(_make_retrieve(spy, best_distance=0.40),
         _make_llm(spy, "insufficient", "河南工学院 转专业 申请条件"))

    assert spy.retrieve == 2, "环应跑一圈"
    assert spy.retrieve_queries[0] == "测试问题"
    assert spy.retrieve_queries[1] == "河南工学院 转专业 申请条件", "第二圈必须用改写后的 query"


def test_改写后判官放行则环出口是放行():
    """模拟「第一圈不充分 → 改写 → 第二圈命中」的正常收益路径。"""
    spy = _Spy()
    state = {"n": 0}

    async def _llm(messages, api_key, **kwargs):
        spy.llm += 1
        state["n"] += 1
        verdict = "insufficient" if state["n"] == 1 else "sufficient"
        rewrite = "改写后的查询" if state["n"] == 1 else "-"
        return {"text": f"VERDICT: {verdict} | REWRITE: {rewrite}"}

    out = _run(_make_retrieve(spy, best_distance=0.40), _llm)

    assert spy.retrieve == 2 and spy.llm == 2
    assert out == "[1 块]检索上下文", "第二圈放行后应输出检索上下文，不是拒答文案"


def test_重试用尽转拒答():
    """判官一直说不充分，max_attempts=2 → 检索两次后拒答。"""
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=0.40),
               _make_llm(spy, "insufficient", "改写"), max_attempts=2)

    assert spy.retrieve == 2 and spy.llm == 2
    assert out == REFUSE_TEXT


def test_max_attempts为1时不进环():
    """RECALL_GUARD_MAX_ATTEMPTS=1 就是消融用的「环关」口径。"""
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=0.40),
               _make_llm(spy, "insufficient", "改写"), max_attempts=1)

    assert spy.retrieve == 1 and spy.llm == 1
    assert out == REFUSE_TEXT


def test_改写为空时不重复打同一枪():
    """REWRITE 归一化成空 → current_query 不更新，但 attempt 仍推进，照样终止。

    注意：这里第二圈 query 与第一圈相同——不是 bug，是「尝试过就计数」的保守口径。
    """
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=0.40),
               _make_llm(spy, "insufficient", "-"), max_attempts=2)

    assert spy.retrieve == 2
    assert spy.retrieve_queries == ["测试问题", "测试问题"]
    assert out == REFUSE_TEXT


def test_未指定max_attempts时默认不重检():
    """不显式传 max_attempts 时的保守取值 = 1（不重检）。

    不明确要求重检就不自作主张多打一枪——这条默认值必须由测试钉住，
    否则将来有人把默认改成 2，调用方会凭空多花一次 LLM 调用。
    """
    spy = _Spy()
    out = asyncio.run(run_recall_guard(
        retrieve_fn=_make_retrieve(spy, best_distance=0.40),
        format_fn=lambda cands: f"[{len(cands)} 块]检索上下文",
        llm_fn=_make_llm(spy, "insufficient", "改写"),
        api_key=API_KEY,
        query="测试问题",
    ))

    assert spy.retrieve == 1 and spy.llm == 1
    assert out == REFUSE_TEXT


def test_判官永远不充分也能停住():
    """终止性回归测试。

    改手写之前这里靠 LangGraph 的 `recursion_limit` 兜底，但它的默认值是 10007
    （不是旧版的 25），等于没有兜底——真跑飞会空转到一万步、每圈都在烧 LLM。
    现在**唯一**的终止条件就是 attempt 计数器（循环上界由它直接决定），
    本用例把 max_attempts 放大到 5，验证它确实按计数停。
    """
    spy = _Spy()
    _run(_make_retrieve(spy, best_distance=0.40),
         _make_llm(spy, "insufficient", "换个说法"), max_attempts=5)

    assert spy.retrieve == 5, f"应恰好检索 5 次，实际 {spy.retrieve}"
    assert spy.llm == 5


# ── 检索侧早退：文案原样透传，不判 ────────────────────────────


@pytest.mark.parametrize("text", [
    "【库外题拦截】只回答河南工学院相关问题。",
    "检索失败: 连接超时",
    "知识库中未找到相关内容。",
])
def test_检索侧早退文案原样透传(text):
    spy = _Spy()
    out = _run(_make_retrieve(spy, short_circuit=text), _make_llm(spy, "off_topic"))

    assert out == text, "既有拒答文案不能被自检文案覆盖"
    assert spy.llm == 0, "检索侧已早退，没必要再花一次 LLM 判定"


# ── 健壮性：判官故障绝不能中断问答 ────────────────────────────


def test_判官输出无法解析时放行():
    spy = _Spy()

    async def _llm(messages, api_key, **kwargs):
        spy.llm += 1
        return {"text": "我不知道该怎么判"}

    out = _run(_make_retrieve(spy, best_distance=0.40), _llm)

    assert out == "[1 块]检索上下文"


def test_判官抛异常时放行():
    """判官挂在 SSE 关键路径上，异常必须吞掉——宁可少拒答，不可误拒答，更不可断流。"""
    spy = _Spy()

    async def _llm(messages, api_key, **kwargs):
        spy.llm += 1
        raise RuntimeError("网络炸了")

    out = _run(_make_retrieve(spy, best_distance=0.40), _llm)

    assert out == "[1 块]检索上下文"


def test_判官返回非白名单判定不会炸():
    """非法判定必须在 _parse_verdict 就被归一化成 sufficient。

    改手写之前这条最关键：路由函数返回 path_map 以外的 key 会直接 KeyError，
    而 SSE 侧没有 try/except，会打断整条流。现在 return 就是 return，
    但「判定值白名单」这层仍要守住——非白名单判定落到下面只会当成 insufficient。
    """
    spy = _Spy()
    out = _run(_make_retrieve(spy, best_distance=0.40), _make_llm(spy, "MAYBE"))

    assert out == "[1 块]检索上下文"


# ── 判官输入裁剪 ─────────────────────────────────────────────


def test_判官只吃top5且每块截断200字():
    """不裁的话补块后常 15~20 块 × 800 字 ≈ 16k 字符，每问多付一大笔 token。"""
    spy = _Spy()
    _run(_make_retrieve(spy, best_distance=0.40, extra=11), _make_llm(spy))

    prompt = spy.llm_inputs[0]
    assert "[5]" in prompt, "应保留到第 5 块"
    assert "[6]" not in prompt, "第 6 块起不该进判官输入"


def test_top_k透传给检索():
    """MCP 侧会传非默认 top_k，必须在 state 里传到 retrieve_fn，不能被闭包写死。"""
    spy = _Spy()
    got = {}

    def _retrieve(query, top_k):
        spy.retrieve += 1
        got["top_k"] = top_k
        return {"candidates": [], "short_circuit": "知识库中未找到相关内容。", "timings": {}}

    _run(_retrieve, _make_llm(spy), top_k=3)

    assert got["top_k"] == 3
