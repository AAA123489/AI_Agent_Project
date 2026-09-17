"""测试 src/stream_gate.py —— 抑制工具调用前的英文旁白。

纯字符串处理，零依赖（不 import app_backend / chromadb / torch）。

真实缺陷（HEAD 原码就有，6 题基准里 4~5 题命中）：
    "I'll search the knowledge base for information about the 2026 Dragon Boat
     Festival holiday根据知识库中的通知，河南工学院..."
用户看到的回答首句是英文旁白。SSE 是逐 delta 直发的，等这一轮结束发现是 tool 轮时，
旁白早已显示在屏幕上——所以必须在**流开头**就扣住。
"""
from src.stream_gate import OpeningGate


def _feed_all(gate, text, chunk=1):
    """按 chunk 大小切片喂进去，返回实际下发的内容。"""
    out = []
    for i in range(0, len(text), chunk):
        out.append(gate.feed(text[i:i + chunk]))
    return "".join(out)


PREAMBLE = "I'll search the knowledge base for information about the 2026 Dragon Boat Festival holiday"


def test_只有英文旁白的工具轮一个字都不下发():
    gate = OpeningGate()
    emitted = _feed_all(gate, PREAMBLE)

    assert emitted == ""
    assert gate.finish(had_tool_use=True) == ""
    # 旁白一个字都不该漏出去
    assert "I'll" not in emitted


def test_旁白之后的正文照常流式下发():
    gate = OpeningGate()
    emitted = _feed_all(gate, PREAMBLE + "根据通知，2026年端午节放假安排如下。")

    assert emitted == "根据通知，2026年端午节放假安排如下。"
    assert gate.finish(had_tool_use=True) == ""


def test_中文回答几乎无延迟_首个增量就放行():
    """正文以汉字开头时，第一个 delta 就该出，不能被闸门压住——流式的价值在这。"""
    gate = OpeningGate()

    assert gate.feed("根据") == "根据"


def test_纯英文轮整段补发_不能吃掉英文提问的回答():
    """用户用英文提问、模型用英文回答：这一轮没有 tool_use，扣住的内容要补发。"""
    gate = OpeningGate()
    answer = "The 2026 Dragon Boat Festival holiday runs from June 19 to June 21."

    emitted = _feed_all(gate, answer)
    assert emitted == ""            # 全程扣住（没有汉字）
    assert gate.finish(had_tool_use=False) == answer   # 轮末整段补发


def test_纯英文轮且有tool_use_丢弃():
    gate = OpeningGate()

    assert _feed_all(gate, PREAMBLE) == ""
    assert gate.finish(had_tool_use=True) == ""


def test_开闸后finish不再补发():
    """已开闸说明扣住的都放完了，finish 必须是空的，否则会重复输出。"""
    gate = OpeningGate()
    _feed_all(gate, PREAMBLE + "答案在这里")

    assert gate.finish(had_tool_use=False) == ""
    assert gate.finish(had_tool_use=True) == ""


def test_逐字喂和整段喂结果一致():
    """真实流式是 token 级切片，切片边界不能影响结果。"""
    text = PREAMBLE + "根据通知，放假三天。"
    char_by_char = OpeningGate()
    whole = OpeningGate()

    a = _feed_all(char_by_char, text, chunk=1)
    b = _feed_all(whole, text, chunk=1000)

    assert a == b == "根据通知，放假三天。"


def test_英文前缀里夹着汉字标点也算旁白结束():
    """模型可能在旁白里带上中文标点（"…holiday。"），不能把标点误当正文。"""
    gate = OpeningGate()

    # 「。」不是汉字，闸门应继续扣住
    assert gate.feed(PREAMBLE + "。") == ""
    assert gate.feed("根据") == "根据"


def test_首增量就含汉字时立即放行且不丢字():
    gate = OpeningGate()

    assert gate.feed("答案：2026年4月16日") == "答案：2026年4月16日"


def test_空轮不产生输出():
    gate = OpeningGate()

    assert gate.finish(had_tool_use=True) == ""
    assert gate.finish(had_tool_use=False) == ""
