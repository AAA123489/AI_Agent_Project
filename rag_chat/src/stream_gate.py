"""抑制「调用工具前的播报」流到前端 —— OpeningGate。

## 问题

模型在发起工具调用前，常先吐一句旁白：

    I'll search the knowledge base for information about the 2026 Dragon Boat
    Festival holiday.

然后才发出 tool_use 块。而 SSE 是**逐 delta 直发**的（`app_backend.run_stream`
里 `yield TextEvent(content=chunk)`），等这一轮结束、看到 tool_use 时，英文早就
显示在用户屏幕上了。

这一轮的文本从头到尾都不属于回答：`run_stream` 往 `messages` 里只 append 了
tool_use 块，**文本块被丢弃**——它本来就没被当成答案，却被实时播了出去。
实测 6 题基准里 4~5 题的首句是这个英文旁白（HEAD 原码也一样，不是改造引入的）。

## 规则

每轮开头先扣住不发光，直到出现**第一个汉字**：

- 出现汉字 → 扣住的英文前缀**直接丢弃**，从汉字起原样放行。回答正文基本以汉字开头，
  所以正常流式几乎无延迟（第一个 token 就能出）。
- 整轮结束都没出现汉字 → 交给调用方决定：这轮**有 tool_use** 说明扣住的就是旁白，
  丢弃；**没有 tool_use** 说明是一段正常英文回答，整段补发（别把英文提问的回答吃掉）。

为什么用「汉字」当放行信号而不是「遇到 tool_use 才丢」：tool_use 块在流里**排在文本块
之后**才到达，等它到就已经太晚了；而本轮是纯文本轮还是工具轮，要等整轮流完才知道。
「汉字」是唯一在流**开头**就能拿到、且能区分旁白与正文的信号——SYSTEM_PROMPT 规则 9
要求用中文回复，正文必然含汉字。
"""

from __future__ import annotations

import re

# 汉字（CJK 统一表意文字 U+4E00–U+9FFF）。只用它当"正文开始了"的信号，不追求覆盖全部中日韩字符。
_CJK = re.compile(r"[一-鿿]")


class OpeningGate:
    """每轮一个实例。`feed()` 逐 delta 喂，`finish()` 在轮次结束时收尾。"""

    def __init__(self) -> None:
        self._held: list[str] = []
        self._open = False

    def feed(self, delta: str) -> str:
        """喂一个流式增量，返回**应当发往前端**的文本（可能为空 = 继续扣住）。"""
        if self._open:
            return delta
        self._held.append(delta)
        buf = "".join(self._held)
        m = _CJK.search(buf)
        if m is None:
            return ""  # 还全是非汉字：扣着，不下发
        self._open = True
        self._held.clear()
        # 丢弃 buf[:m.start()] 的旁白，只放行汉字及其后
        return buf[m.start():]

    def finish(self, *, had_tool_use: bool) -> str:
        """轮次结束。返回仍扣着、需要补发的文本。

        - 已开闸（本轮出过汉字）：没有残留，返回 ""
        - 未开闸且本轮有 tool_use：扣住的是英文旁白，丢弃
        - 未开闸且本轮无 tool_use：整轮没有汉字，是一段正常英文/数字回答，整段补发
        """
        if self._open:
            self._held.clear()
            return ""
        buf = "".join(self._held)
        self._held.clear()
        if had_tool_use:
            return ""
        self._open = True
        return buf
