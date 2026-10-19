"""
test_eval_verify.py — 规则核对器的边界
=======================================
`eval_verify.py` 的价值全在「判得准」：漏判会把幻觉放过去，误判会让 CI 永远红。
这里用合成结果钉住四类边界：

1. 来源正文/元数据里的日期、数字、链接 → 通过；
2. 答案自己编的日期/数字/链接 → 失败；
3. 来源只写「6月16日」而答案写「2026-06-16」→ 通过（按 MM-DD 回退匹配）；
4. 拒答不需要来源；运行日期（system prompt 注入的「今天」）不算幻觉。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_verify import (
    check_result, extract_dates, extract_quantities, looks_like_refusal,
)


def _result(answer, question="", sources=None):
    return {"question": question, "answer": answer, "sources": sources or []}


def _source(text="", date="", url="https://www.hait.edu.cn/info/1/2.htm"):
    return {"text": text, "date": date, "url": url}


def test_带单位数字提取():
    assert extract_quantities("共3天，覆盖42%的学生，补助1000元") == {"3天", "42%", "1000元"}
    assert extract_quantities("1,200人") == {"1200人"}


def test_日期提取统一格式():
    assert extract_dates("2026年6月16日") == {"2026-06-16"}
    assert extract_dates("2026-06-16") == {"2026-06-16"}
    assert extract_dates("6月16日") == {"06-16"}


def test_有来源的事实全部落地则通过():
    result = _result(
        "通知于2026-06-16发布，放假3天，覆盖42%的学院。详见 https://www.hait.edu.cn/info/1/2.htm",
        sources=[_source("发布日期 2026-06-16，放假 3 天，覆盖 42%", url="https://www.hait.edu.cn/info/1/2.htm")],
    )

    check = check_result(1, result)

    assert check.ok, (check.ungrounded_dates, check.ungrounded_quantities, check.bad_urls)


def test_编造的日期和数字被抓出来():
    result = _result("通知于2026-07-01发布，放假 9 天，覆盖 88% 的学院。",
                     sources=[_source("发布日期 2026-06-16，放假 3 天")])

    check = check_result(1, result)

    assert check.ungrounded_dates == ["2026-07-01"]
    assert set(check.ungrounded_quantities) == {"9天", "88%"}


def test_来源只写月日时按MM_DD回退匹配():
    result = _result("放假时间为2026-06-19。",
                     sources=[_source("6月19日（星期五）至6月21日放假")])

    assert check_result(1, result).ok


def test_运行日期不算幻觉():
    """「当前已是2026-09-16」来自 system prompt，不是知识库事实。"""
    result = _result("当前已是2026-09-16，该活动已结束。",
                     sources=[_source("活动时间 6月—10月")])

    check = check_result(1, result, run_date="2026-09-16")

    assert check.ok
    assert check.exempt_dates == ["2026-09-16"]


def test_问题里给的数字不算模型主张():
    result = _result("按您说的 3 天计算，共 3 天。", question="放假 3 天吗",
                     sources=[_source("放假安排另行通知")])

    check = check_result(1, result)

    assert check.ok, check.ungrounded_quantities


def test_拒答没有来源也通过():
    answer = "知识库中没有河北工学院的录取分数线信息，建议您前往该校官网查询。"
    assert looks_like_refusal(answer)

    check = check_result(1, _result(answer, question="河北工学院分数线"))

    assert check.ok and check.refusal


def test_声称有答案却没有来源要失败():
    check = check_result(1, _result("2026年端午节放假3天。"))

    assert not check.ok and check.no_source


def test_答案链接必须是检索来源():
    result = _result("详见 https://evil.example.com/page。",
                     sources=[_source("正文", url="https://www.hait.edu.cn/info/1/2.htm")])

    assert check_result(1, result).bad_urls == ["https://evil.example.com/page"]
