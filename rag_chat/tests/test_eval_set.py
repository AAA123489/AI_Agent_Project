"""
test_eval_set.py — 固定评测题库的结构校验
=========================================
题库 `eval_sets/qa_50.json` 是论文检索消融的唯一数据来源，一旦被手改坏
（少字段、正例没标 URL、混进栏目名标题），`eval_retrieval.py` 会静默给出
错误的命中率。这里用零网络的静态断言把住结构。

不做的事：不校验题目语义是否合理（那是人工审题库的活），也不调 LLM。
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper.html_parser import _looks_like_column_heading

_SET_PATH = _PROJECT_ROOT / "eval_sets" / "qa_50.json"


def _load() -> dict:
    return json.loads(_SET_PATH.read_text(encoding="utf-8"))


def test_题库结构与声明规模一致():
    data = _load()

    positives = [q for q in data["questions"] if q["type"] == "positive"]
    negatives = [q for q in data["questions"] if q["type"] == "negative"]

    assert data["seed"] == 20261019
    assert len(data["questions"]) == 50
    assert len(positives) == 40
    assert len(negatives) == 10
    # 负例口径：6 硬负例 + 4 无关题
    assert sum(1 for q in negatives if q["kind"] == "hard") == 6
    assert sum(1 for q in negatives if q["kind"] == "out_of_kb") == 4


def test_正例字段完整且可定位():
    for q in _load()["questions"]:
        if q["type"] != "positive":
            continue
        assert q["question"].strip()
        assert len(q["answer_span"].strip()) >= 5
        assert q["expected_url"].startswith("http")
        assert q["category"]


def test_正例标题不是栏目名():
    """栏目名当标题会让同类多篇文章共用 source，题库就无法定位到具体文章。"""
    bad = [
        q["source"] for q in _load()["questions"]
        if q["type"] == "positive" and _looks_like_column_heading(q["source"])
    ]

    assert not bad, f"题库里有栏目名标题: {bad}"
