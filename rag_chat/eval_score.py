"""
eval_score.py — 6 题基准打分器（基准事实表 + 60 分制）
============================================
读取 eval_baseline.py 跑出的结果 json，按「基准事实表 + 60 分制评分规则」
调 DeepSeek 给每题打分，输出每题分数 + 总分。

事实表与评分规则恢复自 2026-08-08 备份（原 qwen_score_prompt_*.md，逐字取自 eval_score.pyc 常量表）。

用法:
    python eval_score.py                              # 默认读 eval_results_baseline.json
    python eval_score.py eval_results_baseline_topk8.json
    python eval_score.py --out eval_score_topk8.md    # 顺带把打分原文存档
"""

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys

if __name__ == "__main__":
    # Windows 控制台 GBK 兼容（只在作为脚本运行时改 stdout；放模块级会劫持
    # import 本模块者的 stdout，pytest 拆捕获流时崩）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_score")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import aiohttp  # noqa: E402

API_URL = os.getenv("API_URL", "https://api.deepseek.com/anthropic/v1/messages")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")

# ── 基准事实表：唯一评分依据，按源文档逐条提取 ──
FACT_TABLE = """
1. 2026年端午节放假安排（《关于2026年端午节放假安排的通知》，2026-06-16）：
   6月19日（星期五）至6月21日（星期日）放假，共3天。
   要求：总值班室24小时在岗；放假前各单位全面安全隐患排查；6月17日（星期三）17:00前各部门上报值班安排表。
2. 河南工学院第九届运动会（《关于举办河南工学院第九届运动会的通知》，2026-04-07）：
   2026年4月16日至17日在科苑田径场举办。
3. 中国国际大学生创新大赛（2026）河南工学院校赛（《关于举办中国国际大学生创新大赛（2026）河南工学院校赛的通知》，2026-06-12）：
   2026年6月举办。大赛主题"我敢闯、我会创"。主办：创新创业指导中心；承办：电气工程与自动化学院。
   赛程三阶段：1.学院初赛 6月10日—6月24日；2.学校复赛 6月25日—6月26日（暂定）；3.学校决赛 7月2日（暂定）。
   报名截止 = 学院初赛截止（6月24日）。
4. 2026年大学生暑期"三下乡"社会实践活动（《关于开展河南工学院2026年大学生暑期"三下乡"社会实践活动的通知》，2026-06-10）：
   活动时间 2026年6月—10月。主题"建功'十五五'·青春为中国式现代化挺膺担当"。参与人员：全体在校学生。
   分为个人项目和团队项目：个人项目 7月10日前在"学习通"提交申报；学院团队 7月10日前在"学习通"申报；
   校级团队 6月15日前在"学习通"申报，校团委评选立项。
   阶段：团队组建申报（即日起—6月15日）→ 行前培训（6月中下旬）→ 实施开展（6月中旬—8月下旬）→ 成果提交（9月，9月27日前学习通提交结项申报书+总结报告）→ 总结表彰（10月）。
5. 2026年春季学期学生心理健康教育（《关于做好2026年春季学期学生心理健康教育有关工作的通知》，2026-03-03）：
   ① 心理健康知识宣传普及；② 3月3日—3月15日开展全覆盖返校学生心理健康排查（心理测评+辅导员走访）；
   ③ 排查发现的心理异常/危机风险学生，须于3月20日前完成台账信息在钉钉"学生心理危机预警与干预管理系统"录入与更新；
   ④ 分级分类干预（建档管理、个性化方案、一对一帮扶、转介专业机构）；⑤ 规范复学学生心理健康管理。
6. 河北工学院的录取分数线：基准表中无此信息（库外题，应拒答）。
"""

# ── 评分提示词 = 表头 + 事实表 + 固定题目与评分规则 ──
_PROMPT_HEAD = """你是我的 RAG 问答系统评测助手。我会给你一份"基准事实表"和一套固定测试题。
后续我会发来系统对不同测试题的作答，你只需要严格按照基准表打分，不要用你自己的知识判断对错。

【基准事实表】以下来自河南工学院官网全量库文档的真实内容，是唯一评分依据：
"""
_PROMPT_RULES = """

【固定测试题】共6题：
Q1 2026年端午节放假是怎么安排的？
Q2 河南工学院第九届运动会是什么时候举办？
Q3 中国国际大学生创新大赛（2026）河南工学院校赛的报名截止时间是什么时候？
Q4 学校最近有安排暑期社会实践吗？具体怎么做？
Q5 2026年春季学期学校在学生心理健康方面有哪些安排？
Q6 河北工学院的录取分数线是多少？

【评分规则】每题满分10分：
- 答案涵盖基准表该题全部"命中事实点"（日期/数字/名称/条数等关键信息）且无误 → 满分
- 漏掉基准表中应有的关键信息 → 每题扣 1-3 分
- 关键信息答错或张冠李戴（如把别的项目的数字安到本题头上）→ 该点不得分并扣 2 分
- 答案出现基准表中不存在、且与题目无关的具体数字 → 判为"幻觉"，额外扣 3 分并在批注中说明
- Q4 为同义改写题：提问用词与原文不同（如"暑期社会实践"指"三下乡"），只要命中基准事实点即算对
- Q6 为库外拒答题：正确说"知识库中没有相关内容"→ 10 分；编造了答案（含基准表中不存在的数字）→ 0 分并判为严重幻觉
- 只依据基准表判断；基准表未涉及的内容一律视为"无法验证"，不算对也不算错（该条仅适用于 Q1-Q5）

【输出格式】严格按此格式，最后给出总分：
Q1：X分
- 命中：……
- 问题：……
（Q2~Q6 同上）
总分：XX/60"""

SCORING_PROMPT = _PROMPT_HEAD + FACT_TABLE + _PROMPT_RULES

# ── 输入输出路径（题目满分 10 分 × 6 题 = 60 分）──
_TOTAL_MAX = 60


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="6 题基准打分器")
    p.add_argument(
        "json_path",
        nargs="?",
        default="eval_results_baseline.json",
        help="eval_baseline.py 跑出的评测结果 json",
    )
    p.add_argument("--out", default=None, help="把打分原文存档到 md 文件")
    return p.parse_args()


_ARGS = _parse_args()


async def _score_once(answers_text: str) -> str:
    """调一次 DeepSeek 打分，返回原始文本。"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "max_tokens": 2000,
                "temperature": 0,
                "system": SCORING_PROMPT,
                "messages": [{"role": "user", "content": "请对以下系统作答打分：\n\n" + answers_text}],
                "thinking": {"type": "disabled"},
            },
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                return f"API 错误 ({resp.status}): {await resp.text()}"
            data = await resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []))


def _parse_scores(raw: str) -> dict:
    """从打分输出解析每题分数 + 总分。解析失败时返回 raw 供人工查看。"""
    scores = {}
    for i in range(1, 7):
        m = re.search(f"Q{i}\\s*[:：]\\s*(\\d+)\\s*分", raw)
        if m:
            scores[f"Q{i}"] = int(m.group(1))
    m = re.search(r"总分\s*[:：]\s*(\d+)\s*/\s*60", raw)
    if m:
        scores["total"] = int(m.group(1))
    return scores or {"raw": raw}


async def main() -> None:
    with open(_ARGS.json_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    print(f"评测结果: {_ARGS.json_path}")
    print(f"配置: {data.get('config', '(未记录)')}")

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"Q{i} 作答：\n{r['answer']}\n")
    answers_text = "".join(parts)

    print(f"⏳ 正在让 DeepSeek 打分（{MODEL_NAME}）...")
    raw = await _score_once(answers_text)

    print("\n" + "=" * 60)
    print(raw)
    print("=" * 60)

    scores = _parse_scores(raw)
    print("\n📊 解析分数:")
    print(json.dumps(scores, ensure_ascii=False, indent=2))

    total = scores.get("total")
    if total is not None:
        print(f"\n总分: {total}/{_TOTAL_MAX}")
    else:
        print("\n⚠️ 未能解析出总分，请人工查看上面的打分原文")

    if _ARGS.out:
        with open(_ARGS.out, "w", encoding="utf-8") as f:
            f.write(f"# 6 题基准评分（{MODEL_NAME}）\n\n")
            f.write(f"- 评测结果: {_ARGS.json_path}\n")
            f.write(f"- 配置: {data.get('config', '(未记录)')}\n")
            f.write(f"- 时间: {data.get('timestamp', '(未记录)')}\n\n")
            f.write(raw)
            f.write("\n\n## 解析分数\n\n```json\n")
            f.write(json.dumps(scores, ensure_ascii=False, indent=2))
            f.write("\n```\n")
        print(f"\n📄 打分存档: {_ARGS.out}")


if __name__ == "__main__":
    asyncio.run(main())
