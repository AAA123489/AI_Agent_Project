"""
app_backend.py — Agent 后端封装
===============================
封装 VectorStore 检索、LLM 调用（Anthropic Tool Use 格式）、Agent 循环。
不修改任何现有代码，只做导入和适配。

为 Gradio 前端提供 run_agent() 异步生成器。
"""

import ast
import asyncio
import concurrent.futures
import json
import logging
import operator as op
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import aiohttp
from dotenv import load_dotenv

# ── 路径：确保能导入同目录的 src 模块 ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vector_store import VectorStore
from src.hybrid_retriever import rrf_fuse, get_bm25_retriever, get_reranker
from src.stream_gate import OpeningGate

load_dotenv()

logger = logging.getLogger("gradio_agent")

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════

API_KEY = os.getenv("API_KEY", "")
API_URL = os.getenv("API_URL", "https://api.deepseek.com/anthropic/v1/messages")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-pro")
# 生成随机度（0~2）：0 = 完全保守，2 = 最大发散。RAG 问答默认 0.3 偏严谨
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
# 回答长度上限（tokens）：上限而非目标，只有超出才会截断。
# 消融实验参数：改 .env 的 MAX_TOKENS 即可，无需动代码（当前默认 1000）
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
# 向量检索返回片段数（top_k）。消融实验参数：改 .env 的 TOP_K 即可。
# 消融最优（2026-08-07，6题60分制）：top_k=8
TOP_K = int(os.getenv("TOP_K", "8"))
# 检索模式：hybrid（向量+BM25 RRF 融合，默认）| vector（纯向量，消融对照）
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
# 重排开关：on 时对融合后候选用 CrossEncoder 精排再截断 | off（默认，省加载模型）
RETRIEVAL_RERANK = os.getenv("RETRIEVAL_RERANK", "off")
# 重排候选数：重排后保留多少条进 LLM 上下文。
# 消融发现 N=8 对宽泛题（如 Q5「有哪些安排」）会截掉埋在补块里的关键句，
# N=16 时 Q3/Q5 关键句命中恢复 3/3、4/4（重排只排序不牺牲召回）
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "16"))
# 召回自检开关：on 时检索后走 LangGraph 自检图（判定不充分→改写 query 重检，
# 主题不符→拒答），补上"本校但库里没有"这类题的低置信度拒答 | off（默认，行为同改造前）
RECALL_GUARD = os.getenv("RECALL_GUARD", "off")
# 自检图允许的检索轮数：1 = 不重检（首次就判定），2 = 允许一次改写重检（环跑一圈）。
# 默认 1（环关）——这是消融的结论（`eval_guard_probe.py`，硬负例 6 题 + 无关题 4 题 + 正例 5 题）：
# 在干净负例上环开环关打平（都 6/6 拒答、4/4 拒无关题），环**没带来任何收益**；
# 但在灰区题上环有害——「招生办咨询电话」环关正确拒答，环开翻成放行（两次独立运行复现）。
# 机理：判官第二轮**不记得自己已经判过不充分**，改写重检后对着同样无关的新片段重新判，
# 翻成了 sufficient——环把一次正确的拒答变成了放行。无收益 + 有反效果 → 默认不开。
# 环的代码保留且可开关（消融要能复跑）。样本只有十几题，结论强度有限；
# 将来若要重开环，先让判官带上「上一轮已判不充分」的上下文，再重跑消融。
RECALL_GUARD_MAX_ATTEMPTS = int(os.getenv("RECALL_GUARD_MAX_ATTEMPTS", "1"))
# 知识库主体院校：用于识别「问别的学校」的库外题，避免张冠李戴幻觉（如 Q6 河北工学院）
KB_SUBJECT_SCHOOL = os.getenv("KB_SUBJECT_SCHOOL", "河南工学院")
# Agent 循环上限：工具结果已完整返回（800 字覆盖整个块），单题 1~2 轮即可回答。
# 设 6 兜底：正常单题 2~3 轮完成，极端复杂题也不会干等到 10 轮。
MAX_ROUNDS = 6
# 历史消息截断（防上下文超长）：只保留最近 MAX_HISTORY_TURNS 条，
# 每条 content 截断到 MAX_HISTORY_CHARS 字符（DeepSeek 上下文窗口有限）。
MAX_HISTORY_TURNS = 10
MAX_HISTORY_CHARS = 2000
# LLM 调用重试次数（含首次）：网络错误/HTTP 5xx/请求阶段超时重试 1 次，降低公网抖动导致的失败
LLM_MAX_RETRIES = 2


def get_active_params() -> str:
    """返回当前生效的运行时参数摘要（用于对话日志标注消融实验组）。

    分块参数（CHUNK_SIZE/CHUNK_OVERLAP）在重建阶段，运行时读不到，
    需靠 .env 的 EXPERIMENT_TAG 手动标注。
    """
    return (
        f"model={MODEL_NAME} | temperature={TEMPERATURE} | max_tokens={MAX_TOKENS} | "
        f"top_k={TOP_K} | retrieval_mode={RETRIEVAL_MODE} | rerank={RETRIEVAL_RERANK} | "
        f"recall_guard={RECALL_GUARD}"
        + (f"(attempts={RECALL_GUARD_MAX_ATTEMPTS})" if RECALL_GUARD == "on" else "")
    )

# ═══════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个 AI Agent 智能助手，能自主判断用户意图并调用相应工具。\n"
    "\n"
    "## 可用工具\n"
    "- **search_knowledge_base**: 在知识库中语义检索文档内容，获取专业信息\n"
    "- **get_knowledge_base_stats**: 查看知识库统计信息（文档数、分块数、分类）\n"
    "- **get_current_time**: 获取当前日期和时间\n"
    "- **get_weather**: 查询指定城市的实时天气（温度、湿度、风速、天气状况）\n"
    "- **calculate**: 安全计算数学表达式（支持 + - * / ** 和括号）\n"
    "\n"
    "## 工具使用规则\n"
    "1. 用户问专业知识、文档内容 → 先调用 search_knowledge_base 检索\n"
    "2. 用户问知识库有哪些文档、共多少文章、知识库规模 → 调用 get_knowledge_base_stats\n"
    "3. 用户问时间 → 调用 get_current_time\n"
    "4. 用户问天气 → 调用 get_weather（支持中文城市名，如「正阳县」「北京」「郑州」）\n"
    "5. 用户需要数学计算 → 调用 calculate\n"
    "6. 普通对话或已有足够信息 → 直接回答，不要无意义地反复调用工具\n"
    "7. **重要**：知识库检索不到相关内容时，不要用通用知识作答或编造学校内部信息，"
    "直接如实告知用户「知识库中暂无相关信息」\n"
    "8. 引用知识库内容时标注来源文件名\n"
    "9. 用中文回复\n"
    "10. 用户问「有哪些安排/如何开展/工作怎么做」这类概括性问题时，"
    "以覆盖多项工作的综合性通知（标题常含「有关工作的通知」，内容涵盖宣传、排查、干预、复学、活动等多方面）为作答主干，"
    "完整列出其中的各项工作及具体日期要求；不要只依据单次活动的通知作答。"
    "若检索结果同时含综合通知与单次活动通知，综合通知优先，单次活动通知作为补充。\n"
    "11. **重要**：判定「知识库没有该信息」之前，必须逐条核对检索返回的**全部片段**（[1]…[N]），"
    "一个片段没有 ≠ 知识库没有。只要任一片段含答案，就必须据此作答，"
    "并优先采用与问题直接对应的片段（如同文档的评分表、分项数值等）。"
    "片段里已含答案却回答「未找到/未给出/无明确分值」，属于严重错误。"
    "检索片段中的**具体数值、人数、日期、篇目名**等，能直接引用原文的必须逐字引用，"
    "不得用「未明确给出」「没有提到具体数字」等概括性说法替代，"
    "尤其当问题问的是数值/人数/分项时，即使答案片段排在较后（如 [6]…[N]）也必须找到并引用。"
    "只有逐条核对后确认所有片段均不含答案，才按规则 7 诚实说明。\n"
    "12. **重要**：涉及知识库文档中的具体事实（日期、天数、人数、第几个、百分比、分值等），"
    "必须先调用 search_knowledge_base 检索原文，答案必须以检索片段为准，"
    "严禁用 calculate 自行推算（如「3月20日至25日共几天」「2025年是第几个」），"
    "也不能脱离原文自行推导（原文可能明确写「共5天」「第十个」）。"
    "calculate 仅用于用户给出明确的数学表达式（如「3*5」「2**10」）时。\n"
    "13. **重要**：用户问「有哪些安排/如何开展/有哪些内容/具体要求/怎么做」这类需要完整列举的问题时，"
    "回答必须把检索片段中出现的**每一条**具体安排及其对应时间（日期/时间段/截止日）逐一列出，"
    "**不得只挑几条作答而漏掉其余**。先扫一遍全部片段，把每条「项目+时间」列成清单再组织答案；"
    "回答完后自查一遍：片段中出现过的每个日期（如 X月X日前、X月X日—X月X日）是否都已出现在答案里，"
    "漏了就是回答不完整。宁可多列，不可漏列。\n"
    "14. 用户问「截止时间/报名截止」时，若检索片段给出了该事项的阶段时间范围（如「学院初赛 6月10日—6月24日」），"
    "就把范围最后一天作为截止时间明确回答（如「报名截止即学院初赛截止，6月24日」），并补充「各学院可另行通知」。"
    "不得因片段没写「报名截止」四个字就回答「未明确/无统一截止时间」。"
    "此规则针对基于片段阶段时间给出截止日，不违反规则 12（规则 12 禁止的是用 calculate 推算或编造原文没有的数字）。\n"
    "15. **重要**：查询具体人名/专有名词（如「张尊舒」）时，第一步直接以该名词本身作为检索词"
    "（如检索「张尊舒」），不要拼接「学院/专业/学生/获奖」等修饰词——修饰词是高频词，"
    "会把目标文档挤出检索结果。若拼接修饰词检索没拿到答案，回退用裸词再检一次。\n"
    "16. **重要**：**调用工具前不要输出任何文字说明**——不要写「I'll search the knowledge base…」"
    "「让我先查一下」「我将调用…」这类旁白，直接发起工具调用。工具返回后直接给最终答案，"
    "不要复述检索过程。全流程只用中文，任何情况下都不要输出英文（代码、专有名词、URL 除外）。\n"
    "\n"
    "## 隐私保护规则（重要）\n"
    "- 用户若询问**某位具体老师/员工的个人手机号、邮箱或私人住址**，一律不提供。\n"
    "  统一回复：「您好，本系统不提供个人联系方式查询，请前往河南工学院官网 www.hait.edu.cn 查找官方电话。」\n"
    "- 禁止编造任何电话号码、邮箱或住址。若检索结果中没有联系方式，不得自行补全。\n"
    "- 禁止编造任何日期、时间。若检索原文与元数据日期中均无明确时间，如实回答「知识库未记录具体时间」，建议用户查看官网原文；若只有文件发布日期，须注明「这是文件发布时间，并非政策中的时间」。\n"
    "- 学校的**官方办公电话**（如教务处、各学院办公室等公开渠道发布的座机）可以正常回答，但只能依据知识库原文，不得杜撰。"
)

# ═══════════════════════════════════════════════════════
# Tool Definitions（Anthropic Tool Use 格式）
# ═══════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": "在知识库中进行向量语义检索，返回相关文档片段、来源和相似度。适用于用户询问专业知识、文档内容等场景。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本，使用与用户问题最相关的关键词",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_current_time",
        "description": "获取当前的日期和时间（精确到秒）",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate",
        "description": "安全计算数学表达式。支持加减乘除、幂运算、括号。使用 AST 白名单确保安全性。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '(3 + 5) * 2' 或 '2 ** 10'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_knowledge_base_stats",
        "description": "查看知识库的统计信息，包括文档总数、文本块总数、各分类的文章数量。当用户询问「知识库有哪些文档」「有多少文章」「知识库规模」时使用。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weather",
        "description": "查询指定城市的实时天气信息，包括温度、体感温度、湿度、风速、天气状况等。支持中文城市名（如「北京」「郑州」「正阳县」）和英文城市名。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，支持中文（如「正阳县」「北京」「郑州」）或英文（如「London」「Tokyo」）",
                },
            },
            "required": ["city"],
        },
    },
]

# ═══════════════════════════════════════════════════════
# 开场白
# ═══════════════════════════════════════════════════════

GREETING = (
    "你好！我是 **AI Agent 智能助手** 🤖\n\n"
    "我能帮你做这些事：\n\n"
    "- 📄 **检索知识库** —— 搜索文档内容，回答专业问题\n"
    "- ⏰ **查询时间** —— 获取当前的日期和时间\n"
    "- 🧮 **数学计算** —— 安全地计算数学表达式\n\n"
    "我会根据你的问题自动判断需要哪种能力，并自主调用相应的工具。有什么可以帮你的？"
)

# ═══════════════════════════════════════════════════════
# VectorStore 单例
# ═══════════════════════════════════════════════════════

_vector_store: VectorStore | None = None
# 单例构建锁：检索走线程池（AgentLoop 与 MCP 是同进程两条线程），
# 并发首访会同时通过 None 检查各建一份（Chroma 持久化客户端重复打开）。
# 照 src/hybrid_retriever.py:114 的 _bm25_build_lock 做双检锁。
_vector_store_lock = threading.Lock()


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                _vector_store = VectorStore(
                    db_path=str(_PROJECT_ROOT / "chroma_db"),
                    collection_name="my_rag_collection",
                )
                logger.info("VectorStore 初始化完成")
    return _vector_store


# ═══════════════════════════════════════════════════════
# 工具实现
# ═══════════════════════════════════════════════════════

# 泛称前缀：含这些字的"XX学院"通常是泛指（哪些学院/各学院），不是具体校名，跳过
_SCHOOL_NAME_NOISE_CHARS = set("哪各本该这那我这你那谁每某多同几两")

def _extract_school_names(text: str) -> list[str]:
    """从文本中提取形如「XX大学/XX学院」的学校名候选。

    - 用 (?!生) 排除"大学生"这类误匹配
    - 前缀含泛指字（哪些/各/本…）的视为非校名，跳过
    """
    names = []
    for m in re.finditer(r"([一-龥]{2,8}?)(大学(?!生)|学院)", text):
        prefix, suffix = m.group(1), m.group(2)
        if any(ch in prefix for ch in _SCHOOL_NAME_NOISE_CHARS):
            continue
        names.append(prefix + suffix)
    return names


_internal_org_cache: set[str] | None = None


def _get_internal_org_names() -> set[str]:
    """知识库内的校内机构名集合（如"智能工程学院"）。

    从 Chroma category 元数据提取（去掉"新闻/通知/公告"后缀），用于库外闸门放行：
    校内二级学院（智能工程学院/车辆与交通工程学院…）不是"别的学校"，应该允许检索。
    模块级缓存；失败时回退空集合（闸门退回旧行为，只误伤不崩）。
    """
    global _internal_org_cache
    if _internal_org_cache is not None:
        return _internal_org_cache
    orgs: set[str] = set()
    try:
        vs = _get_vector_store()
        metas = vs.collection.get(include=["metadatas"])["metadatas"]
        for m in metas:
            cat = (m or {}).get("category", "")
            for suf in ("新闻", "通知", "公告"):
                if cat.endswith(suf) and len(cat) > len(suf):
                    cat = cat[: -len(suf)]
            if cat and ("学院" in cat or "大学" in cat):
                orgs.add(cat)
    except Exception as e:
        logger.warning("读取校内机构名单失败，库外闸门可能误伤校内学院: %s", e)
    _internal_org_cache = orgs
    return orgs


def _is_internal_org(name: str, internal_orgs: set[str]) -> bool:
    """判断 name 是否为校内机构（含简称）。

    支持两种匹配：
    - 精确：name 在名单里（如"智能工程学院"）
    - 简称前缀：name 去掉"学院/大学"后缀后的核心词，
      是某校内机构去掉后缀后的开头（如"车辆学院"→"车辆"
      是"车辆与交通工程学院"→"车辆与交通工程"的前缀）
    """
    core = name[: -2] if name.endswith(("学院", "大学")) else name
    for org in internal_orgs:
        if name == org:
            return True
        org_core = org[: -2] if org.endswith(("学院", "大学")) else org
        if core and org_core.startswith(core):
            return True
    return False


def _refuse_out_of_kb_school(query: str) -> str | None:
    """库外主体校验闸门：问的是知识库主体院校之外的学校 → 返回拒答文案，不让检索跑。

    规则：
    - query 里含知识库主体校名（如"河南工学院"）→ 放行
    - query 提到校内二级学院（精确名或简称，如"智能工程学院""车辆学院"）→ 放行
    - query 提到「XX大学」或「XX工学院」类独立院校（如"清华大学""河北工学院"）→ 拒答
    - 其他「XX学院」措辞（如"产业学院""现代产业学院联盟"）是普通名词，
      不作为库外学校拦截 → 放行（检索捞不到自然答"未找到"，无幻觉风险）
    - 没提任何具体校名 → 放行
    """
    if KB_SUBJECT_SCHOOL in query:
        return None
    internal_orgs = _get_internal_org_names()
    # 协作类词：外校名后紧跟这些词时，该校只是「背景/合作方」而非提问主体（如
    # "联赛由河南师范大学牵头"→ 问的是联赛不是师大）。放行让检索兜底，避免误伤
    # 知识库内关于本校参与活动的问答（eval Q8 败因）。
    _COLLAB = ("牵头", "联合", "共同", "与", "和", "邀", "合作", "主办", "协办", "参加", "参赛", "联")
    for name in _extract_school_names(query):
        if KB_SUBJECT_SCHOOL in name:
            return None
        if _is_internal_org(name, internal_orgs):
            continue
        # 只拦独立院校特征明显的：XX大学 / XX工学院
        if name.endswith("大学") or name.endswith("工学院"):
            # 检查该校名后紧跟的词是否属协作类（背景提及则放行）
            idx = query.find(name)
            tail = query[idx + len(name): idx + len(name) + 2]
            if tail and any(tail.startswith(w) for w in _COLLAB):
                continue
            return (
                f"【库外题拦截】用户查询的【{name}】不在知识库范围内。"
                f"知识库仅收录【{KB_SUBJECT_SCHOOL}】相关信息。\n"
                f"请直接如实告知用户：知识库中没有【{name}】的任何信息，无法回答该问题。"
                f"严禁补充【{KB_SUBJECT_SCHOOL}】或其他学校的任何信息（包括录取分数、招生数据等），"
                f"即使作为「另外还可以帮您…」之类的附加说明也不行。只拒绝，不展开。"
            )
    return None


def _extract_year(query: str) -> str | None:
    """从 query 里抓第一个 20xx 年份；没有就返回 None（不过滤）。"""
    m = re.search(r"20\d{2}", query)
    return m.group(0) if m else None


def _ms(a, b) -> str:
    """毫秒差格式化；任一为 None 返回 '-'（该阶段未执行，检索埋点用）。"""
    if a is None or b is None:
        return "-"
    return f"{int((b - a) * 1000)}"


# ═══════════════════════════════════════════════════════
# 罕见词精确召回（表格块/密集名单里人名的兜底）
# ═══════════════════════════════════════════════════════
#
# 背景：人物姓名常出现在「获奖名单表格块」里（一块挤 20 个名字+奖项，如 4项一等奖
# 文章的 Simuro 获奖行），向量嵌入被整块名字稀释 → 搜不到；BM25 靠 jieba 分词，
# 人名放进句子（「张尊舒是谁」）时 jieba 切法不稳定 → 精确匹配也失效。
# 兜底：从查询提取中文 2/3 字片段，用 Chroma $contains 对原文精确子串匹配，
# 只注入「罕见词」（命中块 ≤ 阈值）的匹配；高频词（学生/比赛/介绍 等）跳过防噪声。

_EXACT_RECALL_MAX_MATCH = 20   # 命中块数 ≤ 此值才视为罕见词注入；高频词命中成百上千块跳过
_EXACT_RECALL_MAX_INJECT = 3    # 单次检索最多注入的兜底块数（防上下文膨胀）
_EXACT_RECALL_MAX_SCAN = 10     # 单次检索最多 $contains 扫描次数（含被跳过的高频词），限时防长查询拖慢


def _is_cjk(ch: str) -> bool:
    """是否 CJK 汉字（用于判断片段是否独立成词：两侧非汉字才算独立）。"""
    return "一" <= ch <= "鿿"


def _is_standalone(text: str, gram: str) -> bool:
    """gram 在 text 中是否至少一次独立成词（两侧为空白/标点/顿号/边界，而非嵌在更长的词里）。

    区分「真实体」（获奖名单里的名字：余世民、刘京涛、张尊舒）与「词边界碎片」
    （如「深入贯彻」里恰好出现的 2 字子串「下人」）——后者不是实体，注入会引入噪声。
    """
    i = text.find(gram)
    while i != -1:
        before_ok = i == 0 or not _is_cjk(text[i - 1])
        after_ok = i + len(gram) >= len(text) or not _is_cjk(text[i + len(gram)])
        if before_ok and after_ok:
            return True
        i = text.find(gram, i + 1)
    return False


def _extract_recall_grams(query: str) -> list[str]:
    """从查询提取候选罕见词片段：中文 2/3 字 n-gram，去重后按长度降序。

    不做「短片段被长片段包含就丢弃」：人名常是 2 字（如「李娜」），若因被
    3 字片段（如下李娜）包含而丢弃就漏检了。噪声交给调用方的命中数阈值过滤。
    """
    runs = re.findall(r"[一-鿿]+", query)  # 连续中文字符串（数字/英文/空格自然分隔）
    grams: set[str] = set()
    for seg in runs:
        for n in (3, 2):
            for i in range(len(seg) - n + 1):
                grams.add(seg[i:i + n])
    return sorted(grams, key=lambda g: (-len(g), g))


def _rare_token_exact_recall(query: str, merged: list[dict], vs) -> list[dict]:
    """对主检索漏掉的人名/专有名词做 $contains 精确召回，注入 merged（就地追加并返回）。

    仅注入主结果未覆盖的罕见词匹配；主结果已含该词 → 跳过（检索已成功，不占扫描预算）。
    """
    injected = 0
    scans = 0
    for gram in _extract_recall_grams(query):
        if any(gram in d.get("text", "") for d in merged):
            continue  # 主结果已含该词，无需兜底
        scans += 1
        if scans > _EXACT_RECALL_MAX_SCAN:
            break
        try:
            hits = vs.collection.get(
                where_document={"$contains": gram},
                limit=_EXACT_RECALL_MAX_MATCH + 1,
            )
        except Exception as e:
            logger.error("精确召回失败 gram=%r: %s", gram, e)
            continue
        docs, metas = hits.get("documents", []), hits.get("metadatas", [])
        if len(docs) > _EXACT_RECALL_MAX_MATCH:
            continue  # 高频词，跳过防噪声
        present = {d.get("text", "") for d in merged}
        for text, meta in zip(docs, metas):
            if text in present:
                continue
            if not _is_standalone(text, gram):
                continue  # 词边界碎片（如「深入」里的「下人」），非实体，跳过防噪声
            merged.append({"text": text, "metadata": meta, "distance": 0.1})
            present.add(text)
            injected += 1
            if injected >= _EXACT_RECALL_MAX_INJECT:
                return merged
    return merged


def _retrieve_candidates(query: str, top_k: int = TOP_K) -> dict:
    """混合检索前半段：库外校验 → 年份过滤 → 向量+BM25 融合召回 → 去重 → 补块 → 可选重排。

    返回**信封**而不是纯候选列表。原因：这一段有三个非正常出口（库外拒答 /
    检索失败 / 未找到），装不进 list[dict]；埋点耗时也要一并带出。信封结构：
      {"candidates": list[dict], "short_circuit": str | None, "timings": dict}
    short_circuit 非 None 时，调用方应直接把它当检索结果返回，忽略 candidates。

    各环节解决什么问题：
    - 库外主体校验：问别校直接拒答（防张冠李戴）
    - 年份过滤：query 提到某年就只看该年文档，清掉跨年旧通知挤占名额
    - BM25+RRF 融合：纯向量捞不到的关键词精确命中（如 Q3「报名截止」），
      由 BM25 关键词召回补上，两路名次用 Reciprocal Rank Fusion 无参融合
    - 同文档补块：宽泛问题（如「有哪些安排」）常只命中文档开头，
      关键句（如 3月3-15日排查）埋在正文第二节，相似度捞不到，必须按 source 元数据整篇补齐
    - 可选重排：CrossEncoder 精排再截断，提高关键块进 LLM 上下文的概率

    模式开关（.env）：
    - RETRIEVAL_MODE=hybrid（默认）| vector（退纯向量，消融对照）
    - RETRIEVAL_RERANK=on/off（默认 off）
    BM25 依赖缺失或建索引失败 → 自动回退纯向量，不崩。
    """
    t0 = time.perf_counter()  # 检索耗时埋点起点
    refuse = _refuse_out_of_kb_school(query)
    if refuse:
        logger.info("检索拒绝（库外主体）: %r", query)
        return {"candidates": [], "short_circuit": refuse, "timings": {}}
    vs = _get_vector_store()
    t_vec = t_bm25 = t_mix = t_rr = t_ctx = None  # 各检索阶段时间戳（埋点；None = 该阶段未执行）

    # 1. 年份过滤（检索时过滤，不是返回后再滤）
    # 放宽为 ±1 年窗口：query 里的年份常是「内容年份」（2026年挑战杯/2024年度评选），
    # 而文档 year 元数据是「发布年份」，两者常差一年（如 2025-12 发布 2026 挑战杯通知、
    # 2025-01 发布 2024 年度评选公示）。精确单年会把这些目标文档滤掉（eval Q11/Q23 败因）。
    # 空年份（发布页无日期）仍放行；窗口外（≥2 年前）的旧通知仍排除，保持"清跨年旧通知"效果。
    where = None
    year = _extract_year(query)
    if year:
        y = int(year)
        where = {"year": {"$in": [str(y - 1), year, str(y + 1), ""]}}

    # 2. 召回：纯向量 / 混合（向量 + BM25 两路 RRF 融合）
    try:
        vector_results = vs.search_similar(query, n_results=top_k, where=where)
        t_vec = time.perf_counter()
    except Exception as e:
        logger.error("检索失败: %s", e)
        return {"candidates": [], "short_circuit": f"检索失败: {e}", "timings": {}}

    results = vector_results
    if RETRIEVAL_MODE == "hybrid":
        bm25 = get_bm25_retriever(vs.collection)
        if bm25 is not None:
            try:
                bm25_results = bm25.search(query, top_n=top_k, year=year)
                t_bm25 = time.perf_counter()
                results = rrf_fuse(vector_results, bm25_results, top_n=top_k)
                t_mix = time.perf_counter()
            except Exception as e:
                logger.error("混合融合失败，回退纯向量: %s", e)
                results = vector_results

    # 2b. 年份过滤兜底：带年份过滤召回到空 → 说明目标文档没打年份元数据
    # （发布页无日期），放宽到不过滤重试一次，避免把内容相关的旧文档滤丢
    if year and not results:
        logger.info("年份过滤(%s)无结果，回退不过滤重试", year)
        try:
            vector_results = vs.search_similar(query, n_results=top_k)
            t_vec = time.perf_counter()
            results = vector_results
            if RETRIEVAL_MODE == "hybrid":
                bm25 = get_bm25_retriever(vs.collection)
                if bm25 is not None:
                    bm25_results = bm25.search(query, top_n=top_k, year=None)
                    t_bm25 = time.perf_counter()
                    results = rrf_fuse(vector_results, bm25_results, top_n=top_k)
                    t_mix = time.perf_counter()
        except Exception as e:
            logger.error("年份回退检索失败: %s", e)
    if not results:
        return {"candidates": [], "short_circuit": "知识库中未找到相关内容。", "timings": {}}

    # 3. 按文本内容去重（同一通知被爬进两个分类目录 + 重叠窗口会产生内容相同的块）
    merged = []
    seen = set()
    for doc in results:
        text = doc.get("text", "")
        if text in seen:
            continue
        seen.add(text)
        merged.append(doc)

    # 3b. 罕见词精确召回兜底：向量+BM25 都漏掉的人名/专有名词，$contains 原文精确匹配强制召回。
    #     放在补块之前，让补块能一并带出该来源的上下文块（如获奖名单所在文章的正文）。
    merged = _rare_token_exact_recall(query, merged, vs)

    # 4. 同文档补块：对初始命中的每个来源，按 source 元数据拉取该文档全部块，
    #    追加未召回的后续块（关键句常埋在文档中后段，如 排查通知的 3月3-15日 和 台账 分别落在第二、三块）。
    #    - 跳过「【来源】...」元数据头块（纯噪音）
    #    - 每个来源限补 3 块、最多补 5 篇，避免块多的文档（如测评通知 7 块）独占预算
    sources_ordered = []
    src_best_sim: dict[str, float] = {}
    for doc in merged:
        src = (doc.get("metadata") or {}).get("source", "")
        sim = max(0.0, 1.0 - doc.get("distance", 0))
        if src:
            if src not in sources_ordered:
                sources_ordered.append(src)
            src_best_sim[src] = max(src_best_sim.get(src, 0.0), sim)
    cap = TOP_K + 12  # 初始 8 条 + 补块，总封顶 20（补块要给足预算，否则块多的文档独占名额）
    for src in sources_ordered[:5]:  # 最多补 5 篇文档
        try:
            extra = vs.collection.get(where={"source": src})
        except Exception:
            continue
        added = 0
        extra_texts = extra.get("documents", [])
        extra_metas = extra.get("metadatas", [])
        # 该来源可补块的顺序：前 3 块 + 中段 1 块 + 末块。
        # 末块常含落款/署名/日期/联系电话（如通知文末的发布日期），只补前 3 块会漏掉（eval Q10 败因）。
        # 中段块常埋关键句（如心理健康通知的「3月20日前台账钉钉」落在第 4 块），仅前3+末块会跳过它（eval Q5 败因）。
        fill_order = list(range(min(3, len(extra_texts))))
        if len(extra_texts) > 3:
            fill_order.append(len(extra_texts) - 1)
            if len(extra_texts) >= 6:  # 6 块以上才有真正的中段块（mid≥3，不与前3重叠）
                fill_order.insert(-1, len(extra_texts) // 2)  # 中段块插在末块之前
        for fi in fill_order:
            if len(merged) >= cap or added >= 4:
                break
            text = extra_texts[fi]
            if text.startswith("【来源】") or text in seen:
                continue
            seen.add(text)
            # 补块本身没算过与 query 的相似度，沿用所属来源的最高相似度展示，避免误导为 100%
            merged.append({"text": text, "metadata": extra_metas[fi], "distance": 1.0 - src_best_sim.get(src, 0.5)})
            added += 1
    t_ctx = time.perf_counter()  # 去重 + 同文档补块完成

    # 5. 可选重排：对补块后的候选用 CrossEncoder 精排再截断（默认关，省加载模型）。
    #    重排后 merged 长度 ≤ RERANK_TOP_N，天然被下方 `merged[:cap]` 截断兜底。
    if RETRIEVAL_RERANK == "on":
        try:
            merged = get_reranker().rerank(query, merged, top_n=RERANK_TOP_N)
            t_rr = time.perf_counter()
        except Exception as e:
            logger.error("重排失败，保留原顺序: %s", e)

    # ── 检索明细日志（配合 request_id 排查：各段耗时 + 命中块数）──
    t_total = time.perf_counter()
    ctx_start = t_mix if t_mix is not None else t_vec  # 补块段起点：hybrid 从融合后计，纯向量从召回后计
    timings = {
        "vector": _ms(t0, t_vec),
        "bm25": _ms(t_vec, t_bm25),
        "fuse": _ms(t_bm25, t_mix),
        "complement": _ms(ctx_start, t_ctx),
        "rerank": _ms(t_ctx, t_rr),
    }
    logger.info(
        "检索明细 query=%r mode=%s 向量=%sms BM25=%sms 融合=%sms 补块=%sms 重排=%sms 总=%dms 命中=%d块",
        query[:60], RETRIEVAL_MODE,
        timings["vector"], timings["bm25"], timings["fuse"],
        timings["complement"], timings["rerank"],
        int((t_total - t0) * 1000), len(merged[:cap]),
    )
    return {"candidates": merged[:cap], "short_circuit": None, "timings": timings}


def _format_context(candidates: list[dict]) -> str:
    """把候选块格式化成给 LLM 读的检索文本。

    格式与 _parse_sources（下方）的严格正则逐字耦合——多一个空格，
    前端「参考来源」卡片就会解析成空。改这里务必同步核对 _parse_sources。
    """
    lines = []
    for i, doc in enumerate(candidates, 1):
        # 截断到 800 字：块默认 500 字，前 200 字会漏掉后半块的关键数字，
        # 导致 LLM 反复换说法搜索却拿不到答案，轮数耗尽报"处理超时"。
        text = doc.get("text", "")[:800]
        distance = doc.get("distance", 0)
        similarity = max(0.0, 1.0 - distance)
        metadata = doc.get("metadata", {}) or {}
        source = metadata.get("source", "未知文档")
        source_url = metadata.get("url", "")
        source_date = metadata.get("publish_date", "")
        source_year = metadata.get("year", "")
        source_cat = metadata.get("category", "")
        source_site = metadata.get("source_site", "")
        # 附加信息行
        extra = ""
        if source_date:
            extra += f" | 日期: {source_date}"
        if source_year:
            extra += f" | 年份: {source_year}"
        if source_cat:
            extra += f" | 分类: {source_cat}"
        if source_site:
            extra += f" | 站点: {source_site}"
        lines.append(
            f"[{i}] 相似度: {similarity:.1%} | 来源: {source}{extra}\n"
            f"   原文链接: {source_url}\n"
            f"   片段: {text}..."
        )
    return "\n\n".join(lines)


def _search_plain(query: str, top_k: int = TOP_K) -> str:
    """检索 + 格式化，不做召回自检（= 改造前的原行为）。"""
    env = _retrieve_candidates(query, top_k)
    if env["short_circuit"]:
        return env["short_circuit"]
    return _format_context(env["candidates"])


# ── 召回自检图单例（懒构建；构建锁防并发双建，与 VectorStore 单例同理）──
_recall_graph = None
_recall_graph_lock = threading.Lock()


def _get_recall_graph():
    global _recall_graph
    if _recall_graph is None:
        with _recall_graph_lock:
            if _recall_graph is None:
                from src.recall_guard import build_recall_graph

                _recall_graph = build_recall_graph(
                    retrieve_fn=_retrieve_candidates,
                    format_fn=_format_context,
                    llm_fn=_call_llm,
                    api_key=API_KEY,
                )
                logger.info("召回自检图构建完成（max_attempts=%d）", RECALL_GUARD_MAX_ATTEMPTS)
    return _recall_graph


async def _search_knowledge_base_async(query: str, top_k: int = TOP_K) -> str:
    """检索 + 召回自检（异步真实实现）。RECALL_GUARD=off 时等价于原路径。"""
    if RECALL_GUARD != "on":
        return _search_plain(query, top_k)
    from src.recall_guard import run_recall_guard

    return await run_recall_guard(_get_recall_graph(), query, top_k, RECALL_GUARD_MAX_ATTEMPTS)


def _search_knowledge_base(query: str, top_k: int = TOP_K) -> str:
    """混合检索（公开同步 API）—— AgentLoop 的工具与 MCP 都调这里。

    RECALL_GUARD=on 时走 LangGraph 召回自检图（判定不充分会改写 query 重检、
    主题不符会拒答）；off 时是原样的"检索直接格式化"。
    """
    if RECALL_GUARD != "on":
        return _search_plain(query, top_k)

    # 图的节点是 async（判官要 await _call_llm），这里从同步 API 桥过去。
    # 两个调用方（AgentLoop / MCP server）都在 asyncio.to_thread 里，工作线程内
    # 没有 running loop，asyncio.run 是安全的；但仍探测一次兜住"将来有人直接在
    # async 上下文里调它"的情况。不用 nest_asyncio——它会全局 patch asyncio 且
    # 仍是在当前 loop 里同步阻塞，后果一样却多一份全局状态污染。
    #
    # 整体 try/except 是必须的：AgentLoop 调工具那行（await asyncio.to_thread(
    # execute_tool, ...)）外面没有错误边界，图里任何异常抛出去都会打断整条 SSE 流。
    try:
        coro = _search_knowledge_base_async(query, top_k)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        logger.warning("检测到 running loop，召回自检降级到新线程执行（会阻塞当前事件循环数秒）")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    except Exception:
        logger.exception("召回自检图执行失败，降级为无自检检索")
        return _search_plain(query, top_k)


def _get_current_time() -> str:
    now = datetime.now()
    wd = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} {wd[now.weekday()]}"


# AST 白名单（来自项目三）


_AST_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and type(node.op) not in _AST_OPS:
                return f"❌ 不支持的运算符: {type(node.op).__name__}"
            if isinstance(node, ast.UnaryOp) and type(node.op) not in _AST_OPS:
                return f"❌ 不支持的运算符: {type(node.op).__name__}"
            if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.JoinedStr)):
                return f"❌ 不支持的操作: {type(node).__name__}"
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except ZeroDivisionError:
        return "❌ 错误: 除数不能为零"
    except Exception as e:
        return f"❌ 计算错误: {e}"


def _guard_tool_call(name: str, user_question: str) -> str | None:
    """工具调用前的**库外主体二次校验**，用用户原话。返回拒答文案，或 None 放行。

    检索层自己那道闸门（`_retrieve_candidates` 里的 `_refuse_out_of_kb_school`）
    只看得到**工具参数**，而那是 LLM 改写过的：问「河北工学院的录取分数线」时
    LLM 可能把参数写成 `query='招生录取分数'`——**校名被改写掉了**，闸门没东西可拦，
    于是本校（河南工学院）的分数被如实列进回答，评分按「含基准表外数字即判幻觉」给 0。
    实测同一份代码跑两遍得到 60/60 与 50/60，差别就在这一次改写。所以这里用
    **用户原话**再拦一次 —— 它才是「问的是不是外校」的权威依据。

    只对 search_knowledge_base 生效：时间/计算/天气与知识库主体无关。

    不会误伤混合问句：`_refuse_out_of_kb_school` 只要在 query 里看到主体校名就早退放行，
    所以「河南工学院和河北工学院哪个好」照常放行。
    """
    if name != "search_knowledge_base":
        return None
    return _refuse_out_of_kb_school(user_question or "")


def execute_tool(name: str, input_: dict) -> str:
    if name == "search_knowledge_base":
        return _search_knowledge_base(**input_)
    if name == "get_current_time":
        return _get_current_time()
    if name == "calculate":
        return _calculate(**input_)
    if name == "get_knowledge_base_stats":
        return _get_kb_stats_tool()
    if name == "get_weather":
        return _get_weather(**input_)
    return f"未知工具: {name}"


def _get_weather(city: str) -> str:
    """通过 wttr.in 免费 API 查询城市天气（无需 API Key）"""
    import urllib.request
    import urllib.parse

    if not city or not city.strip():
        return "⚠️ 请提供城市名称，如「北京」「正阳县」「Tokyo」"

    city_enc = urllib.parse.quote(city.strip())
    url = f"https://wttr.in/{city_enc}?format=j1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current_condition", [{}])[0]
        if not current:
            return f"⚠️ 未找到城市「{city}」的天气数据，请检查城市名是否正确。"

        # 基本信息
        temp_c = current.get("temp_C", "?")
        feels_like = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        wind_speed = current.get("windspeedKmph", "?")
        wind_dir = current.get("winddir16Point", "?")
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "未知")
        visibility = current.get("visibility", "?")

        # 最近预报（今天 + 明天）
        forecast_lines = []
        for day in data.get("weather", [])[:2]:
            date = day.get("date", "?")
            max_t = day.get("maxtempC", "?")
            min_t = day.get("mintempC", "?")
            avg_t = day.get("avgtempC", "?")
            hourly = day.get("hourly", [])
            desc = hourly[0].get("weatherDesc", [{}])[0].get("value", "") if hourly else ""
            forecast_lines.append(
                f"  {date}: {desc}，{min_t}°C ~ {max_t}°C（均温 {avg_t}°C）"
            )

        lines = [
            f"🌤️ **{city.strip()}** 实时天气",
            f"",
            f"🌡️ 温度: {temp_c}°C（体感 {feels_like}°C）",
            f"💧 湿度: {humidity}%",
            f"🌬️ 风速: {wind_speed} km/h（{wind_dir}）",
            f"👁️ 能见度: {visibility} km",
            f"☁️ 天气: {weather_desc}",
        ]
        if forecast_lines:
            lines.append(f"")
            lines.append(f"📅 未来预报：")
            lines.extend(forecast_lines)

        lines.append(f"")
        lines.append(f"数据来源: wttr.in")

        return "\n".join(lines)

    except urllib.error.URLError as e:
        logger.error("天气查询网络错误: %s", e)
        return f"❌ 天气查询网络超时，请稍后重试"
    except Exception as e:
        logger.exception("天气查询失败")
        return f"❌ 天气查询失败: {e}"


# ═══════════════════════════════════════════════════════
# 来源解析
# ═══════════════════════════════════════════════════════

def _parse_sources(result_text: str) -> list[dict]:
    """从检索结果文本中提取结构化的来源信息"""
    sources = []
    pattern = (
        r'\[(\d+)\] 相似度: ([\d.]+)% \| 来源: (.+?)(?: \| 日期: (.+?))?'
        r'(?: \| 年份: (.+?))?(?: \| 分类: (.+?))?(?: \| 站点: (.+?))?\n'
        r'   原文链接: (.+?)\n'
        r'   片段: (.+?)\.\.\.'
    )
    for match in re.finditer(pattern, result_text, re.DOTALL):
        sources.append({
            "index": int(match.group(1)),
            "similarity": float(match.group(2)) / 100,
            "source": match.group(3).strip(),
            "date": (match.group(4) or "").strip(),
            "year": (match.group(5) or "").strip(),
            "category": (match.group(6) or "").strip(),
            "site": (match.group(7) or "").strip(),
            "url": (match.group(8) or "").strip(),
            "text": match.group(9).strip(),
        })
    return sources


# ═══════════════════════════════════════════════════════
# LLM 调用（Anthropic-compatible，支持 Tool Use）
# ═══════════════════════════════════════════════════════

async def _call_llm(
    messages: list[dict],
    api_key: str,
    on_delta=None,
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict:
    """调用 Anthropic-compatible API，返回 {text, tool_uses, stop_reason}。

    on_delta 传入时启用流式（payload 加 stream: True）：每个 text_delta 实时回调一次，
    供 SSE 边生成边出字；不传则一次性等全量响应（同旧行为）。

    system / tools / max_tokens / temperature 是给"非主循环调用方"（如召回自检的判官
    LLM）留的覆盖口，**默认值全部保持主循环原状**，不传就与改动前完全一致：
    - system=None → SYSTEM_PROMPT
    - tools=None  → TOOL_DEFINITIONS；传 [] 则整个 tools 字段不下发（判官不需要工具）
    - max_tokens / temperature = None → .env 的 MAX_TOKENS / TEMPERATURE

    重试策略：网络错误 / HTTP 5xx / 请求阶段超时 在发起后重试（最多 LLM_MAX_RETRIES 次）。
    流式中途断开不重试（可能已向客户端吐出部分文本，重发会造成重复），
    由调用方（run_stream）回退到非流式整段输出。
    """
    last_error = ""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        if attempt > 1:
            logger.warning("LLM 调用重试第 %d/%d 次（前次: %s）", attempt, LLM_MAX_RETRIES, last_error)
            # 重试前退避一下，避免连续抖动时立刻再次失败
            await asyncio.sleep(min(1.5 * (attempt - 1), 3.0))

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        _tools = TOOL_DEFINITIONS if tools is None else tools
        payload = {
            "model": MODEL_NAME,
            "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
            "temperature": temperature if temperature is not None else TEMPERATURE,
            "system": system if system is not None else SYSTEM_PROMPT,
            "messages": messages,
            # 禁用思考模式：deepseek 默认返回 thinking 块，工具调用后回传 assistant
            # 必须原样带 thinking+signature，否则报 400。RAG 查询场景无需思考链，直接禁用
            "thinking": {"type": "disabled"},
        }
        if _tools:  # tools=[] 的调用方（判官）不传该字段，避免下发空数组
            payload["tools"] = _tools
        if on_delta is not None:
            payload["stream"] = True

        in_body = False     # 是否已进入响应体读取（流式中断需区分，避免重复输出）
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    API_URL, headers=headers, json=payload,
                    # 显式超时：deepseek 单轮生成最长约 120s，超时返回友好提示而非挂死
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("LLM API 错误 (%d): %s", resp.status, error_text[:200])
                        last_error = f"HTTP {resp.status}"
                        if resp.status < 500:
                            # 4xx 是请求本身的问题，重试无意义，直接失败
                            return {
                                "text": f"API 调用失败 ({resp.status})",
                                "tool_uses": [],
                                "stop_reason": "end_turn",
                            }
                        continue  # 5xx 服务端抖动，重试
                    in_body = True
                    if on_delta is not None:
                        return await _parse_anthropic_stream(resp, on_delta)
                    data = await resp.json()
                    break
        except asyncio.TimeoutError:
            if in_body and on_delta is not None:
                raise  # 流式中途超时：已向客户端吐出部分文本，交给上层回退非流式
            logger.error("LLM API 调用超时 (>120s)")
            last_error = "timeout"
            continue
        except aiohttp.ClientError as e:
            if in_body and on_delta is not None:
                raise  # 流式连接断开：同上，交给上层回退
            logger.error("LLM 网络错误: %s", e)
            last_error = f"network: {type(e).__name__}"
            continue

    if 'data' not in locals():
        # 重试全部耗尽，返回友好提示
        return {
            "text": "API 调用暂时失败，请稍后重试。",
            "tool_uses": [],
            "stop_reason": "end_turn",
        }

    content_blocks = data.get("content", [])
    text_parts = []
    tool_uses = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_uses.append(block)

    if tool_uses:
        logger.info("LLM → %d 个工具: %s", len(tool_uses), [t["name"] for t in tool_uses])

    return {
        "text": "\n".join(text_parts),
        "tool_uses": tool_uses,
        "stop_reason": data.get("stop_reason", "end_turn"),
    }


async def _parse_anthropic_stream(resp, on_delta) -> dict:
    """解析 Anthropic 兼容的 SSE 流，重建 {text, tool_uses, stop_reason}。

    事件：
    - content_block_start：标记 text / tool_use 块（tool_use 带 id/name）
    - content_block_delta：text_delta → 实时回调 + 累积；input_json_delta → 累积重建工具入参
    - message_delta：取 stop_reason
    - message_stop：流结束

    流中没有任何有效内容块（如服务端返回非 SSE 错误体）时抛 RuntimeError，
    由调用方回退到非流式重发，保证不静默丢答案。
    """
    text_parts: list[str] = []
    tool_blocks: dict[int, dict] = {}
    stop_reason = "end_turn"
    saw_content = False

    while True:
        raw_line = await resp.content.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "content_block_start":
            saw_content = True
            block = evt.get("content_block", {})
            if block.get("type") == "tool_use":
                tool_blocks[evt.get("index", len(tool_blocks))] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "partial_json": "",
                }
        elif etype == "content_block_delta":
            delta = evt.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text", "")
                if text:
                    on_delta(text)
                    text_parts.append(text)
            elif dtype == "input_json_delta":
                block = tool_blocks.get(evt.get("index"))
                if block:
                    block["partial_json"] += delta.get("partial_json", "")
        elif etype == "message_delta":
            sr = (evt.get("delta") or {}).get("stop_reason")
            if sr:
                stop_reason = sr

    if not saw_content:
        raise RuntimeError("流式响应无有效内容块")

    tool_uses = []
    for idx in sorted(tool_blocks):
        blk = tool_blocks[idx]
        inp = {}
        if blk["partial_json"]:
            try:
                inp = json.loads(blk["partial_json"])
            except json.JSONDecodeError:
                logger.warning("工具入参 JSON 解析失败: %s", blk["partial_json"][:200])
        tool_uses.append({"id": blk["id"], "name": blk["name"], "input": inp})

    if tool_uses:
        logger.info("LLM → %d 个工具: %s", len(tool_uses), [t["name"] for t in tool_uses])

    return {
        "text": "\n".join(text_parts),
        "tool_uses": tool_uses,
        "stop_reason": stop_reason,
    }


# ═══════════════════════════════════════════════════════
# Agent 事件类型 —— 连接 Agent Loop 和表现层（SSE / Rich）
# 对齐项目二工作流引擎：事件驱动，for-step 循环。
# 六个事件类型与项目二一致；ThinkingEvent/DoneEvent 因项目一
# 前端协议需要携带格式化好的步骤文本与完整来源。
# ═══════════════════════════════════════════════════════

@dataclass
class ThinkingEvent:
    """每轮推理开始时触发。step 为格式化好的完整步骤文本（前端直接展示）。"""
    step: str


@dataclass
class ToolCallEvent:
    """Agent 决定调用某个工具。"""
    tool: str
    args: dict


@dataclass
class ToolResultEvent:
    """工具执行完毕。sources 为项目一扩展：检索工具附带结构化来源。"""
    tool: str
    success: bool
    output: str
    sources: list | None = None


@dataclass
class TextEvent:
    """Agent 最终回复文本。"""
    content: str


@dataclass
class DoneEvent:
    """任务正常完成，携带完整思考步骤与来源（前端落盘用）。"""
    thinking: list
    sources: list


@dataclass
class ErrorEvent:
    """发生错误（API 故障 / 超时 / 达到上限）。"""
    message: str


# 所有事件类型的联合（用于类型标注）
AgentEvent = ThinkingEvent | ToolCallEvent | ToolResultEvent | TextEvent | DoneEvent | ErrorEvent


class AgentLoop:
    """Agent 核心循环 —— 事件驱动版（对齐项目二工作流引擎）。

    流程：用户输入 → LLM 分析 → 调工具 / 给出答案 → 循环 → 返回结果

    支持两种调用方式：
    - run(): 返回最终文本
    - run_stream(): async generator，yield 事件对象（SSE / 前端消费）
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        max_rounds: int = MAX_ROUNDS,
    ):
        self.api_key = api_key
        self.max_rounds = max_rounds

    # ── 公开 API ──────────────────────────────────────────

    async def run(self, message: str, history: list[dict] | None = None) -> str:
        """执行一次对话，返回最终文本。内部收集 run_stream 的事件。"""
        final_text = ""
        async for event in self.run_stream(message, history):
            if isinstance(event, TextEvent):
                final_text += event.content  # TextEvent 现为逐 chunk，需累加成完整答案
        return final_text

    async def run_stream(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """事件驱动主循环：for-step 保证有限步数终止，每步 yield 事件对象。"""
        thinking_steps: list[str] = []
        sources: list[dict] = []
        step_no = 0
        start_time = time.time()

        def now() -> str:
            return datetime.now().strftime("%H:%M:%S")

        def add_step(text: str) -> str:
            nonlocal step_no
            step_no += 1
            step = f"[{now()}] Step {step_no}: {text}"
            thinking_steps.append(step)
            return step

        def _normalize_content(content) -> str:
            """历史消息 content 可能是 list[dict] 或 str，统一转成 str"""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return str(content) if content else ""

        # ── 构建消息列表（先截断 history 防上下文超长）──
        messages: list[dict] = []
        for msg in (history or [])[-MAX_HISTORY_TURNS:]:
            role = msg.get("role", "user")
            content = _normalize_content(msg.get("content", ""))
            if not content or content.startswith("⏳"):
                continue
            messages.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})
        messages.append({"role": "user", "content": message})

        # ── 初次事件：用户消息已接收 ──
        yield ThinkingEvent(
            step=add_step(f"接收 Query: [{message[:60]}{'...' if len(message) > 60 else ''}]")
        )

        for rnd in range(self.max_rounds):
            yield ThinkingEvent(step=add_step(f"第 {rnd + 1} 轮推理 — 调用 LLM..."))

            # ── 流式调用 LLM：边生成边 yield TextEvent，同时拿到完整 response ──
            # on_delta 把每个 text 增量塞进队列，本协程并发排空队列实时转发；
            # 流式解析失败时回退非流式重发（stream_error 置位），答案交给回退路径整段输出。
            chunk_queue: asyncio.Queue = asyncio.Queue()
            stream_error = [False]
            streamed_chunks = 0

            async def _llm_with_stream() -> dict:
                try:
                    return await _call_llm(messages, self.api_key, on_delta=chunk_queue.put_nowait)
                except Exception:
                    stream_error[0] = True
                    logger.exception("流式调用失败，回退非流式")
                    return await _call_llm(messages, self.api_key)
                finally:
                    await chunk_queue.put(None)  # 哨兵：排空循环结束

            llm_task = asyncio.create_task(_llm_with_stream())
            # 每轮一个闸门：扣住轮首的英文旁白（详见 src/stream_gate.py）
            gate = OpeningGate()
            try:
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        break
                    if stream_error[0]:
                        continue  # 流式已失败：丢弃残留缓冲，不输出残缺片段
                    out = gate.feed(chunk)
                    if out:
                        streamed_chunks += 1
                        yield TextEvent(content=out)
                response = await llm_task
                # 收尾：本轮扣着没发的，有 tool_use 就是旁白（丢），没有就是纯英文回答（补发）
                tail = gate.finish(had_tool_use=bool(response.get("tool_uses")))
                if tail:
                    streamed_chunks += 1
                    yield TextEvent(content=tail)
            finally:
                if not llm_task.done():
                    llm_task.cancel()

            if response.get("tool_uses"):
                for tu in response["tool_uses"]:
                    name = tu["name"]
                    inp = tu.get("input", {})
                    tid = tu.get("id", f"tool_{rnd}")

                    yield ToolCallEvent(tool=name, args=inp)

                    # 同步阻塞的工具执行（ChromaDB 查询 / BM25 打分 / weather 请求）
                    # 丢线程池执行，避免卡住事件循环（多人并发问答互不阻塞）
                    # 先过库外主体二次校验（用户原话）——命中就不必跑检索了
                    blocked = _guard_tool_call(name, message)
                    result_text = (
                        blocked if blocked
                        else await asyncio.to_thread(execute_tool, name, inp)
                    )

                    if name == "search_knowledge_base":
                        ev_sources = _parse_sources(result_text)
                        sources = ev_sources
                        yield ToolResultEvent(
                            tool=name, success=True,
                            output=result_text, sources=ev_sources,
                        )
                        # 两种拒答都要在"检索到 N 条"判定之前，否则会出现
                        # "检索到 0 条" + "已拒答" 两个自相矛盾的步骤。
                        # 前缀必须具体：宽泛的「【」会把库外拦截误标成自检未通过。
                        # 必须 yield ThinkingEvent——只 add_step 的话它只进 thinking_steps
                        # 列表（最后才随 DoneEvent 一次性下发），流式过程中客户端看不到，
                        # 于是"这一步到底走没走"在实时日志和评测回放里都查不出来。
                        if result_text.startswith("【召回自检"):
                            yield ThinkingEvent(step=add_step("🛡️ 召回自检未通过，已拒答"))
                        elif result_text.startswith("【库外题拦截"):
                            yield ThinkingEvent(step=add_step("🚫 库外题拦截，已拒答"))
                        else:
                            yield ThinkingEvent(step=add_step(f"📋 检索到 {len(ev_sources)} 条相关结果"))
                    else:
                        yield ToolResultEvent(
                            tool=name, success=True,
                            output=result_text, sources=None,
                        )
                        yield ThinkingEvent(step=add_step(f"📋 返回: {result_text[:120]}"))

                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
                    })
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tid, "content": result_text}],
                    })
            else:
                answer = response.get("text", "")
                elapsed = time.time() - start_time
                add_step(f"✅ 生成完成 | {rnd + 1} 轮 | 耗时 {elapsed:.1f}s")
                # 已流式输出的（有增量且无失败）不必重复发整段；
                # 未流式/流式失败场景在此整段补发，保证答案完整。
                if stream_error[0] or streamed_chunks == 0:
                    yield TextEvent(content=answer)
                yield DoneEvent(thinking=list(thinking_steps), sources=list(sources))
                return

        # 达到最大轮数
        add_step(f"⚠️ 达到最大轮数 {self.max_rounds}，强制终止")
        fallback = "抱歉，处理超时，请简化问题后重试。"
        yield TextEvent(content=fallback)
        yield DoneEvent(thinking=list(thinking_steps), sources=list(sources))


# ═══════════════════════════════════════════════════════
# Agent 循环（异步生成器，每步 yield 一次给 Gradio）—— 旧版，仅作兼容保留
# ═══════════════════════════════════════════════════════

async def run_agent(message: str, history: list[dict], api_key: str):
    """
    Agent 核心循环。
    每完成一个步骤（推理/调工具/结果），yield (messages_to_append, sources, thinking_steps)
    参考：项目二事件驱动 + 项目三 Anthropic Tool Use 格式
    """
    thinking_steps: list[str] = []
    sources: list[dict] = []
    start_time = time.time()
    step_no = 0

    def now() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def add_step(text: str):
        nonlocal step_no
        step_no += 1
        thinking_steps.append(f"[{now()}] **Step {step_no}**: {text}")

    def ts():
        """简短时间戳，显示在气泡上方"""
        return datetime.now().strftime("%H:%M")

    # ── 构建消息列表 ──
    def _normalize_content(content) -> str:
        """Gradio 6.x 的 content 可能是 list[dict] 或 str，统一转成 str"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content) if content else ""

    messages: list[dict] = []
    for msg in (history or []):
        role = msg.get("role", "user")
        content = _normalize_content(msg.get("content", ""))
        if not content or content.startswith("⏳"):
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    # ── 初次 yield：用户消息 ──
    add_step(f"接收 Query: [{message[:60]}{'...' if len(message) > 60 else ''}]")
    yield [{"role": "user", "content": message, "metadata": {"title": ts()}}], sources, list(thinking_steps)

    try:
        for rnd in range(MAX_ROUNDS):
            add_step(f"第 {rnd + 1} 轮推理 — 调用 LLM...")
            yield [], sources, list(thinking_steps)

            response = await _call_llm(messages, api_key)

            if response.get("tool_uses"):
                for tu in response["tool_uses"]:
                    name = tu["name"]
                    inp = tu.get("input", {})
                    tid = tu.get("id", f"tool_{rnd}")

                    add_step(f"🔧 调用工具: `{name}({json.dumps(inp, ensure_ascii=False)})`")
                    yield [], sources, list(thinking_steps)

                    # 库外主体二次校验（用户原话），命中就不跑检索
                    blocked = _guard_tool_call(name, message)
                    result_text = blocked if blocked else execute_tool(name, inp)

                    if name == "search_knowledge_base":
                        sources = _parse_sources(result_text)
                        if result_text.startswith("【召回自检"):
                            add_step("🛡️ 召回自检未通过，已拒答")
                        elif result_text.startswith("【库外题拦截"):
                            add_step("🚫 库外题拦截，已拒答")
                        else:
                            add_step(f"📋 检索到 {len(sources)} 条相关结果")
                    else:
                        add_step(f"📋 返回: {result_text[:120]}")

                    yield [], sources, list(thinking_steps)

                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
                    })
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tid, "content": result_text}],
                    })
            else:
                answer = response.get("text", "")
                elapsed = time.time() - start_time
                add_step(f"✅ 生成完成 | {rnd + 1} 轮 | 耗时 {elapsed:.1f}s")

                yield [
                    {"role": "assistant", "content": answer, "metadata": {"title": ts()}},
                ], sources, list(thinking_steps)
                return

        add_step(f"⚠️ 达到最大轮数 {MAX_ROUNDS}，强制终止")
        yield [
            {"role": "assistant", "content": "抱歉，处理超时，请简化问题后重试。", "metadata": {"title": ts()}},
        ], sources, list(thinking_steps)

    except Exception as e:
        logger.exception("Agent 异常")
        add_step(f"❌ 异常: {e}")
        yield [
            {"role": "assistant", "content": f"抱歉，处理请求时出错: {e}", "metadata": {"title": ts()}},
        ], sources, list(thinking_steps)


# ═══════════════════════════════════════════════════════
# 知识库统计 + 文件入库（供 Gradio 调用）
# ═══════════════════════════════════════════════════════

def _get_kb_stats_tool() -> str:
    """工具版本：返回人类可读的知识库统计信息"""
    stats = get_kb_stats()
    total_articles = stats.get("total_articles", 0)
    total_chunks = stats.get("total_chunks", 0)
    category_counts = stats.get("category_counts", {})
    lines = [
        f"知识库统计：",
        f"- 文档总数：{total_articles} 篇",
        f"- 文本块总数：{total_chunks} 块",
    ]
    if category_counts:
        lines.append("- 各分类文章数：")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  · {cat}：{count} 篇")
    else:
        lines.append("- 暂无分类统计")
    return "\n".join(lines)


def get_kb_stats() -> dict:
    """查询知识库统计信息（文章数、分块数、分类明细）。"""
    try:
        from campus_scraper.pipeline import get_knowledge_base_stats
        vs = _get_vector_store()
        stats = get_knowledge_base_stats(vs)
        return {
            "total_articles": stats.total_articles,
            "total_chunks": stats.total_chunks,
            "category_counts": stats.category_counts,
        }
    except Exception as e:
        logger.exception("查询 KB 统计失败")
        return {
            "total_articles": 0,
            "total_chunks": _get_vector_store().count(),
            "category_counts": {},
            "error": str(e),
        }


async def ingest_files(file_paths: list[str]) -> dict:
    """处理用户上传的文件，存入知识库。"""
    try:
        from campus_scraper.pipeline import ingest_uploaded_files
        vs = _get_vector_store()
        result = await ingest_uploaded_files(file_paths, vs)
        return result
    except Exception as e:
        logger.exception("文件入库失败")
        return {"total_chunks": 0, "files_processed": 0, "errors": [str(e)]}


async def run_scraper_pipeline(max_pages: int = 2) -> dict:
    """
    运行爬虫管线（爬取 → 清洗 → 入库）。
    max_pages 控制每类爬几页，默认 2 用于测试。
    """
    try:
        from campus_scraper.pipeline import run_scrape_pipeline
        vs = _get_vector_store()
        result = await run_scrape_pipeline(max_pages=max_pages, vector_store=vs)
        return result
    except Exception as e:
        logger.exception("爬虫管线失败")
        return {"total_articles": 0, "total_chunks": 0, "errors": [str(e)]}
