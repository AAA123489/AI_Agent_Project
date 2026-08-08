"""
app_backend.py — Agent 后端封装
===============================
封装 VectorStore 检索、LLM 调用（Anthropic Tool Use 格式）、Agent 循环。
不修改任何现有代码，只做导入和适配。

为 Gradio 前端提供 run_agent() 异步生成器。
"""

import ast
import asyncio
import json
import logging
import operator as op
import os
import re
import sys
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
# 知识库主体院校：用于识别「问别的学校」的库外题，避免张冠李戴幻觉（如 Q6 河北工学院）
KB_SUBJECT_SCHOOL = os.getenv("KB_SUBJECT_SCHOOL", "河南工学院")
# Agent 循环上限：工具结果已完整返回（800 字覆盖整个块），单题 1~2 轮即可回答。
# 设 6 兜底：正常单题 2~3 轮完成，极端复杂题也不会干等到 10 轮。
MAX_ROUNDS = 6


def get_active_params() -> str:
    """返回当前生效的运行时参数摘要（用于对话日志标注消融实验组）。

    分块参数（CHUNK_SIZE/CHUNK_OVERLAP）在重建阶段，运行时读不到，
    需靠 .env 的 EXPERIMENT_TAG 手动标注。
    """
    return f"model={MODEL_NAME} | temperature={TEMPERATURE} | max_tokens={MAX_TOKENS} | top_k={TOP_K}"

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
    "7. **重要**：知识库检索不到相关内容时，请基于你自己的知识直接回答用户，"
    "并诚实说明「知识库中暂无相关信息，以下是基于通用知识的回答」\n"
    "8. 引用知识库内容时标注来源文件名\n"
    "9. 用中文回复\n"
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


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            db_path=str(_PROJECT_ROOT / "chroma_db"),
            collection_name="my_rag_collection",
            distance_threshold=0.85,
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


def _refuse_out_of_kb_school(query: str) -> str | None:
    """库外主体校验闸门：问的是知识库主体院校之外的学校 → 返回拒答文案，不让检索跑。

    规则：
    - query 里含知识库主体校名（如"河南工学院"）→ 放行
    - query 提到别的学校（如"河北工学院""清华大学"）→ 拒答
    - 没提任何具体校名 → 放行
    """
    if KB_SUBJECT_SCHOOL in query:
        return None
    for name in _extract_school_names(query):
        if KB_SUBJECT_SCHOOL in name:
            return None
        return (
            f"【库外题拦截】用户查询的【{name}】不在知识库范围内。"
            f"知识库仅收录【{KB_SUBJECT_SCHOOL}】相关信息。\n"
            f"请直接如实告知用户：知识库中没有【{name}】的任何信息，无法回答该问题；"
            f"切勿使用【{KB_SUBJECT_SCHOOL}】或其他学校的信息代替作答。"
        )
    return None


def _search_knowledge_base(query: str, top_k: int = TOP_K) -> str:
    """真实向量检索（入口先做库外主体校验）"""
    refuse = _refuse_out_of_kb_school(query)
    if refuse:
        return refuse
    try:
        vs = _get_vector_store()
        results = vs.search_similar(query, n_results=top_k)
        if not results:
            return "知识库中未找到相关内容。"

        lines = []
        for i, doc in enumerate(results, 1):
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
    except Exception as e:
        logger.error("检索失败: %s", e)
        return f"检索失败: {e}"


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

async def _call_llm(messages: list[dict], api_key: str) -> dict:
    """调用 Anthropic-compatible API，返回 {text, tool_uses, stop_reason}"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        # 禁用思考模式：deepseek 默认返回 thinking 块，工具调用后回传 assistant
        # 必须原样带 thinking+signature，否则报 400。RAG 查询场景无需思考链，直接禁用
        "thinking": {"type": "disabled"},
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                API_URL, headers=headers, json=payload,
                # 显式超时：deepseek 单轮生成最长约 120s，超时返回友好提示而非挂死
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("LLM API 错误 (%d): %s", resp.status, error_text[:200])
                    return {
                        "text": f"API 调用失败 ({resp.status})",
                        "tool_uses": [],
                        "stop_reason": "end_turn",
                    }
                data = await resp.json()
        except asyncio.TimeoutError:
            logger.error("LLM API 调用超时 (>120s)")
            return {
                "text": "API 调用超时，请稍后重试。",
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
                final_text = event.content
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

        # ── 构建消息列表 ──
        messages: list[dict] = []
        for msg in (history or []):
            role = msg.get("role", "user")
            content = _normalize_content(msg.get("content", ""))
            if not content or content.startswith("⏳"):
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        # ── 初次事件：用户消息已接收 ──
        yield ThinkingEvent(
            step=add_step(f"接收 Query: [{message[:60]}{'...' if len(message) > 60 else ''}]")
        )

        for rnd in range(self.max_rounds):
            yield ThinkingEvent(step=add_step(f"第 {rnd + 1} 轮推理 — 调用 LLM..."))

            response = await _call_llm(messages, self.api_key)

            if response.get("tool_uses"):
                for tu in response["tool_uses"]:
                    name = tu["name"]
                    inp = tu.get("input", {})
                    tid = tu.get("id", f"tool_{rnd}")

                    yield ToolCallEvent(tool=name, args=inp)

                    result_text = execute_tool(name, inp)

                    if name == "search_knowledge_base":
                        ev_sources = _parse_sources(result_text)
                        sources = ev_sources
                        yield ToolResultEvent(
                            tool=name, success=True,
                            output=result_text, sources=ev_sources,
                        )
                        add_step(f"📋 检索到 {len(ev_sources)} 条相关结果")
                    else:
                        yield ToolResultEvent(
                            tool=name, success=True,
                            output=result_text, sources=None,
                        )
                        add_step(f"📋 返回: {result_text[:120]}")

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

                    result_text = execute_tool(name, inp)

                    if name == "search_knowledge_base":
                        sources = _parse_sources(result_text)
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
