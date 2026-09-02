"""
LangGraph 多 Agent 编排引擎 —— 项目三
=====================================
基于 LangGraph 状态机实现「Router 意图路由 + 双子代理」的多 Agent 架构。

核心架构：
    ┌────────────┐  route   ┌────────────────────┐
    │   router   │─────────→│   kb_agent 子代理   │  (知识库检索)
    │ (LLM 意图分类) │──→    ├────────────────────┤
    └────────────┘  route   │   calc_agent 子代理 │  (安全数学计算)
                │           ├────────────────────┤
                └──────────→│  chat_agent 子代理  │  (普通对话)
                            └────────────────────┘

每个子代理是独立的 ReAct 子图（StateGraph + 条件路由 + 各自工具集）：
    START → agent → [should_continue] → tools → agent → ... → END
              ├─ 有 tool_use → tools
              └─ 无 tool_use → END

Router 节点：用轻量 LLM 调用判断用户意图，把请求分派给对应子代理。

技术栈：
    - LangGraph StateGraph（父图 + 子图嵌套，条件路由）
    - DeepSeek API（Anthropic Messages 兼容格式，支持 Tool Use）
    - aiohttp 异步 HTTP
    - 对接项目一 rag_chat 的 VectorStore

运行方式：
    cd langgraph_agent
    python agent.py

"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import aiohttp
import operator
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# ── 路径设置：让 langgraph_agent 能导入同目录的 tools 模块 ──
_agent_dir = Path(__file__).resolve().parent
if str(_agent_dir) not in sys.path:
    sys.path.insert(0, str(_agent_dir))

from tools import TOOL_DEFINITIONS, execute_tool, set_session  # noqa: E402

# ── 加载环境变量（优先读 rag_chat/.env，其次读当前目录 .env）──
_rag_chat_env = Path(__file__).resolve().parent.parent / "rag_chat" / ".env"
if _rag_chat_env.exists():
    load_dotenv(_rag_chat_env)
else:
    load_dotenv()

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Agent] %(levelname)s %(message)s",
)
logger = logging.getLogger("langgraph_agent")

# ── 配置 ──
API_KEY = os.getenv("API_KEY", "")
API_URL = os.getenv("API_URL", "https://api.deepseek.com/anthropic/v1/messages")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-pro")
MAX_ROUNDS = 10  # 安全上限：最多 10 轮推理，防止死循环

# ═══════════════════════════════════════════════════════════════
# System Prompt —— 每个子代理 + Router 各自的行为边界
# ═══════════════════════════════════════════════════════════════

ROUTER_PROMPT = (
    "你是任务路由器。只判断用户问题应该交给哪个子代理处理，不做任何回答。\n"
    "只输出一个词，不要任何解释或标点：\n"
    '- "kb"：需要查询知识库/学校文档/通知政策的问题\n'
    '- "calc"：需要数学计算的问题\n'
    '- "chat"：普通闲聊，不需要任何工具\n'
    "\n"
    "示例：\n"
    '  "2026年学费减免政策" → kb\n'
    '  "(100+200)*3 等于多少" → calc\n'
    '  "你好，你是谁" → chat\n'
)

KB_PROMPT = (
    "你是一个知识库问答子代理，负责回答学校相关信息。\n"
    "\n"
    "## 工作流程\n"
    "1. 需要查询知识库 → 调用 search_knowledge_base\n"
    "2. 需要了解知识库内容 → 调用 list_knowledge_sources\n"
    "3. 得到信息后给出最终答案\n"
    "\n"
    "## 重要规则\n"
    "- 工具返回的内容是知识库中的权威信息，优先参考\n"
    "- 如果知识库没有找到相关信息，诚实告知用户\n"
    "- 最终答案中引用知识库内容时，标注来源文件名\n"
    "- 用户透露的个人信息或偏好（姓名、年级、所在校区等）→ 调用 save_memory 主动记住\n"
    "- 回答前如有相关记忆先调用 search_memory 查一下（如问题涉及用户之前提过的信息）\n"
    "- 用户明确要求忘记某信息 → 调用 clear_memory\n"
    "- 用中文回复用户"
)

CALC_PROMPT = (
    "你是一个安全计算子代理，只负责数学计算。\n"
    "\n"
    "## 工作流程\n"
    "1. 需要计算 → 调用 calculate 工具，把数学表达式作为参数\n"
    "2. 得到结果后给出最终答案\n"
    "\n"
    "## 重要规则\n"
    "- 只做纯数学计算，不接受变量或函数调用\n"
    "- 计算得到结果后直接回答，不要调用其他工具\n"
    "- 用中文回复用户"
)

CHAT_PROMPT = (
    "你是一个友好的对话子代理。\n"
    "\n"
    "## 重要规则\n"
    "- 只使用工作记忆工具（save_memory / search_memory / clear_memory），不调用其他工具\n"
    "- 用户透露的个人偏好、重要信息 → 主动 save_memory 记住\n"
    "- 回答前先 search_memory 查一下是否有该用户的相关记忆\n"
    "- 用户明确要求忘记某信息 → clear_memory\n"
    "- 用中文回复用户"
)

# ═══════════════════════════════════════════════════════════════
# 工具集划分 —— 每个子代理只用自己那部分工具
# ═══════════════════════════════════════════════════════════════

KB_TOOL_NAMES = {"search_knowledge_base", "list_knowledge_sources"}
CALC_TOOL_NAMES = {"calculate"}
MEMORY_TOOL_NAMES = {"save_memory", "search_memory", "clear_memory"}

# kb 子代理：知识库检索 + 列表 + 工作记忆
KB_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in KB_TOOL_NAMES | MEMORY_TOOL_NAMES]
CALC_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in CALC_TOOL_NAMES]
# chat 子代理：只有工作记忆工具（记住/回忆用户偏好）
CHAT_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in MEMORY_TOOL_NAMES]

# ═══════════════════════════════════════════════════════════════
# Agent 状态定义
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """LangGraph 状态：messages 使用 operator.add 累加，round_count / route 每轮替换。

    messages 格式：[{role, content}, ...]
      其中 content 可以是：
      - 纯文本字符串（用户消息/简单回复）
      - Anthropic content blocks 列表：[{type, text/tool_use/tool_result}, ...]

    route: Router 判断出的子代理类别（kb / calc / chat）
    """
    messages: Annotated[list[dict], operator.add]
    round_count: int
    route: str

# ═══════════════════════════════════════════════════════════════
# LLM 调用（Anthropic-Compatible API，支持 Tool Use）
# ═══════════════════════════════════════════════════════════════

async def call_llm(
    messages: list[dict],
    api_key: str = API_KEY,
    api_url: str = API_URL,
    model: str = MODEL_NAME,
    system: str | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """调用 LLM，返回包含 content_blocks 和 stop_reason 的 dict。

    Anthropic Messages API 请求格式：
        {model, max_tokens, system, messages, tools}

    响应格式：
        {content: [{type: "text", text: "..."} / {type: "tool_use", ...}],
         stop_reason: "end_turn" / "tool_use"}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": model,
        "max_tokens": 1000,
        "system": system or "",
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools

    logger.info("LLM 请求: %d 条历史消息", len(messages))

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error("LLM API 错误 (%d): %s", resp.status, error_text[:200])
                return {
                    "content": [{"type": "text", "text": f"API 调用失败: {resp.status}"}],
                    "stop_reason": "end_turn",
                }

            data = await resp.json()

    # 提取文本和工具调用块
    content_blocks = data.get("content", [])
    text_parts = []
    tool_uses = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_uses.append(block)

    stop_reason = data.get("stop_reason", "end_turn")

    if tool_uses:
        logger.info(
            "LLM 响应: 想调用 %d 个工具 → %s",
            len(tool_uses),
            [tu["name"] for tu in tool_uses],
        )
    else:
        logger.info("LLM 响应: 直接回答 (stop_reason=%s)", stop_reason)

    return {
        "content": content_blocks,  # 原始 content blocks（保留 tool_use 的 id 等字段）
        "text": "\n".join(text_parts) if text_parts else "",
        "tool_uses": tool_uses,
        "stop_reason": stop_reason,
    }

# ═══════════════════════════════════════════════════════════════
# Router 节点 —— 意图分类 + 分派
# ═══════════════════════════════════════════════════════════════

def _parse_route(text: str) -> str:
    """从 Router 的 LLM 回复中提取子代理类别。"""
    t = (text or "").strip().lower()
    if "calc" in t:
        return "calc"
    if "kb" in t:
        return "kb"
    return "chat"  # 兜底：识别不出就闲聊子代理，不拒绝用户


async def router_node(state: AgentState) -> dict:
    """Router 节点：用轻量 LLM 调用判断用户意图，返回 route 字段。

    输入：用户消息历史
    输出：{"route": "kb" | "calc" | "chat"}
    """
    logger.info("── Router 节点：意图分类中... ──")

    # 取最后一条用户消息
    user_query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            break

    response = await call_llm(
        [{"role": "user", "content": user_query}],
        system=ROUTER_PROMPT,
        tools=None,  # Router 不调用工具，只做分类
    )

    route = _parse_route(response["text"])
    logger.info("Router 判定: %s → %s", user_query[:30], route)
    print(f"[Router] 分派: [{route}] -> {user_query[:30]}")

    return {"route": route}


# ═══════════════════════════════════════════════════════════════
# 子代理节点（工厂函数 —— 每个子代理复用同一套 ReAct 逻辑）
# ═══════════════════════════════════════════════════════════════

def make_agent_node(system_prompt: str, tools: list[dict]):
    """生成子代理的 agent 节点：调用 LLM，让模型决定下一步行动。"""
    async def agent_node(state: AgentState) -> dict:
        logger.info("── %s agent 节点：LLM 推理中... (第 %d 轮) ──",
                    system_prompt[:6], state.get("round_count", 0) + 1)

        response = await call_llm(
            state["messages"],
            system=system_prompt,
            tools=tools,
        )

        assistant_msg = {
            "role": "assistant",
            "content": response["content"],
        }

        return {
            "messages": [assistant_msg],
            "round_count": state.get("round_count", 0) + 1,
        }
    return agent_node


def make_tool_node():
    """生成子代理的 tools 节点：执行 LLM 请求的工具调用。"""
    async def tool_node(state: AgentState) -> dict:
        # 找到最后一条 assistant 消息中的 tool_use 块
        last_assistant_msg = None
        for msg in reversed(state["messages"]):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg
                break

        if not last_assistant_msg:
            logger.warning("tools 节点：找不到 assistant 消息")
            return {"messages": [], "round_count": state["round_count"]}

        content = last_assistant_msg.get("content", [])
        tool_uses = [
            block for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]

        if not tool_uses:
            logger.warning("tools 节点：assistant 消息中没有 tool_use 块")
            return {"messages": [], "round_count": state["round_count"]}

        # 逐个执行工具
        tool_result_blocks = []
        for tu in tool_uses:
            tool_name = tu["name"]
            tool_id = tu["id"]
            tool_input = tu.get("input", {})

            print(f"\n  [工具] 调用: {_safe_gbk(tool_name)}({_safe_gbk(json.dumps(tool_input, ensure_ascii=False))})")

            result_text = execute_tool(tool_name, tool_input)

            # 打印工具结果摘要
            preview = result_text[:150].replace("\n", " ") + ("..." if len(result_text) > 150 else "")
            print(f"  [结果] {_safe_gbk(preview)}")

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_text,
            })

        logger.info("── tools 节点：%d 个工具执行完成 ──", len(tool_result_blocks))

        # 工具结果以 user 角色发回（Anthropic 协议要求）
        return {
            "messages": [{"role": "user", "content": tool_result_blocks}],
            "round_count": state["round_count"],
        }
    return tool_node


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """判断下一步：继续调工具，还是结束？"""
    if state.get("round_count", 0) >= MAX_ROUNDS:
        logger.warning("达到最大轮数 %d，强制结束", MAX_ROUNDS)
        return END

    for msg in reversed(state["messages"]):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return "tools"
        # 有 assistant 消息但没有 tool_use → 纯文本回复，结束
        return END

    return END


# ═══════════════════════════════════════════════════════════════
# 构建子代理子图 + 顶层 Router 父图
# ═══════════════════════════════════════════════════════════════

def build_sub_agent(name: str, system_prompt: str, tools: list[dict]):
    """构建一个独立的 ReAct 子代理子图，可作为父图节点嵌套。

    图结构（子代理内部）：
        START → agent → [条件判断]
                          ├─ tool_use → tools → agent (循环)
                          └─ text     → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", make_agent_node(system_prompt, tools))
    graph.add_node("tools", make_tool_node())

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info("子代理 [%s] 构建完成（%d 个工具）", name, len(tools))
    return compiled


def build_agent() -> StateGraph:
    """构建并编译多 Agent 状态图。

    图结构：
        START → router → [条件路由]
                           ├─ kb   → kb_agent 子代理
                           ├─ calc → calc_agent 子代理
                           └─ chat → chat_agent 子代理
    """
    graph = StateGraph(AgentState)

    # Router 节点
    graph.add_node("router", router_node)

    # 三个子代理（各自独立的 ReAct 子图）
    graph.add_node("kb_agent", build_sub_agent("kb", KB_PROMPT, KB_TOOLS))
    graph.add_node("calc_agent", build_sub_agent("calc", CALC_PROMPT, CALC_TOOLS))
    graph.add_node("chat_agent", build_sub_agent("chat", CHAT_PROMPT, CHAT_TOOLS))

    # 入口：Router
    graph.set_entry_point("router")

    # Router 之后按 route 条件分派
    graph.add_conditional_edges(
        "router",
        lambda state: state.get("route", "chat") + "_agent",
        {
            "kb_agent": "kb_agent",
            "calc_agent": "calc_agent",
            "chat_agent": "chat_agent",
        },
    )

    compiled = graph.compile()
    logger.info("多 Agent 状态图构建完成")
    return compiled

# ═══════════════════════════════════════════════════════════════
# 工具函数：从 Agent 状态中提取最终答案
# ═══════════════════════════════════════════════════════════════

def _safe_gbk(text: str) -> str:
    """把文本里 GBK 编不了的字符（emoji 等）替换为 ?，防止 Windows 终端崩溃。

    Windows 默认控制台是 GBK 编码，LLM 返回的 emoji（等）直接 print 会
    UnicodeEncodeError。此函数确保任何文本都能安全打印。
    """
    try:
        text.encode("gbk")
        return text
    except UnicodeEncodeError:
        return "".join(c if _can_gbk(c) else "?" for c in text)


def _can_gbk(ch: str) -> bool:
    try:
        ch.encode("gbk")
        return True
    except UnicodeEncodeError:
        return False


def extract_final_answer(state: AgentState) -> str:
    """从最终的 Agent 状态中提取文本答案。

    只取【最后一条】assistant 消息的 text 块——那才是最终答案；
    中间轮次的 text 是 Agent 的思考过程（如"让我搜索…"），不应泄漏到答案里。
    """
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        blocks = msg.get("content", [])
        texts = [
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    return "（Agent 未生成回复）"

# ═══════════════════════════════════════════════════════════════
# 交互式 CLI
# ═══════════════════════════════════════════════════════════════

async def main():
    """交互式对话入口。"""

    # 检查配置
    if not API_KEY:
        print("[X]  请设置 API_KEY 环境变量")
        print("   复制 rag_chat/.env 到当前目录，或设置环境变量")
        return

    print("=" * 60)
    print("  [Bot]  LangGraph 多 Agent 编排引擎")
    print("  架构: Router 意图路由 → 三子代理 (知识库 / 安全计算 / 闲聊)")
    print("  能力: 主动工作记忆（save_memory / search_memory / clear_memory）")
    print("  技术栈: LangGraph + DeepSeek API + Chroma 知识库 + Redis 记忆")
    print("=" * 60)
    print("  输入 'quit' 退出 | 输入 'tools' 查看可用工具")
    print()

    agent = build_agent()

    # 工作记忆：CLI 固定会话 "cli"，记忆跨重启仍可检索（TTL 内）
    set_session("cli")

    # 对话历史（整个会话保持，每次提问追加）
    conversation_messages: list[dict] = []

    while True:
        try:
            user_input = input("[You]  你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye]  再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("[Bye]  再见！")
            break

        if user_input.lower() == "tools":
            print("\n[Tools]  可用工具:")
            for t in TOOL_DEFINITIONS:
                print(f"  • {t['name']}: {t['description'][:80]}...")
            print()
            continue

        # 把用户消息加入对话历史
        user_msg = {"role": "user", "content": user_input}
        conversation_messages.append(user_msg)

        print("\n[Wait]  Agent 思考中...\n")

        # 运行 Agent（传入当前所有对话消息）
        initial_state: AgentState = {
            "messages": list(conversation_messages),  # 复制列表
            "round_count": 0,
            "route": "",
        }

        try:
            final_state = await agent.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Agent 执行异常")
            print(f"\n[X]  Agent 出错: {e}")
            continue

        # 提取最终答案
        answer = extract_final_answer(final_state)

        print(f"\n[Bot]  Agent: {_safe_gbk(answer)}\n")
        print("-" * 60)

        # 把最终答案也加入对话历史（供下一轮多轮对话使用）
        for msg in reversed(final_state.get("messages", [])):
            if msg.get("role") == "assistant":
                conversation_messages.append(msg)
                break

if __name__ == "__main__":
    asyncio.run(main())
