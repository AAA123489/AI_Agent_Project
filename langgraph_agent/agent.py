"""
LangGraph ReAct Agent —— 项目三
=================================
基于 LangGraph 状态机实现 ReAct (Reasoning + Acting) 循环的 AI Agent。

核心架构：
    ┌──────────┐  tool_use?   ┌──────────┐
    │  agent   │ ───────────→ │   tools  │
    │ (LLM推理) │ ←─────────── │ (工具执行) │
    └──────────┘  tool_result  └──────────┘
         │
         │ end_turn (不再需要工具)
         ▼
     最终答案输出给用户

agent 节点：调用 LLM，模型决定"直接回答"还是"调用工具"。
           如果返回 tool_use → 路由到 tools 节点
           如果返回 text     → 路由到 END，输出答案

tools 节点：执行工具（search_knowledge_base / list_sources / calculate），
           把结果包装成 tool_result 消息，路由回 agent 节点继续推理。

技术栈：
    - LangGraph StateGraph（状态机 + 条件路由）
    - DeepSeek API（Anthropic Messages 兼容格式，支持 Tool Use）
    - aiohttp 异步 HTTP
    - 对接项目一 rag_chat 的 VectorStore

运行方式：
    cd langgraph_agent
    python agent.py

面试要点（看完代码后，你应该能回答）：
    1. ReAct 循环的四个阶段：Thought → Action → Observation → Thought...
    2. LangGraph 节点和条件边的设计理由
    3. 为什么手搓 agent 循环 vs 用 LangGraph 预置的 create_react_agent
    4. Anthropic Tool Use 协议 vs OpenAI Function Calling 的区别
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

from tools import TOOL_DEFINITIONS, execute_tool  # noqa: E402

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
# System Prompt —— 定义 Agent 的行为边界和工具使用策略
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个智能 AI Agent，能够自主决定何时调用工具、调用哪个工具。\n"
    "\n"
    "## 工作流程\n"
    "1. 理解用户的问题\n"
    "2. 判断是否需要工具：\n"
    "   - 需要查找技术知识 → 调用 search_knowledge_base\n"
    "   - 需要了解知识库内容 → 调用 list_knowledge_sources\n"
    "   - 需要数学计算 → 调用 calculate\n"
    "   - 普通对话/已有足够信息 → 直接回答\n"
    "3. 如果需要工具：调用工具 → 根据结果决定是否需要更多工具\n"
    "4. 给出最终答案\n"
    "\n"
    "## 重要规则\n"
    "- 工具返回的内容是知识库中的权威信息，优先参考\n"
    "- 如果知识库没有找到相关信息，诚实告知用户，并基于你的知识回答（注明来源）\n"
    "- 每次调用工具后，综合所有已获取的信息判断是否还需要更多工具\n"
    "- 得到足够信息后，立即给出最终答案，不要无意义地反复调用工具\n"
    "- 最终答案中引用知识库内容时，标注来源文件名\n"
    "- 用中文回复用户"
)


# ═══════════════════════════════════════════════════════════════
# Agent 状态定义
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """LangGraph 状态：messages 使用 operator.add 累加，round_count 每轮替换。

    messages 格式：[{role, content}, ...]
      其中 content 可以是：
      - 纯文本字符串（用户消息/简单回复）
      - Anthropic content blocks 列表：[{type, text/tool_use/tool_result}, ...]
    """
    messages: Annotated[list[dict], operator.add]
    round_count: int


# ═══════════════════════════════════════════════════════════════
# LLM 调用（Anthropic-Compatible API，支持 Tool Use）
# ═══════════════════════════════════════════════════════════════

async def call_llm(
    messages: list[dict],
    api_key: str = API_KEY,
    api_url: str = API_URL,
    model: str = MODEL_NAME,
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

    payload = {
        "model": model,
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
    }

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
# LangGraph 节点
# ═══════════════════════════════════════════════════════════════

async def agent_node(state: AgentState) -> dict:
    """agent 节点：调用 LLM，让模型决定下一步行动。

    输入：当前消息历史（包含用户问题 + 之前的工具调用结果）
    输出：LLM 的回复消息（可能是纯文本，也可能包含 tool_use 指令）
    """
    logger.info("── agent 节点：LLM 推理中... (第 %d 轮) ──", state.get("round_count", 0) + 1)

    response = await call_llm(state["messages"])

    # 将 LLM 的回复包装为 Anthropic 格式的 assistant 消息
    assistant_msg = {
        "role": "assistant",
        "content": response["content"],
    }

    return {
        "messages": [assistant_msg],
        "round_count": state.get("round_count", 0) + 1,
    }


async def tool_node(state: AgentState) -> dict:
    """tools 节点：执行 LLM 请求的工具调用。

    从最新一条 assistant 消息中提取 tool_use 块，
    逐个执行，把结果包装为 tool_result 消息返回给 LLM。
    """
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

        print(f"\n  🔧 调用工具: {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")

        result_text = execute_tool(tool_name, tool_input)

        # 打印工具结果摘要
        preview = result_text[:150].replace("\n", " ") + ("..." if len(result_text) > 150 else "")
        print(f"  📋 结果: {preview}")

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


# ═══════════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """判断下一步：继续调工具，还是结束？

    检查最后一条 assistant 消息：
    - 包含 tool_use → 路由到 "tools" 节点
    - 没有 tool_use → 路由到 END（输出最终答案）
    - 超过最大轮数 → 强制结束
    """
    # 安全上限
    if state.get("round_count", 0) >= MAX_ROUNDS:
        logger.warning("达到最大轮数 %d，强制结束", MAX_ROUNDS)
        return END

    # 找最后一条 assistant 消息
    for msg in reversed(state["messages"]):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return "tools"
        # 有 assistant 消息但没有 tool_use → 纯文本回复，结束
        return END

    # 没有 assistant 消息（首次进入）→ 不应该出现，结束
    return END


# ═══════════════════════════════════════════════════════════════
# 构建 LangGraph 状态图
# ═══════════════════════════════════════════════════════════════

def build_agent() -> StateGraph:
    """构建并编译 ReAct Agent 状态图。

    图结构：
        START → agent → [条件判断]
                          ├─ tool_use → tools → agent (循环)
                          └─ text     → END
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # 入口
    graph.set_entry_point("agent")

    # 条件边：agent 之后根据 stop_reason 决定去向
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # tools 之后总是回到 agent（让 LLM 判断是否还需要更多工具）
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info("Agent 状态图构建完成")
    return compiled


# ═══════════════════════════════════════════════════════════════
# 工具函数：从 Agent 状态中提取最终答案
# ═══════════════════════════════════════════════════════════════

def extract_final_answer(state: AgentState) -> str:
    """从最终的 Agent 状态中提取文本答案。

    遍历所有 assistant 消息中 type=text 的内容块，拼接输出。
    """
    answers = []
    for msg in state.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                answers.append(block.get("text", ""))
    return "\n".join(answers) if answers else "（Agent 未生成回复）"


# ═══════════════════════════════════════════════════════════════
# 交互式 CLI
# ═══════════════════════════════════════════════════════════════

async def main():
    """交互式对话入口。"""

    # 检查配置
    if not API_KEY:
        print("❌ 请设置 API_KEY 环境变量")
        print("   复制 rag_chat/.env 到当前目录，或设置环境变量")
        return

    print("=" * 60)
    print("  🤖 LangGraph ReAct Agent")
    print("  技术栈: LangGraph + DeepSeek API + Chroma 知识库")
    print("  工具: search_knowledge_base | list_knowledge_sources | calculate")
    print("=" * 60)
    print("  输入 'quit' 退出 | 输入 'tools' 查看可用工具")
    print()

    agent = build_agent()

    # 对话历史（整个会话保持，每次提问追加）
    conversation_messages: list[dict] = []

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("👋 再见！")
            break

        if user_input.lower() == "tools":
            print("\n📦 可用工具:")
            for t in TOOL_DEFINITIONS:
                print(f"  • {t['name']}: {t['description'][:80]}...")
            print()
            continue

        # 把用户消息加入对话历史
        user_msg = {"role": "user", "content": user_input}
        conversation_messages.append(user_msg)

        print("\n⏳ Agent 思考中...\n")

        # 运行 Agent（传入当前所有对话消息）
        initial_state: AgentState = {
            "messages": list(conversation_messages),  # 复制列表
            "round_count": 0,
        }

        try:
            final_state = await agent.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Agent 执行异常")
            print(f"\n❌ Agent 出错: {e}")
            continue

        # 提取最终答案
        answer = extract_final_answer(final_state)

        print(f"\n🤖 Agent: {answer}\n")
        print("-" * 60)

        # 把最终答案也加入对话历史（供下一轮多轮对话使用）
        # 找到最后一条 assistant 消息加入历史
        for msg in reversed(final_state.get("messages", [])):
            if msg.get("role") == "assistant":
                conversation_messages.append(msg)
                break


if __name__ == "__main__":
    asyncio.run(main())
