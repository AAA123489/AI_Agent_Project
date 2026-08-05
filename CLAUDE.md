# AI Agent Project - 项目文档

## 项目概况

AI Agent 实习生求职项目，包含三个递进式项目，证明从底层到框架的完整能力。

## 三个项目

### 项目一：手搓 RAG 知识库系统
- 位置：`rag_chat/`
- 状态：✅ **已全部吃透**（15 个模块逐行理解 + 20 道面试问答）
- 核心：自研递归切分器 + ChromaDB + DeepSeek SSE 流式 + MCP Server
- 面试文档：桌面 `RAG项目面试问答集.docx`

### 项目二：Agent 工作流引擎
- 位置：`../AI 工作流 Agent —— 自然语言驱动的多工具编排系统/`
- 状态：🔄 **待开**
- 核心：手搓 Agent 循环（推理→工具调用→推理），多工具编排
- 阅读顺序：config.py → tools/ → prompts.py → agent_loop.py → rich_display.py → main.py

### 项目三：LangGraph ReAct Agent
- 位置：`langgraph_agent/`
- 状态：⏳ **待开**
- 核心：用 LangGraph 框架实现 ReAct Agent，对接项目一知识库

## 面试叙事线

"先手搓搞懂底层 → 再手搓 Agent 循环 → 最后用框架提效"

三个项目不是三件独立的事，是一条能力递进线。

## 当前进度

- [x] 项目一吃透（模块理解 + 面试问答 + Word 文档导出）
- [ ] 项目二吃透
- [ ] 项目三吃透
- [ ] 三个项目联动面试练习

## 记忆文件

- 上次完整对话复盘：`2026-07-29-session-summary.md`
- 面试问答集：桌面 `RAG项目面试问答集.docx`
