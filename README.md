# AI Agent 项目集

三个递进式的 AI Agent 实践项目：从**底层手写 Agent 循环** → **RAG 知识库问答** → **LangGraph 多 Agent 编排**，覆盖 Agent 开发的完整能力链路。

```
项目一（RAG 知识库）
    │
    ├── MCP 协议 ──→ 项目二（手写 Agent）调用知识库
    │
    └── 直接导入 ──→ 项目三（LangGraph 多 Agent）调用知识库

项目二 ── 框架升级 ──→ 项目三
（手写循环）              （状态机）
```

## 项目总览

| 项目 | 定位 | 技术栈 | 状态 |
|------|------|--------|------|
| **项目一** | RAG 知识库问答系统 | FastAPI · ChromaDB · DeepSeek · Redis · SSE | ✅ 完成 |
| **项目二** | 手写 Agent 工作流引擎（事件驱动） | Python · asyncio · SSE · MCP | ✅ 完成（独立仓库） |
| **项目三** | LangGraph 多 Agent 编排引擎 | LangGraph · DeepSeek · ChromaDB · Redis | ✅ 完成 |

## 项目结构

```
AI_Agent_Project/
├── rag_chat/          # 项目一：RAG 知识库问答系统
├── langgraph_agent/   # 项目三：LangGraph 多 Agent 编排引擎
└── README.md
```

## 项目一：RAG 知识库问答系统

基于 **FastAPI + ChromaDB + DeepSeek** 的检索增强生成（RAG）问答后端，对校园公开文档做知识库问答。

**核心能力**
- 文档解析（TXT / Markdown / PDF）+ 自研递归切分器（中文标点分隔 + 重叠窗口，消融实验验证最优块大小）
- **混合检索**：向量 + BM25 双路召回 + RRF 融合，命中率 95%+（25 题 96% / 50 题 100% / 100 题 95%）
- 长尾召回优化：2/3 字 n-gram + 精确匹配兜底（解决中文人名切分不准的漏召回）
- SSE 流式对话 + Redis 多轮记忆 + 查询级缓存 + 库外拒答 + 同文档补块
- MCP Server 暴露 3 个工具，供外部 Agent 调用
- pytest 57 条用例

**技术栈**：FastAPI · ChromaDB · DeepSeek · Redis · SSE · aiohttp

📖 详见 [rag_chat/README.md](rag_chat/README.md)

## 项目二：Agent 工作流引擎（独立仓库）

为理解 Agent 底层机制，**不依赖 LangChain 等框架**手写的 Agent 循环（事件驱动架构）。

**核心能力**
- 手写 for-step 事件循环 + 条件路由，将「思考-行动-观察」完全解耦
- 6 种事件类型（Thinking / ToolCall / ToolResult / Text / Done / Error）
- 7 个内置工具（时间 / 文件 / RAG / 搜索 / 天气），通过 MCP 协议暴露
- SSE + Rich 双端流式输出（TTY 自适应）
- pytest 22 条用例

> 该项目位于独立仓库，未包含在本仓库内。架构设计由 [项目三](langgraph_agent/README.md) 以 LangGraph 状态机形式框架化升级。

## 项目三：LangGraph 多 Agent 编排引擎

基于 **LangGraph 状态机**实现「**Router 意图路由 + 三子代理**」多 Agent 架构，对接项目一知识库。

**核心能力**
- **Router 意图路由**：LLM 分类 `kb / calc / chat`，识别不出走 chat 兜底，不拒绝用户
- **三子代理**：kb（知识库检索）/ calc（安全计算）/ chat（闲聊），各为独立 ReAct 子图，工具集按角色隔离
- **主动工作记忆**：Agent 通过 `save_memory / search_memory / clear_memory` 自己决定存什么、查什么（Redis，按会话隔离）
- **Web 界面**：FastAPI + SSE 节点级流式，前端实时可见「Router 分派 → 子代理推理 → 工具调用」全过程
- 安全：AST 白名单计算、`MAX_ROUNDS=10` 防死循环、X-API-Key 鉴权
- pytest 36 条用例（mock LLM 全链路）

**技术栈**：LangGraph · DeepSeek · ChromaDB · Redis · FastAPI · aiohttp

📖 详见 [langgraph_agent/README.md](langgraph_agent/README.md)

## 快速开始

```bash
# 项目一：RAG 问答
cd rag_chat && python app_fastapi.py    # http://localhost:8000

# 项目三：多 Agent（CLI）
cd langgraph_agent && python agent.py

# 项目三：多 Agent（Web，复用项目一前端）
cd langgraph_agent && python app_web.py  # http://localhost:8001

# 运行测试
cd rag_chat && python -m pytest tests/ -v
cd langgraph_agent && python -m pytest tests/ -v
```

> 环境变量参考各子目录 `.env.example`（需自行配置 `API_KEY`；Redis 可选）。

## License

MIT
