# AI Agent 项目集

三个递进式项目：从底层手写到框架应用，覆盖 RAG 知识库、Agent 工作流引擎、LangGraph ReAct Agent。

## 项目结构

```
AI_Agent_Project/
├── rag_chat/          # 项目一：RAG 知识库问答系统
├── langgraph_agent/   # 项目三：LangGraph ReAct Agent
└── README.md
```

> 项目二（Agent 工作流引擎）位于独立仓库：`AI 工作流 Agent —— 自然语言驱动的多工具编排系统`

## 项目一：RAG 知识库问答系统

`rag_chat/` — 基于 FastAPI + ChromaDB + DeepSeek 的检索增强生成问答后端。

| 能力 | 技术 |
|------|------|
| 文档解析 | TXT / Markdown / PDF |
| 智能分块 | 自研递归切分器（中文标点分隔 + 重叠窗口） |
| 向量检索 | ChromaDB + 余弦相似度 + distance_threshold 阈值 |
| 流式对话 | SSE（Server-Sent Events）流式推送 |
| 多轮记忆 | Redis LPUSH + EXPIRE 30 分钟过期 |
| MCP 工具 | 3 个工具暴露给外部 Agent 调用 |

详见 [rag_chat/README.md](rag_chat/README.md)

### 当前状态

**知识库**：全量爬虫已完成，1195 篇文档 / 13 分类。分块 `CHUNK_SIZE=300 / OVERLAP=50`（消融最优）。

**运行参数**（在 `.env`，改配置不用改代码）：
- `TEMPERATURE=0.3`、`MAX_TOKENS=1000`、`TOP_K=8`、`KB_SUBJECT_SCHOOL=河南工学院`
- `RETRIEVAL_MODE=hybrid`（向量 + BM25 RRF 融合）、`RETRIEVAL_RERANK=off`（重排默认关，遇表格行值题可临时开，代价每问 +1.9s）

**命中率成绩单**（评测脚本已清理删除，如需复测需重建）：

| 评测 | 命中率 |
|------|--------|
| 6 题基准（Qwen 60 分制） | **60/60** |
| 25 题随机 | **96%** |
| 50 题随机 | **100%** |
| 100 题随机（回归后） | **95%** |

> 详细评测过程与改造历史见 [会话备份.md](会话备份.md) 与 [rag_chat/docs/改进记录.md](rag_chat/docs/改进记录.md)。

## 项目二：Agent 工作流引擎

独立仓库 — 手写 Agent 循环，事件驱动架构，多工具编排。

| 能力 | 技术 |
|------|------|
| Agent 循环 | 手写 for-step 循环 + 条件路由 |
| 事件系统 | 6 种事件类型（Thinking / ToolCall / ToolResult / Text / Done / Error） |
| 流式输出 | SSE + Rich 终端显示（TTY 自适应） |
| 工具生态 | 7 个内置工具（时间/文件/RAG/搜索/天气） |
| 测试 | pytest 22 条（Agent 循环 + 工具单元 + Mock LLM） |

## 项目三：LangGraph ReAct Agent

`langgraph_agent/` — 基于 LangGraph 状态机的 ReAct Agent，对接项目一知识库。

| 能力 | 技术 |
|------|------|
| 状态管理 | StateGraph + 条件路由（should_continue） |
| LLM 协议 | Anthropic Tool Use 格式 |
| 工具 | 知识库检索 / 文档列表 / 安全数学计算 |
| 安全 | AST 白名单、MAX_ROUNDS=10 兜底 |

详见 [langgraph_agent/README.md](langgraph_agent/README.md)

## 三个项目的关系

```
项目一（RAG 知识库）
    │
    ├── MCP 协议 ──→ 项目二（手写 Agent）调用知识库
    │
    └── 直接导入 ──→ 项目三（LangGraph Agent）调用知识库

项目二 ── 框架升级 ──→ 项目三
（手写循环）              （状态机）
```

从底层手写到框架应用，覆盖同一个知识库系统。

## License

MIT
