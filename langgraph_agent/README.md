# LangGraph ReAct Agent

基于 LangGraph 状态机实现 ReAct（Reasoning + Acting）循环的 AI Agent，对接项目一的 ChromaDB 知识库。

## 核心架构

```
START → agent → [should_continue] → tools → agent → ... → END
              ├─ 有 tool_use → tools
              └─ 无 tool_use → END
```

| 节点 | 职责 |
|------|------|
| `agent_node` | 调用 LLM，决定直接回答还是调用工具 |
| `tool_node` | 执行 LLM 请求的工具，结果返回 agent 继续推理 |
| `should_continue` | 条件路由：有 tool_use → tools，没有 → 结束 |

## 工具列表

| 工具 | 说明 |
|------|------|
| `search_knowledge_base` | 向量语义检索，对接项目一 ChromaDB |
| `list_knowledge_sources` | 列出知识库已有文档及块数 |
| `calculate` | 安全数学表达式计算（AST 白名单） |

## 技术栈

| 层 | 技术 |
|---|---|
| 状态机 | LangGraph StateGraph + 条件路由 |
| LLM 协议 | Anthropic Messages API（Tool Use 格式） |
| HTTP | aiohttp 异步请求 |
| 向量库 | ChromaDB（复用项目一） |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |

## 快速开始

```bash
# 1. 进入目录
cd langgraph_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（优先读取 rag_chat/.env）
cp .env.example .env
# 编辑 .env，填入 API_KEY

# 4. 运行
python agent.py
```

## 配置（`.env`）

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_KEY` | ✅ | DeepSeek API 密钥 |
| `API_URL` | ❌ | Anthropic 兼容接口（默认 DeepSeek） |
| `MODEL_NAME` | ❌ | 模型名称（默认 deepseek-v4-pro） |

## 安全机制

- 数学计算使用 AST 白名单解析，拒绝任意代码执行
- 最大推理轮数 `MAX_ROUNDS = 10` 防止死循环
- 工具执行异常统一捕获并返回错误信息

## 与项目一的关系

```
项目一（RAG 知识库）         项目三（LangGraph Agent）
┌──────────────────┐         ┌──────────────────────┐
│  ChromaDB        │◀────────│  search_knowledge_base │
│  VectorStore     │   检索   │  list_knowledge_sources│
│  MCP Server      │         └──────────────────────┘
└──────────────────┘
```

Agent 直接导入项目一的 `VectorStore`，指向同一个 `rag_chat/chroma_db` 数据库。

## 项目结构

```
langgraph_agent/
├── agent.py          # LangGraph 状态机 + ReAct 循环 + CLI
├── tools.py          # 工具定义（Anthropic Tool Use 格式）+ 执行函数
├── requirements.txt  # 项目依赖
└── .env.example      # 环境变量模板
```

## License

MIT
