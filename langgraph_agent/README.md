# LangGraph 多 Agent 编排引擎

基于 LangGraph 状态机实现的**「Router 意图路由 + 双子代理」**多 Agent 架构，对接项目一的 ChromaDB 知识库。

## 核心架构

```
                  ┌────────────────────┐
  START ──► router  │   router 节点      │
  用户问题 │ (LLM 意图分类) │   ── kb/calc/chat
          └────────────────────┘
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ kb_agent │ │calc_agent│ │chat_agent│
  │ 知识库检索 │ │ 安全计算   │ │ 普通对话  │
  └──────────┘ └──────────┘ └──────────┘
```

- **Router 节点**（`router_node`）：用轻量 LLM 调用判断用户意图 → `kb` / `calc` / `chat`，把请求分派给对应子代理
- **三子代理**：每个子代理是**独立的 ReAct 子图**（StateGraph + 条件路由 + 各自限定的工具集），可单独演进
  - `kb_agent`：知识库检索（工具：search_knowledge_base / list_knowledge_sources）+ 工作记忆
  - `calc_agent`：安全数学计算（工具：calculate）
  - `chat_agent`：普通闲聊 + 工作记忆（save_memory / search_memory / clear_memory）

每个子代理内部：
```
START → agent → [should_continue] → tools → agent → ... → END
              ├─ 有 tool_use → tools
              └─ 无 tool_use → END
```

## 关键实现

| 组件 | 说明 |
|------|------|
| `router_node` | LLM 意图分类，返回 route 字段 |
| `_parse_route` | 解析 Router 输出，含噪声兜底（识别不出走 chat） |
| `build_sub_agent` | 子代理子图工厂（复用同一套 ReAct 逻辑） |
| `build_agent` | 父图：START→router→条件路由→子代理嵌套 |
| `extract_final_answer` | 只取最后一条 assistant 消息，防思考过程泄漏 |
| `_safe_gbk` | 打印时过滤 GBK 不可编码字符（Windows 终端防崩） |
| `save_memory` / `search_memory` / `clear_memory` | 主动工作记忆：Agent 自己决定存什么、查什么（Redis） |

## 工具列表

| 工具 | 说明 | 归属子代理 |
|------|------|-----------|
| `search_knowledge_base` | 向量语义检索，对接项目一 ChromaDB | kb |
| `list_knowledge_sources` | 列出知识库已有文档及块数 | kb |
| `calculate` | 安全数学表达式计算（AST 白名单） | calc |
| `save_memory` | 主动保存一条工作记忆（如用户偏好） | kb / chat |
| `search_memory` | 检索之前保存的工作记忆 | kb / chat |
| `clear_memory` | 删除记忆（'*' 清空全部） | kb / chat |

## 主动工作记忆

Agent 不依赖外部代码「被动喂历史」，而是通过工具**自己决定**记住什么、回答前查什么：

- 用户透露偏好（姓名、校区、年级…）→ Agent 主动 `save_memory`
- 回答涉及用户之前提过的信息 → 先 `search_memory`
- 用户要求忘记 → `clear_memory`

存储走 Redis（复用项目一），按会话隔离（Web 下并发用户互不串扰），默认保留 7 天（`.env` 配 `MEMORY_TTL` 可调）。Redis 挂掉时记忆工具降级返回提示，不影响 Agent 主流程。

> 设计取舍：短期工作记忆放 Redis 够用；长期/语义记忆才需要向量库（ChromaDB）——那是另一层，当前不混用。

## 技术栈

| 层 | 技术 |
|---|---|
| 状态机 | LangGraph StateGraph（父图 + 子图嵌套，条件路由） |
| LLM 协议 | Anthropic Messages API（Tool Use 格式） |
| HTTP | aiohttp 异步请求（LLM 调用）/ FastAPI + SSE（Web） |
| 向量库 | ChromaDB（复用项目一） |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| 记忆 | Redis（主动工作记忆 + 多轮对话持久化，复用项目一） |
| 测试 | pytest（36 条，mock LLM 全链路） |

## 快速开始

```bash
# 1. 进入目录
cd langgraph_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（优先读取 rag_chat/.env，含 REDIS_URL）
cp .env.example .env
# 编辑 .env，填入 API_KEY

# 4a. 命令行交互
python agent.py

# 4b. Web 聊天（复用项目一前端 chat.html，SSE 节点级流式）
python app_web.py        # → http://localhost:8001
# 前端填写：X-API-Key（= rag_chat/.env 的 API_KEY）+ user_id（随便填）

# 5. 跑测试
python -m pytest tests/ -v
```

## 配置（`.env`，优先读 rag_chat/.env）

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_KEY` | ✅ | DeepSeek API 密钥 |
| `API_URL` | ❌ | Anthropic 兼容接口（默认 DeepSeek） |
| `MODEL_NAME` | ❌ | 模型名称（默认 deepseek-v4-pro） |
| `REDIS_URL` | ❌ | 记忆 / 对话持久化（默认 redis://localhost:6379） |
| `MEMORY_TTL` | ❌ | 工作记忆保留秒数（默认 7 天） |
| `WEB_PORT` | ❌ | Web 服务端口（默认 8001） |

## 安全机制

- 数学计算使用 AST 白名单解析，拒绝任意代码执行
- 每个子代理只暴露自己的工具集，减少误用面
- 最大推理轮数 `MAX_ROUNDS = 10` 防止死循环
- 工具执行异常统一捕获并返回错误信息
- 打印输出经 `_safe_gbk` 过滤，Windows GBK 终端不崩
- 记忆工具 Redis 不可用时降级返回提示，绝不影响 Agent 主流程
- Web 接口 X-API-Key 鉴权（恒时比较），防被白嫖服务端 key

## 与项目一的关系

```
项目一（RAG 知识库）         项目三（LangGraph 多 Agent）
┌──────────────────┐         ┌──────────────────────┐
│  ChromaDB        │◀────────│  kb_agent 子代理       │
│  VectorStore     │   检索   │  (search_knowledge_base)│
└──────────────────┘         └──────────────────────┘
```

Agent 直接导入项目一的 `VectorStore`，指向同一个 `rag_chat/chroma_db` 数据库。

## 项目结构

```
langgraph_agent/
├── agent.py          # 多 Agent 状态机（Router + 子代理嵌套）+ CLI
├── tools.py          # 工具定义（Anthropic Tool Use 格式）+ 执行函数 + 工作记忆
├── app_web.py        # Web 服务：FastAPI + SSE 节点级流式（复用项目一前端）
├── tests/            # pytest 36 条（纯函数 + mock LLM 全链路）
├── requirements.txt  # 项目依赖
└── .env.example      # 环境变量模板
```

## License

MIT
