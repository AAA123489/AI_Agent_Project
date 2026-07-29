# AI 智能知识库问答系统（RAG Chat）

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Chroma](https://img.shields.io/badge/Chroma-vector--db-FF6B6B?style=flat&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)
[![MCP](https://img.shields.io/badge/MCP-protocol-6E3FF3?style=flat&logo=anthropic&logoColor=white)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE)

基于 FastAPI 的 RAG（检索增强生成）智能问答后端，支持文档入库、向量语义检索、多轮对话记忆、SSE 流式响应，端到端闭环。

## 核心能力

| 能力 | 说明 |
|------|------|
| 📄 文档解析 | 支持 TXT / Markdown / PDF，自动解析并提取文本 |
| ✂️ 智能分块 | 自研递归文本切分器，按段落→句子→空格优先级切分，带重叠窗口 |
| 🔢 向量检索 | Chroma 向量数据库，余弦相似度检索，distance 阈值兜底 |
| 💬 流式对话 | SSE（Server-Sent Events）流式推送，逐字返回 LLM 生成内容 |
| 🧠 多轮记忆 | Redis 短期记忆管道，LPUSH + EXPIRE 30 分钟自动过期 |
| 🔐 API 鉴权 | 自定义 X-API-Key 认证，FastAPI 依赖注入实现 |
| 📊 对话持久化 | SQLAlchemy 异步 ORM，对话历史自动入库、支持查询 |

## 技术栈

- **Web 框架**：FastAPI + Uvicorn（ASGI）
- **数据库**：SQLite + SQLAlchemy 2.0 ORM
- **缓存 / 记忆**：Redis（异步客户端 redis-py）
- **向量库**：Chroma（本地持久化 + HNSW 余弦索引）
- **Embedding 模型**：paraphrase-multilingual-MiniLM-L12-v2（中英多语言支持）
- **LLM 调用**：aiohttp 异步流式请求 → DeepSeek API
- **流式协议**：SSE（Server-Sent Events）
- **数据校验**：Pydantic V2
- **日志系统**：RotatingFileHandler，500KB 自动滚动

## 系统架构

```
用户浏览器（chat.html）
        │
        ▼
   POST /chat ──────────────────────────────┐
        │                                    │
        ▼                                    │
   X-API-Key 鉴权（依赖注入）                 │
        │                                    │
        ▼                                    │
   ┌─ 向量检索 ─┐   ┌─ Redis ─┐             │
   │ Chroma     │   │ 查询历史 │             │
   │ 语义匹配   │   │ 多轮对话 │             │
   └─────┬──────┘   └───┬─────┘             │
         │               │                   │
         ▼               ▼                   │
   ┌──────────────────────────┐              │
   │  拼装 RAG Prompt         │              │
   │  历史 + 上下文 + 问题     │              │
   └──────────┬───────────────┘              │
              │                              │
              ▼                              │
   ┌──────────────────────────┐              │
   │  DeepSeek API 流式调用    │              │
   │  aiohttp 异步逐块读取     │              │
   └──────────┬───────────────┘              │
              │                              │
              ▼                              │
   ┌──────────────────────────┐              │
   │  SSE StreamingResponse   │──────────────┘
   │  逐字推送给前端           │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  Redis 写入最新对话        │
   │  SQLite 持久化历史记录     │
   └──────────────────────────┘
```

## 目录结构

```
AI_Agent_Project/
├── main.py                  # FastAPI 应用入口（生命周期、CORS、路由）
├── database.py              # SQLAlchemy 引擎 & 会话工厂
├── model.py                 # ChatHistory ORM 数据模型
├── redis_client.py          # Redis 短期记忆管道
├── prompts.py               # RAG Prompt 模板（历史+上下文+问题）
├── rag_pipeline.py          # RAG 文档入库流水线（解析→切分→Hash→入库）
├── text_splitter.py         # 自研递归文本切分器
├── document_parser.py       # PDF 解析器（pypdf）
├── requirements.txt         # 项目依赖
├── .env.example             # 环境变量模板
├── test_sse.html            # 浏览器 SSE 流式测试页面（调试用）
├── static/
│   └── chat.html            # 聊天前端页面（演示用）
├── app/
│   ├── dependencies/__init__.py   # API Key 鉴权 + 数据库会话注入
│   ├── routers/chat.py            # /chat（SSE流式+向量检索）
│   │                              # /history/{user_id}（对话历史）
│   └── schemas/chat.py            # Pydantic 请求/响应模型
└── src/
    ├── config.py             # .env 配置管理
    ├── llm_client.py         # LLM 异步流式调用客户端
    ├── logger.py             # 日志配置
    └── vector_store.py       # Chroma 向量库封装
```

## 环境要求

- **Python**：3.12+
- **操作系统**：Windows / macOS / Linux
- **网络**：需能访问 DeepSeek API（`https://api.deepseek.com`）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/AAA123489/AI_Agent_Project.git
cd AI_Agent_Project
```

### 2. 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Windows Git Bash）
source .venv/Scripts/activate

# 激活（macOS / Linux）
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```bash
cp .env.example .env
# 然后编辑 .env，把 your_api_key_here 替换成真实的 Key
```

`.env` 文件内容：

```ini
API_KEY=你的DeepSeek_API密钥
API_URL=https://api.deepseek.com/anthropic/v1/messages
MODEL_NAME=deepseek-v4-pro
REDIS_URL=redis://localhost:6379
```

> ⚠️ `.env` 已在 `.gitignore` 中排除，不会被提交到 Git。

### 5. 启动服务

```bash
python main.py
```

服务启动后访问：
- 聊天前端：http://127.0.0.1:8000/static/chat.html
- API 文档（Swagger UI）：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

## API 端点

### `GET /health`

健康检查接口，无需鉴权。

**响应示例：**

```json
{"status": "ok", "version": "1.0.0"}
```

### `POST /chat`

SSE 流式对话接口，需 API Key 鉴权。

**请求头：**

| Header | 值 |
|--------|-----|
| `Content-Type` | `application/json` |
| `X-API-Key` | 你的 API Key（与 `.env` 中 `API_KEY` 一致） |

**请求体（JSON）：**

```json
{
  "user_id": "zhangsan",
  "message": "你好，请介绍一下你自己"
}
```

| 字段 | 类型 | 约束 |
|------|------|------|
| `user_id` | `string` | 2-50 字符 |
| `message` | `string` | 2-1000 字符 |

**响应：** `text/event-stream`（SSE 格式），逐块返回 LLM 生成的文本。

### `GET /history/{user_id}`

查询指定用户的对话历史。

**响应示例：**

```json
[
  {
    "id": 1,
    "user_id": "zhangsan",
    "role": "user",
    "content": "你好",
    "created_at": "2026-07-25T10:30:00"
  },
  {
    "id": 2,
    "user_id": "zhangsan",
    "role": "assistant",
    "content": "你好！有什么可以帮助你的吗？",
    "created_at": "2026-07-25T10:30:05"
  }
]
```

## 鉴权方式

本项目使用自定义 Header `X-API-Key` 进行 API Key 认证。

- `.env` 中的 `API_KEY` 同时作为**服务端调用 LLM 的凭证**和**客户端访问服务端的凭证**
- 鉴权逻辑在 `app/dependencies/__init__.py` 中通过 FastAPI `Depends` 实现
- `/health` 端点无需鉴权

**鉴权失败场景：**

| 场景 | HTTP 状态码 | 日志输出 |
|------|-----------|----------|
| 未传 `X-API-Key` Header | 401 | `未提供 API Key，客户端 IP: ...` |
| 传了但 Key 不匹配 | 401 | `API Key 匹配失败，客户端 IP: ...` |

## 环境变量参考

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `API_KEY` | ✅ | — | DeepSeek API 密钥，同时用作服务端鉴权凭证 |
| `API_URL` | ✅ | — | DeepSeek Anthropic 兼容接口地址 |
| `MODEL_NAME` | ❌ | `deepseek-v4-pro` | 对话模型名称 |
| `REDIS_URL` | ❌ | `redis://localhost:6379` | Redis 服务连接地址 |
| `LOG_LEVEL` | ❌ | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |

## 模块说明

| 模块 | 功能 |
|------|------|
| `main.py` | FastAPI 应用入口：SSE 流式 `/chat` 端点、`/health` 健康检查、CORS 中间件、lifespan 生命周期 |
| `app/dependencies/__init__.py` | `verify_api_key` 依赖注入：从 `X-API-Key` Header 读取并校验 API Key；`get_db` 数据库会话 |
| `app/routers/chat.py` | `/chat` SSE 流式对话（向量检索+RAG）、`/health` 健康检查、`/history/{user_id}` 历史查询 |
| `app/schemas/chat.py` | `ChatRequest` / `ChatHistoryResponse` Pydantic 模型 |
| `src/config.py` | `ConfigManager` 类：从 `.env` 读取配置 |
| `src/logger.py` | 统一日志实例：同时输出到 stderr 和 `app.log`（500KB 滚动，保留 3 份） |
| `src/llm_client.py` | `call_llm_stream()`：异步生成器，通过 aiohttp 流式读取 LLM API 响应 |
| `src/vector_store.py` | Chroma 向量数据库封装：文档增删查、相似度检索 |
| `database.py` | SQLAlchemy 数据库引擎与会话工厂 |
| `model.py` | `ChatHistory` ORM 模型 |
| `redis_client.py` | Redis 异步客户端：LPUSH + EXPIRE 存消息，LRANGE 取历史 |
| `rag_pipeline.py` | 文档入库流水线：读取 → 切分 → SHA-256 生成 ID → 批量存入 Chroma |
| `text_splitter.py` | 递归文本切分器（支持多级分隔符） |
| `document_parser.py` | PDF 文件解析器（pypdf） |
| `prompts.py` | RAG Prompt 模板拼装（历史对话 + 检索上下文 + 当前问题） |

## Postman 测试指南

### 环境准备

1. 确保服务已启动：`python main.py`（默认监听 `http://127.0.0.1:8000`）
2. 打开 Postman，新建一个 Collection，在 Collection 的 **Variables** 中设置：

| 变量名 | 初始值 |
|--------|--------|
| `base_url` | `http://127.0.0.1:8000` |
| `api_key` | 你的 API Key（与 `.env` 中 `API_KEY` 一致） |

### 场景一：正向流式输出 ✅

| 配置项 | 值 |
|--------|-----|
| **Method** | `POST` |
| **URL** | `{{base_url}}/chat` |
| **Headers** | `Content-Type: application/json` |
| | `X-API-Key: {{api_key}}` |
| **Body** | `raw` → `JSON` |

```json
{
  "user_id": "zhangsan",
  "message": "写一首赞美程序员的七言律诗"
}
```

**预期结果：** Postman 以 SSE 格式逐块显示 LLM 生成的文本流。

> 💡 确保在 Postman 的 **Settings** → **General** 中，`Request timeout` 设置得足够长（建议 0 表示不超时），避免长文本没流完就被 Postman 截断。

### 场景二：逆向 — 鉴权失败（401）🚫

**测试 2a：不传 API Key**

删除 `X-API-Key` Header，发送请求。

**预期结果：** HTTP `401 Unauthorized`，响应体：

```json
{"detail": "Unauthorized"}
```

同时在 `app.log` 中发现日志：`未提供 API Key，客户端 IP: 127.0.0.1`

**测试 2b：错误的 API Key**

将 `X-API-Key` 的值改成错误的值（如 `wrong-key-123`），发送请求。

**预期结果：** HTTP `401 Unauthorized`，`app.log` 中出现：`API Key 匹配失败，客户端 IP: 127.0.0.1`

### 场景三：逆向 — 参数校验失败（422）🚫

**测试 3a：message 太短**

```json
{
  "user_id": "zhangsan",
  "message": "1"
}
```

**测试 3b：缺少必填字段**

```json
{
  "user_id": "zhangsan"
}
```

**预期结果：** HTTP `422 Unprocessable Entity`，响应体中包含 Pydantic 自动生成的校验错误详情（指明哪个字段、什么约束不满足）。

### 场景四：手动取消请求（测试 finally 兜底）🛑

1. 发送一条会让 LLM 生成较长回复的请求（例如 `"写一篇关于人工智能的 5000 字论文"`）
2. 在 Postman 的响应区域看到流式输出开始后，点击 **Cancel** 按钮（或直接关闭请求 Tab）
3. 打开 `app.log`，验证以下两行日志依次出现：

```
LLM 流式连接关闭
客户端断开
流结束了
```

- `LLM 流式连接关闭` → `src/llm_client.py` 的 `finally` 块触发
- `客户端断开` → `main.py` 中 `asyncio.CancelledError` 被捕获
- `流结束了` → `main.py` 中 `generate_stream` 的 `finally` 块触发

这证明了双层 finally 兜底机制正常工作：客户端断开 → `CancelledError` → 上游 aiohttp 连接关闭 → 资源清理。

## 许可证

MIT
