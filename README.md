# AI Agent 项目 — FastAPI + SSE 流式对话服务

基于 FastAPI 的 SSE（Server-Sent Events）流式 LLM 对话服务，支持 API Key 鉴权、请求参数校验、结构化日志与客户端断连优雅处理。

当前适配的 API：**DeepSeek（Anthropic 兼容接口）**，模型为 `deepseek-v4-pro`。

## 项目简介

本项目是一个学习型 AI Agent 后端服务，核心能力包括：

- **SSE 流式响应** — 通过 `aiohttp` 逐块读取 LLM 原始流 → buffer 拼行 → JSON 过滤提取纯文本 → `StreamingResponse` 推送给前端
- **API Key 鉴权** — 基于 FastAPI `Depends` 依赖注入的自定义 Header 认证，区分"未传 Key"与"Key 不匹配"两种失败场景
- **配置管理** — 通过 `.env` 文件管理 API 密钥、接口地址、日志级别等敏感配置
- **结构化日志** — 统一的日志模块，同时输出到控制台和滚动文件（`app.log`），覆盖鉴权、LLM 调用、流式连接所有关键节点
- **断连优雅处理** — 双层 `finally` + `CancelledError` 捕获，客户端断开时正确关闭上游 LLM 连接

## 目录结构

```
AI_Agent_Project/
├── main.py                  # FastAPI 入口：SSE 流式 /chat 端点 + /health 健康检查
├── dependencies.py          # FastAPI 依赖注入：X-API-Key 鉴权
├── requirements.txt         # 项目依赖清单（pip freeze 锁定版本）
├── .env                     # 环境变量（API_KEY、API_URL 等，已加入 .gitignore）
├── .gitignore               # Git 忽略规则
├── app.log                  # 运行时日志文件（500KB 滚动，保留 3 份）
├── test_sse.html            # 浏览器端 SSE 流式测试页面
├── app/
│   ├── __init__.py          # 包标记文件
│   └── schemas.py           # Pydantic 请求模型（ChatRequest）
└── src/
    ├── __init__.py          # 包标记文件
    ├── config.py            # ConfigManager：读取和管理配置参数
    ├── logger.py            # 统一日志实例（RotatingFileHandler + stderr）
    ├── llm_client.py        # call_llm_stream：异步流式 LLM API 客户端
    └── file_handler.py      # 文件读取工具
```

## 环境要求

- **Python**：3.12+
- **操作系统**：Windows / macOS / Linux
- **网络**：需能访问 DeepSeek API（`https://api.deepseek.com`）

## 快速开始

### 1. 克隆项目

```bash
git clone <你的仓库地址>
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

在项目根目录创建 `.env` 文件，填入以下内容：

```ini
API_KEY=你的DeepSeek_API密钥
API_URL=https://api.deepseek.com/anthropic/v1/messages
MODEL_NAME=deepseek-v4-pro
LOG_LEVEL=INFO
```

> ⚠️ `.env` 已在 `.gitignore` 中排除，不会被提交到 Git。

### 5. 启动服务

```bash
python main.py
```

服务启动后访问：
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

## 鉴权方式

本项目使用自定义 Header `X-API-Key` 进行 API Key 认证。

- `.env` 中的 `API_KEY` 同时作为**服务端调用 LLM 的凭证**和**客户端访问服务端的凭证**
- 鉴权逻辑在 `dependencies.py` 中通过 FastAPI `Depends` 实现
- `/health` 端点无需鉴权

**鉴权失败场景：**

| 场景 | HTTP 状态码 | 日志输出 |
|------|-----------|----------|
| 未传 `X-API-Key` Header | 401 | `未提供 API Key，客户端 IP: ...` |
| 传了但 Key 不匹配 | 401 | `API Key 匹配失败，客户端 IP: ...` |

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

## 环境变量参考

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `API_KEY` | ✅ | — | DeepSeek API 密钥，同时用作服务端鉴权凭证 |
| `API_URL` | ✅ | — | DeepSeek Anthropic 兼容接口地址 |
| `MODEL_NAME` | ❌ | `deepseek-v4-pro` | 对话模型名称 |
| `LOG_LEVEL` | ❌ | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |

## 模块说明

| 模块 | 功能 |
|------|------|
| `main.py` | FastAPI 应用入口：SSE 流式 `/chat` 端点、`/health` 健康检查、CORS 中间件 |
| `dependencies.py` | `verify_api_key` 依赖注入：从 `X-API-Key` Header 读取并校验 API Key |
| `app/schemas.py` | `ChatRequest` Pydantic 模型：校验 `user_id` 和 `message` 字段 |
| `src/config.py` | `ConfigManager` 类：从 `.env` 读取配置，动态设置日志级别 |
| `src/logger.py` | 项目级 `logger` 实例：同时输出到 stderr 和 `app.log`（500KB 滚动，保留 3 份） |
| `src/llm_client.py` | `call_llm_stream()`：异步生成器，通过 aiohttp 流式读取 LLM API 响应 |
| `src/file_handler.py` | `read_text_file()`：以 UTF-8 安全读取文本文件 |

## 核心依赖

| 包名 | 作用 |
|------|------|
| `fastapi` | Web 框架，路由与中间件 |
| `uvicorn` | ASGI 服务器 |
| `pydantic` | 请求数据校验 |
| `starlette` | `StreamingResponse` SSE 流式响应 |
| `aiohttp` | 异步 HTTP 客户端（向 LLM 发流式请求） |
| `python-dotenv` | 加载 `.env` 环境变量 |
| `fastapi-cdn-host` | 国内 CDN 加速 Swagger UI |

## 许可证

MIT
