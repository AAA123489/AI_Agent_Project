# Day9：FastAPI 整合配置读取、LLM 调用与日志链路

## 目标

将上周沉淀的 `ConfigManager`、`call_llm_client`、`logger` 三大模块整合到 FastAPI 的 `POST /chat` 端点中，打通"配置读取 → 请求日志 → LLM 调用 → 响应返回"完整链路。

## 整合内容

### 1. 全局实例化 ConfigManager

- 在 `main.py` 模块级别执行 `config = ConfigManager()`，只初始化一次。
- 在 `POST /chat` 中通过 `config.api_key` / `config.api_url` 直接读取配置。

**关键点：** import 顺序——`import src.logger` 必须写在最前面，确保根 logger 先挂上 handler，后续其他模块的日志才能正常输出。

### 2. async def 端点 + await LLM 调用

- `/chat` 保持 `async def`，内部 `await call_llm_client(prompt, api_key, api_url)`。
- `call_llm_client` 内部使用 `aiohttp` 异步发请求，与 FastAPI 的异步模型匹配。

### 3. 三节点日志

| 节点 | 日志级别 | 位置 |
|---|---|---|
| 收到请求 | `logger.info` | 端点入口，记录 user_id + 消息摘要 |
| LLM 失败 | `logger.error` | 检查 `result.get("error")` 时 |
| 请求成功 | `logger.info` | 正常返回前 |
| 未预料异常 | `logger.error(..., exc_info=True)` | try/except 最外层兜底 |

### 4. 防御性错误处理

- 用 `try/except Exception` 包裹核心逻辑，意外崩溃时通过 `exc_info=True` 将完整堆栈写入 `app.log`，同时向客户端返回友好提示而非 HTTP 500。
- 检查 LLM 返回的 `content` 数组中是否包含 `type: "text"` 的块，缺少时给出明确提示。

## 踩坑记录

### 坑 1：ConfigManager 忘了括号

```python
# ❌ 错误
config = ConfigManager    # 指向类本身，config.api_key 报 AttributeError

# ✅ 正确
config = ConfigManager()  # 实例化
```

### 坑 2：LLM 响应格式用错

DeepSeek 原生 API 返回 Anthropic/Messages 风格，而非 OpenAI 风格：

```python
# ❌ 错误（OpenAI 格式）
result["choices"][0]["message"]["content"]  # KeyError: 'choices'

# ✅ 正确（Anthropic 格式：content 是数组，需要遍历找 type=="text"）
for block in result.get("content", []):
    if block.get("type") == "text":
        reply_text += block.get("text", "")
```

### 坑 3：max_tokens 太小，回复被截断

初始 `max_tokens=100`，全部被模型的 thinking 过程吃掉，最终 `content` 数组里没有 `type: "text"` 的块。改为 `max_tokens=4096` 解决。

### 坑 4：`import src.logger` 看似未使用

`import xxx` 不是废代码——`src/logger.py` 的模块级代码在导入时执行，给根 logger 挂上 handler。去掉这一行，所有 `logger.info/error` 都不会输出。

## 文件变更

| 文件 | 变更 |
|---|---|
| `main.py` | 整合 ConfigManager、call_llm_client、日志链路、异常兜底 |
| `src/llm_client.py` | `max_tokens` 从 100 调整到 4096 |

## 从中学到的

- `exc_info=True` 让异常堆栈记录到日志文件，用户看到的是友好提示而非红色报错。
- 调用第三方 API 前，先打印原始返回值，确认响应结构再做解析。
- 模块导入的副作用也是一种合法的设计模式（如日志初始化）。
