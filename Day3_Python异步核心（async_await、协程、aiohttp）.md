# Day 3：Python 异步核心（async/await、协程、aiohttp）

## 任务目标

1. 封装异步函数 `async def call_llm_client(prompt)`，使用 `aiohttp` 向大模型接口发送 POST 请求
2. 在 `main.py` 中，使用 `asyncio.gather()` 同时并发发起 3 个不同的提问请求
3. 结合 `logging` 模块，在请求开始、成功、失败时记录日志
4. 正确处理异常：HTTP 错误和网络异常统一返回错误字典，不中断其他并发任务

---

## 最终文件结构

```
AI_Agent_Project/
├── .env               # 环境变量（API_KEY, API_URL, MODEL_NAME）
├── .gitignore
├── config.py          # Day 2：ConfigManager 类
├── logger.py          # Day 2：logging 日志配置
├── llm_client.py      # Day 3：异步 LLM 调用封装
├── main.py            # Day 3：asyncio.gather() 并发入口
├── file_handler.py    # Day 1：文件读取工具
├── requirements.txt   # 依赖：python-dotenv, loguru, aiohttp
├── app.log            # 日志输出文件
└── agent_env/         # 虚拟环境
```

---

## 各文件最终代码

### `llm_client.py`

```python
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
from logger import logger

load_dotenv()

async def call_llm_client(prompt: str):
    URL = os.getenv("API_URL")
    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "max_tokens": 100,
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": prompt}]
    }

    logger.info(f"发送请求的日志: {prompt}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("请求成功")
                    return data
                else:
                    error_message = await response.text()
                    logger.error(f"请求失败，状态码 {response.status}: {error_message}")
                    return {"error": True, "status": response.status, "message": error_message}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"请求异常: {str(e)}")
            return {"error": True, "message": str(e)}
```

### `main.py`

```python
from logger import logger
import asyncio
import llm_client

async def main():
    logger.info("开始发送请求到 LLM 客户端")

    prompts = [
        "古诗介绍 Python",
        "请告诉我今天的天气。",
        "推荐三本编程书"
    ]

    results = await asyncio.gather(
        llm_client.call_llm_client(prompts[0]),
        llm_client.call_llm_client(prompts[1]),
        llm_client.call_llm_client(prompts[2]),
    )

    for i, result in enumerate(results):
        if "error" in result:
            logger.error(f"请求 {i+1} 失败: {result['message']}")
        else:
            logger.info(f"请求 {i+1} 成功，响应数据: {result}")

    logger.info("所有请求已完成")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 运行结果

```text
2026-07-09 16:09:23,252 INFO 开始发送请求到 LLM 客户端
2026-07-09 16:09:23,252 INFO 发送请求的日志: 古诗介绍 Python
2026-07-09 16:09:23,252 INFO 发送请求的日志: 请告诉我今天的天气。
2026-07-09 16:09:23,252 INFO 发送请求的日志: 推荐三本编程书
2026-07-09 16:09:23,773 INFO 请求成功
2026-07-09 16:09:23,773 INFO 请求成功
2026-07-09 16:09:23,773 INFO 请求成功
2026-07-09 16:09:23,773 INFO 请求 1 成功，响应数据: {...}
2026-07-09 16:09:23,773 INFO 请求 2 成功，响应数据: {...}
2026-07-09 16:09:23,773 INFO 请求 3 成功，响应数据: {...}
2026-07-09 16:09:23,773 INFO 所有请求已完成
```

> 注意：三个请求的"开始"时间戳几乎相同（.252），证明并发生效。

---

## 踩坑记录（知识点总结）

| # | 错误写法 | 报错/现象 | 正确写法 | 知识点 |
|---|----------|-----------|----------|--------|
| 1 | `"message": [...]`（单数） | API 返回 404/400 | `"messages": [...]`（复数） | Anthropic API 的字段名是复数 `"messages"`；一个 `s` 之差服务端找不到对应字段 |
| 2 | `except Exception as e` | 捕获范围过大 | `except (aiohttp.ClientError, asyncio.TimeoutError) as e` | 要精确捕获与任务相关的异常类型，避免吞掉不相关的错误（如 `KeyboardInterrupt`） |
| 3 | `raise Exception(...)` 在并发任务中 | 一个任务抛异常，`gather` 取消所有其他任务 | `return {"error": True, ...}` | 并发任务应该"吃掉"异常后返回错误对象，保证其他任务不受影响 |
| 4 | `await await asyncio.gather(...)` | 语法错误 | `results = await asyncio.gather(...)` | `gather` 返回的已经是协程结果，不需要再 `await` 两次；`=` 普通赋值即可，不需要海象 `:=` |
| 5 | `asyncio.gather()` 不接返回值 | 结果全部丢失 | `results = await asyncio.gather(...)` | `gather` 返回一个列表，顺序和传入的任务顺序一致 |
| 6 | `if` 语句写在函数调用参数里 | `SyntaxError` | 逻辑写到 `gather()` 外面 | 函数调用括号内只能是表达式，不能放 `if` / `for` 等语句 |
| 7 | 海象运算符 `:=` 用在不需要的地方 | 语法晦涩、易出错 | 普通赋值 `=` | `:=` 用于 `if`/`while` 中"边判断边赋值"的场景，普通赋值用 `=` 就够 |
| 8 | `pip install` 直接安装 | 系统 Python 装好了，虚拟环境里还是 `ModuleNotFoundError` | `.\agent_env\Scripts\python.exe -m pip install aiohttp` | 虚拟环境有独立的 `site-packages`；移动过虚拟环境后 `pip.exe` 路径可能失效，用 `python -m pip` 更可靠 |
| 9 | `await` 写在 `async def` 外面 | Pylance: "await is only allowed in async function" | 放在 `async def` 内部；顶层用 `asyncio.run()` 驱动 | `await` 只能在协程函数内部使用；`asyncio.run()` 是同步世界进入异步世界的入口 |
| 10 | 在文件顶层写 `await call_llm("test")` | SyntaxError | 写在 `if __name__ == "__main__": asyncio.run(main())` | `.py` 文件的顶层是同步上下文，不支持 `await` |
| 11 | DeepSeek think 模型 `max_tokens=100` | 回复被截断，只有 thinking 没有 text | 调大到 4096 或更大 | Think 模型会先内部推理（`type: "thinking"`）再输出正文（`type: "text"`），`max_tokens` 太小会导致正文来不及输出 |

---

## 核心概念对比

### 同步 vs 异步 HTTP 请求

| | `requests`（同步） | `aiohttp`（异步） |
|---|---|---|
| 函数定义 | `def fetch():` | `async def fetch():` |
| 发起请求 | `requests.post(url)` | `async with session.post(url) as resp:` |
| 等待结果 | 阻塞（卡住当前线程） | `await`（挂起协程，让出控制权） |
| 并发 3 个请求 | 顺序执行，总耗时 = A+B+C | 几乎同时发出，总耗时 ≈ max(A, B, C) |
| 适用场景 | 脚本、简单请求 | 高并发、IO 密集 |

### `await` 到底在等什么？

```python
response = await session.post(...)   # # 等的是"IO 完成"这个事件
data = await response.json()         # # 等的是"响应体读取完毕"
```

- `await` 不是"轮询等待 CPU 计算完毕"，而是"把这个协程挂起，告诉事件循环：这个 IO 操作完成了记得把我唤醒"
- 在等待期间，事件循环可以切换去执行其他协程 —— 这就是并发的来源

### `asyncio.gather()` vs `asyncio.run()`

| | `asyncio.run()` | `asyncio.gather()` |
|---|---|---|
| 作用 | 创建一个事件循环，运行一个协程直到它完成 | 并发运行多个协程，等全部完成后返回结果列表 |
| 调用位置 | 同步代码的入口（`main.py` 顶层） | 异步函数内部 |
| 返回值 | 协程的返回值 | `list`：每个协程的返回值，顺序与传入一致 |
| 异常行为 | 异常传播到调用方 | 默认：一个协程抛异常，`gather` 抛异常，其他协程被取消 |

### `aiohttp.ClientSession` 生命周期

```python
# ✅ 正确：用 async with，自动管理连接池的开启和关闭
async with aiohttp.ClientSession() as session:
    async with session.post(url) as response:
        ...

# ❌ 错误：每次请求新建 session，TCP 连接无法复用
async def call():
    session = aiohttp.ClientSession()  # 用完就丢
    ...
```

- `ClientSession` 内部维护连接池，复用 TCP 连接
- 频繁创建/销毁 session 等于每次重新握手，失去异步的优势

---

## 延伸练习（选做）

1. 用 `asyncio.as_completed()` 替代 `gather`，实现"哪个先返回就先处理哪个"
2. 给 `session.post()` 加上 `timeout=aiohttp.ClientTimeout(total=10)` 参数，防止某个请求无限挂起
3. 在 `main.py` 中从 `content` 里提取 `type: "text"` 的字段，只打印模型的最终回答文本，而不是整坨 JSON
4. 把 `prompts` 数量扩展到 10 个，观察并发数是否受限制（了解 `asyncio.Semaphore` 信号量的作用）
5. 对比 `requests` 同步版本和 `aiohttp` 异步版本在 3 个请求时的耗时差异，用代码计时验证
