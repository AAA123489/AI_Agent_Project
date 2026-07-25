# 技术知识库 — RAG 系统测试文档

---

## 一、FastAPI 框架的特点与优势

FastAPI 是一个用于构建 API 的现代 Python Web 框架，基于 Starlette（ASGI）和 Pydantic 构建，专为高性能和开发者体验而设计。

### 核心特点

**1. 高性能**：FastAPI 基于异步（async/await）和 Starlette 的 ASGI 实现，能够处理大量并发请求。在 TechEmpower 基准测试中，FastAPI 的性能可与 Go、Rust 等系统级语言框架相媲美。

**2. 自动文档生成**：FastAPI 会自动生成 OpenAPI 3.0 规范的接口文档（Swagger UI 和 ReDoc），开发者只需运行服务即可在 `/docs` 和 `/redoc` 路径下查看交互式 API 文档。

**3. 类型提示驱动**：基于 Python 的 type hints 和 Pydantic 模型，FastAPI 自动完成请求数据的验证、序列化（JSON 编码）和文档生成，减少样板代码。

**4. 依赖注入系统**：内置强大的依赖注入（Dependency Injection）机制，支持异步依赖、上下文管理器和嵌套依赖，便于实现数据库连接、认证逻辑等横切关注点的复用。

### 代码示例

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="RAG Demo API", version="1.0.0")

class Item(BaseModel):
    name: str
    price: float
    is_available: bool = True

# 内存存储（实际项目中替换为数据库）
items_db: List[Item] = []

@app.get("/items/", response_model=List[Item])
async def list_items():
    """列出所有商品"""
    return items_db

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    """创建新商品"""
    items_db.append(item)
    return item

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """按 ID 获取商品"""
    if item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="商品不存在")
    return items_db[item_id]
```

### 适用场景

- 微服务架构中的 API 网关
- 需要高性能的实时数据处理管道
- 与异步数据库（如 asyncpg、Motor）配合的后端服务
- 需要自动生成文档的团队协作项目

---

## 二、Python 异步编程（async/await）基础概念

Python 的异步编程模型通过 `async` 和 `await` 关键字实现，基于事件循环（Event Loop）机制，允许在单个线程中并发执行多个 I/O 密集型任务，而不会阻塞主线程。

### 核心概念

**协程（Coroutine）**：使用 `async def` 定义的函数称为协程函数，调用它不会立即执行，而是返回一个协程对象（Coroutine Object），需要通过事件循环来运行。

**await 表达式**：在协程中使用 `await` 挂起当前任务的执行，等待某个可等待对象（awaitable）完成。被 await 的对象可以是另一个协程、Task、Future 或实现了 `__await__` 方法的对象。

**事件循环（Event Loop）**：异步程序的核心调度器，负责注册、执行和调度协程。当协程遇到 `await` 时，事件循环会切换到其他就绪的协程继续执行。

### 代码示例

```python
import asyncio
import time

async def fetch_data(source: str, delay: float) -> dict:
    """模拟从不同数据源获取数据"""
    print(f"[{source}] 开始获取数据...")
    await asyncio.sleep(delay)  # 异步等待，不阻塞其他协程
    print(f"[{source}] 数据获取完成")
    return {"source": source, "data": f"result_from_{source}", "latency": delay}

async def main():
    start = time.time()
    
    # 方式1：使用 asyncio.gather 并发执行多个协程
    results = await asyncio.gather(
        fetch_data("数据库", 2.0),
        fetch_data("API接口", 1.5),
        fetch_data("缓存服务", 0.5),
    )
    
    # 方式2：使用 asyncio.create_task 创建独立任务
    task1 = asyncio.create_task(fetch_data("任务A", 1.0))
    task2 = asyncio.create_task(fetch_data("任务B", 1.5))
    task3 = asyncio.create_task(fetch_data("任务C", 0.8))
    results2 = await asyncio.gather(task1, task2, task3)
    
    # 设置超时
    try:
        result = await asyncio.wait_for(fetch_data("慢服务", 5.0), timeout=2.0)
    except asyncio.TimeoutError:
        print("请求超时！")
    
    elapsed = time.time() - start
    print(f"总耗时: {elapsed:.2f} 秒（并发执行远快于串行 7.3 秒）")

asyncio.run(main())
```

### 常见场景

- 高并发 Web 服务器（如 FastAPI 后端处理大量并发请求）
- 网络爬虫（同时发起多个 HTTP 请求）
- 实时数据流处理（WebSocket 连接、消息队列消费）
- 数据库批量操作（并发查询多个数据源并合并结果）

### 注意事项

- `async/await` 仅适用于 I/O 密集型任务，CPU 密集型任务应使用 `multiprocessing` 或 `concurrent.futures.ThreadPoolExecutor`
- 避免在异步代码中调用同步阻塞函数（如 `time.sleep`），应使用 `asyncio.sleep`
- 确保第三方库支持异步（如使用 `httpx` 而非 `requests`）

---

## 三、Redis 的基本用法和常见场景

Redis（Remote Dictionary Server）是一个基于内存的高性能键值（Key-Value）存储系统，支持多种数据结构，常用作数据库、缓存和消息中间件。

### 核心数据结构

| 数据类型 | 说明 | 典型用例 |
|---------|------|---------|
| String | 字符串，支持原子操作 | 计数器、分布式锁、会话存储 |
| Hash | 字段-值对集合 | 用户信息存储、部分更新场景 |
| List | 双向链表 | 消息队列、最近访问列表 |
| Set | 无序唯一集合 | 标签系统、共同好友 |
| ZSet (Sorted Set) | 有序唯一集合 | 排行榜、延迟队列 |

### 代码示例（Python + redis-py）

```python
import redis
import json

# 连接 Redis（默认 localhost:6379）
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# --- String 操作 ---
r.set("user:1001:name", "张三", ex=3600)  # 设置键值，1小时后过期
name = r.get("user:1001:name")
print(f"用户姓名: {name}")

# --- Hash 操作 ---
r.hset("user:1001", mapping={
    "name": "张三",
    "age": "28",
    "email": "zhangsan@example.com"
})
user_info = r.hgetall("user:1001")
print(f"用户信息: {user_info}")

# --- List 操作（简易消息队列） ---
r.lpush("task_queue", json.dumps({"task": "send_email", "to": "user@example.com"}))
task = r.rpop("task_queue")  # 从队列尾部取出并删除
if task:
    print(f"处理任务: {task}")

# --- ZSet 操作（排行榜） ---
r.zadd("leaderboard", {"Alice": 95, "Bob": 88, "Charlie": 92})
rank = r.zrevrange("leaderboard", 0, 2, withscores=True)
print(f"排行榜: {rank}")  # [('Alice', 95.0), ('Charlie', 92.0), ('Bob', 88.0)]

# --- 事务与管道 ---
pipe = r.pipeline()
pipe.set("counter", "0")
pipe.incr("counter")
pipe.incr("counter")
results = pipe.execute()
print(f"管道执行结果: {results}")  # [True, 1, 2]
```

### 常见应用场景

1. **缓存层**：缓存数据库查询结果、API 响应，减轻后端压力
2. **会话存储**：存储用户登录态、购物车等临时数据
3. **分布式锁**：利用 `SETNX` + `EXPIRE` 实现跨进程/服务器的互斥锁
4. **消息队列**：结合 List 或 Stream 实现任务调度（如邮件发送、图片处理）
5. **实时排行榜**：利用 ZSet 的排序能力实现游戏/电商排行榜
6. **限流器**：结合 ZSet 实现滑动窗口限流算法

---

## 四、Chroma 向量数据库简介

Chroma 是一个开源的嵌入式向量数据库，专为 AI 应用设计，以轻量级、易集成和开发者友好著称。它支持将文本、图像等多模态数据转化为向量（Embedding），并提供高效的相似度搜索能力，是构建 RAG 系统的核心组件之一。

### 核心特性

**1. 嵌入式架构**：Chroma 以 Python 库（`chromadb`）的形式直接嵌入应用进程，无需独立部署服务，开发调试极为便捷。同时也支持 Client-Server 模式用于生产环境。

**2. 内置 Embedding 支持**：默认使用 Sentence Transformers 的 `all-MiniLM-L6-v2` 模型进行文本向量化，也可接入 OpenAI、HuggingFace 等外部 Embedding API。

**3. 持久化存储**：支持将向量数据和元数据持久化到磁盘，重启后数据不丢失。

**4. 丰富的元数据过滤**：支持在相似度搜索时通过元数据（metadata）进行精确过滤，实现"先过滤后检索"的两阶段检索策略。

### 代码示例

```python
import chromadb
from chromadb.utils import embedding_functions

# 初始化 Chroma 客户端（默认在内存中运行）
client = chromadb.Client()

# 创建集合（Collection），指定持久化路径
collection = client.get_or_create_collection(
    name="rag_knowledge_base",
    metadata={"description": "RAG 系统测试知识库"},
)

# 批量添加文档（自动向量化）
documents = [
    "FastAPI 是一个高性能的 Python Web 框架。",
    "Redis 是内存键值数据库，支持多种数据结构。",
    "向量数据库将文本转化为数值向量进行相似度搜索。",
    "RAG 通过检索外部知识增强大语言模型的生成能力。",
]
metadatas = [
    {"source": "fastapi_doc", "category": "框架"},
    {"source": "redis_doc", "category": "数据库"},
    {"source": "vector_doc", "category": "数据库"},
    {"source": "rag_doc", "category": "架构"},
]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=["doc_1", "doc_2", "doc_3", "doc_4"],
)

# 相似度搜索（返回最相似的 2 条结果）
results = collection.query(
    query_texts=["什么是向量数据库？"],
    n_results=2,
    where={"category": "数据库"},  # 元数据过滤
)
print(results)
```

### 适用场景

- 小型到中型规模的 RAG 系统原型开发
- 文档问答系统（企业知识库、FAQ 检索）
- 语义搜索（基于含义而非关键词匹配）
- 作为大语言模型的"外部记忆"模块

---

## 五、RAG（检索增强生成）的原理和实现步骤

RAG（Retrieval-Augmented Generation）是一种将信息检索与大语言模型生成能力相结合的架构，通过在生成回答前从外部知识库检索相关信息，显著减少大模型的"幻觉"问题，并使其能够利用最新或私有的知识。

### 基本原理

传统大语言模型（LLM）仅依赖训练数据中的参数化知识生成回答，存在知识截止、事实错误和无法访问私有数据等问题。RAG 的核心思想是：在生成回答之前，先从外部向量知识库中检索与用户查询最相关的文档片段，将这些片段作为上下文（Context）与用户问题一起输入 LLM，从而生成更准确、更有依据的回答。

### 实现步骤

**步骤 1：文档索引（Indexing）**

将原始文档（PDF、网页、数据库记录等）进行分块（Chunking），对每个文本块使用 Embedding 模型将其转化为向量，存入向量数据库。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# 文本分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块 500 字符
    chunk_overlap=50,    # 相邻块重叠 50 字符，保留上下文
    separators=["\n\n", "\n", "。", " "]
)

chunks = splitter.split_text(
    "FastAPI 是基于 Starlette 和 Pydantic 构建的现代 Python Web 框架。"
    "它支持异步编程、自动文档生成、类型提示验证等特性。"
    "FastAPI 在 TechEmpower 基准测试中表现优异..."
)

# 向量化并存储
client = chromadb.Client()
collection = client.get_or_create_collection("docs")
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)
```

**步骤 2：检索（Retrieval）**

对用户查询进行向量化，在向量数据库中执行相似度搜索，取 Top-K 个最相关的文档块。

```python
def retrieve(query: str, top_k: int = 3) -> list:
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]  # 返回最相似的 K 个文档文本
```

**步骤 3：生成（Generation）**

将检索到的文档块与用户问题组合成提示词（Prompt），发送给 LLM 生成最终回答。

```python
def generate_answer(query: str, context: list) -> str:
    prompt = (
        "基于以下参考信息回答问题。如果参考信息不足以回答，请说明无法回答。\n\n"
        "参考信息：\n"
        + "\n".join(context) + "\n\n"
        "用户问题：" + query + "\n\n"
        "回答："
    )
    
    # 调用 LLM（以 OpenAI 为例）
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message["content"]

# 完整流程
context = retrieve("FastAPI 有什么特点？")
answer = generate_answer("FastAPI 有什么特点？", context)
print(answer)
```

### 优化策略

- **重排序（Reranking）**：检索后使用交叉编码器（Cross-Encoder）对候选文档重新排序，提升召回质量
- **混合检索（Hybrid Search）**：结合关键词检索（BM25）和向量检索，兼顾精确匹配和语义理解
- **元数据过滤**：在检索阶段通过文档来源、时间、类别等元数据进行预过滤
- **查询改写（Query Rewriting）**：对用户原始查询进行扩展或改写，提升检索召回率

---

## 六、SSE（Server-Sent Events）协议介绍

SSE（Server-Sent Events）是一种基于 HTTP 的单向实时通信协议，允许服务器主动向客户端推送数据流。与 WebSocket 的双向通信不同，SSE 是单向的（服务器到客户端），但具有自动重连、事件类型区分和简单易用等独特优势。

### SSE 的核心机制

SSE 基于标准的 HTTP 请求，客户端通过 `EventSource` API 建立连接，服务器设置 `Content-Type: text/event-stream` 并持续输出格式化的事件数据流，不关闭连接。

### 协议数据格式

SSE 事件使用简单的文本格式，每条消息以双换行符（`\n\n`）结尾：

```
event: message
id: 1
data: {"text": "Hello, SSE!"}

event: update
data: {"status": "processing", "progress": 50}

event: done
data: {"status": "completed"}
```

- `data:` — 消息体，可多次出现表示多行数据
- `event:` — 事件类型名称（客户端可按类型监听）
- `id:` — 消息 ID，用于断线重连时定位最后收到的消息
- `retry:` — 客户端重连间隔（毫秒）

### 客户端代码示例

```javascript
// 使用浏览器原生 EventSource API
const eventSource = new EventSource('/api/stream');

// 监听默认消息
eventSource.onmessage = (event) => {
    console.log('收到消息:', JSON.parse(event.data));
};

// 监听特定事件类型
eventSource.addEventListener('update', (event) => {
    console.log('更新事件:', JSON.parse(event.data));
});

// 监听连接状态
eventSource.onopen = () => console.log('SSE 连接已建立');
eventSource.onerror = (error) => console.error('SSE 连接错误:', error);

// 手动关闭连接
// eventSource.close();
```

### 服务端代码示例（FastAPI）

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

async def event_stream():
    """生成 SSE 事件流"""
    for i in range(1, 6):
        yield f"event: message\n"
        yield f"id: {i}\n"
        yield f"data: {json.dumps({'step': i, 'message': f'处理进度 {i}/5'})}\n\n"
        await asyncio.sleep(1)  # 模拟异步处理
    
    # 发送完成信号
    yield f"event: done\n"
    yield f"data: {json.dumps({'status': 'complete'})}\n\n"

@app.get("/api/stream")
async def stream():
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 禁用缓冲
        },
    )
```

### SSE vs WebSocket 对比

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 单向（服务器到客户端） | 双向（客户端与服务器） |
| 协议 | 基于 HTTP（升级或不升级） | 需要协议升级（WebSocket Protocol） |
| 自动重连 | 内置支持 | 需手动实现 |
| 事件类型 | 支持命名事件 | 仅消息帧 |
| 浏览器支持 | 所有现代浏览器 | 所有现代浏览器 |
| 适用场景 | 新闻推送、实时通知、流式输出 | 聊天应用、在线游戏、协作编辑 |

### 在 RAG 系统中的应用

SSE 是 RAG 系统中实现**流式回答**的理想选择。当 LLM 逐词生成回答时，后端通过 SSE 将每个生成的 token 实时推送到前端，用户无需等待完整回答即可看到内容逐步呈现，大幅提升交互体验。结合 FastAPI 的异步能力和 Chroma 的检索速度，可以构建低延迟的流式 RAG 问答系统。
