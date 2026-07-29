# 项目亮点 —— 面试话术

> 面试前 10 分钟看一遍，每条 30 秒内讲完。

## 一句话描述

基于 FastAPI + Chroma + Redis 的 RAG 智能知识库问答系统，
支持文档解析、向量检索、多轮对话记忆、SSE 流式输出，
并通过 MCP 协议包装为 Agent 可调用的工具集合。

---

## 五个核心亮点

### 1. RAG 检索增强生成闭环

- 自研递归文本切分器：按 `\n\n` → `\n` → `。` → `.` → 空格 优先级切分，带 overlap 窗口，中英文混合友好
- SHA-256 Hash 生成唯一 ID：支持幂等入库（同一文档重复入库自动覆盖，不产生脏数据）
- Chroma 向量检索 + distance 阈值兜底 + 来源多样性去重（每文档最多 3 条）
- 完整管道：文档解析 → 切分 → Embedding → 入库 → 检索 → Prompt 拼装 → LLM 生成
- 单元测试覆盖：pytest 28 条用例，覆盖切分器各分隔符降级、overlap、空输入、中英混合等场景

### 2. 流式架构 + 断连防护

- aiohttp 异步逐块读取 LLM 响应 → buffer 拼行 → JSON 过滤 → StreamingResponse
- 双层 finally（LLM 连接层 + 流式生成器层）+ CancelledError 捕获
- 客户端断开时正确关闭上游连接，不泄漏资源
- 同步 DB 操作通过 `asyncio.to_thread()` 投入线程池，不阻塞事件循环

### 3. MCP 协议工具化（区别于普通 CRUD 项目的关键亮点）

- 把 RAG 系统的向量检索 + 文档入库能力包装为 MCP Server
- Claude Code 可直接调用 `search_knowledge_base` / `ingest_file` / `list_sources` 三个工具
- 通过 stdio 通信，在 `.mcp.json` 中配置，Claude Code 启动后自动连接
- 体现了「Agent 是大脑，MCP 是手脚」的架构理解——这是 AI 智能体应用层的核心范式

### 4. 多轮对话记忆

- Redis 短期记忆管道：LPUSH 存入 + EXPIRE 30 分钟自动过期
- FastAPI lifespan 管理 Redis 连接池生命周期
- 同时写入 SQLite 做持久化，支持 `/history/{user_id}` 查询

### 5. 工程化实践

- 代码审查：自定义 Claude Code Skill（`.claude/skills/code-review.md`），按 SSE 异步、错误处理、资源管理、安全、日志五维度审查
- 测试覆盖：`text_splitter` 15 条 + `prompts` 13 条，覆盖正常路径 + 边界 + 异常
- RotatingFileHandler 日志（500KB 滚动）、pytest、git 分支开发

---

## 面试官可能追问 & 我的回答

### Q: MCP 是什么？你在项目里怎么用的？
A: MCP 全称 Model Context Protocol，是 Anthropic 发布的开放协议，定义了 AI 应用怎么去发现和调用外部工具。
在我的项目里，我把 RAG 知识库的检索和入库能力封装成了 MCP Server，暴露了三个工具：语义搜索、文档入库、列出数据源。
Claude Code 通过 stdio 与它通信，能直接搜索我的知识库、导入文档。
可以理解为给 AI Agent 装了一个统一的工具接口——跟 USB 一样，不管是键盘还是鼠标，插上就能用。

### Q: MCP 和传统的 REST API 有什么区别？
A: 三个核心区别：
1. 通信方式：REST 走 HTTP，MCP 走 stdio（标准输入输出），Agent 直接以子进程方式启动 MCP Server
2. 发现机制：MCP 有 `tools/list` 让 Agent 自动发现有哪些工具可用，REST 需要提前知道端点
3. 设计目标：REST 是给人用的 Web API，MCP 是给 AI Agent 用的工具协议——它定义了统一的 Tool 描述格式和 `tools/call` 语义

### Q: 为什么选 MCP 而不是直接用 Function Calling？
A: Function Calling 是单个模型厂商的工具调用格式（比如 OpenAI 的 function calling、Anthropic 的 tool use）。
MCP 解决的是更上层的问题：工具怎么被多个 Agent 发现和复用。
我的 MCP Server 可以同时被 Claude Code、Claude Desktop、以及其他支持 MCP 的 Agent 调用——不需要为每个 Agent 重新写一遍工具定义。
这就是协议的价值：一次封装，到处能用。

### Q: 为什么用 Chroma 而不是 Faiss / Milvus？
A: Chroma 轻量、本地持久化、Python 原生支持好、不需要额外部署服务。
对于我这个项目规模（几千到几万条文档）足够。如果生产环境数据量大，可以考虑 pgvector（和 PostgreSQL 一体）或 Milvus（分布式）。

### Q: Redis 会话过期时间为什么是 30 分钟？
A: 用户对话通常不会跨半小时，30 分钟足够覆盖一次完整的问答交互，同时避免内存被长期不活跃的会话占满。这是一个经验值，具体可以根据业务调整。

### Q: 你这个项目最大的技术难点是什么？
A: 两个。
一是 SSE 流式管道中断连处理——客户端断开后 aiohttp 连接还在，需要 CancelledError + 双层 finally 兜底，还得确保同步 DB 操作不阻塞事件循环（用 asyncio.to_thread 解决）。
二是 MCP Server 的工具设计——怎么把已有的 RAG 能力合理地拆成原子工具，让 Agent 能正确选择和组合调用。拆太细 Agent 调用次数多，拆太粗工具不够灵活。

---

## 项目速览

| 维度 | 数据 |
|------|------|
| 开发周期 | 18 天 + 7 天冲刺包装 |
| 技术栈 | FastAPI + Chroma + Redis + SQLAlchemy + MCP + DeepSeek API + SSE |
| 代码行数 | ~2000+ 行 Python |
| 测试覆盖 | pytest 28 条（切分器 + Prompt） |
| MCP 工具 | 3 个（search / ingest / list） |
| 前端 | 原生 chat.html（fetch + ReadableStream） |

---

## 简历项目描述（直接复制粘贴）

```
项目名：AI 智能知识库问答系统（RAG Chat + MCP Server） | 独立开发 | 2026.07

技术栈：FastAPI + Chroma + Redis + SQLAlchemy + MCP + DeepSeek API + SSE

项目描述：
基于 RAG 架构的智能问答系统，支持文档入库、向量语义检索、多轮对话记忆与
流式响应。通过 MCP 协议将知识库能力包装为 Agent 可调用的工具集合，
使 Claude Code 等 AI Agent 能直接操作知识库。

核心工作：
· 自研递归文本切分器，支持优先分隔符策略与重叠窗口，保证语义完整性
· 集成 Chroma 向量库实现余弦相似度检索，设计 distance 阈值 + 来源多样性去重
· 搭建 Redis 短期记忆管道（LPUSH + EXPIRE），实现 30 分钟自动过期的多轮对话上下文
· 实现 SSE 流式响应，双层 finally + CancelledError 确保客户端断连时资源正确释放
· 编写 MCP Server，将 RAG 检索/入库能力暴露为 Agent 可调用的标准化工具
· 编写单元测试（pytest 28 条），使用 Claude Code + 自定义 Skill 辅助开发全流程
```
