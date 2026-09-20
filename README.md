# RAG 智能知识库问答系统

> **本仓库的主体是「项目一：RAG 知识库问答系统」**，代码位于 [`rag_chat/`](rag_chat/)——
> 基于 FastAPI + ChromaDB + DeepSeek 的检索增强问答后端，把河南工学院官网 1195 篇公开文档
> 变成**带拒答能力**的知识库：不知道就说不知道，而不是编造。

![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Tests](https://img.shields.io/badge/tests-118%20passed-brightgreen) ![RAG命中率](https://img.shields.io/badge/检索命中率-95%25%2B-brightgreen)

> ⚠️ **项目二 / 项目三已移出本仓库**，各自维护在独立仓库。本 README 末尾保留它们的
> 架构说明作为索引，但运行命令不在本仓库内。

## 三个项目的关系

```
项目一（RAG 知识库）  ← 本仓库
    │
    ├── MCP 协议 ──→ 项目二（手写 Agent）调用知识库
    │
    └── 直接导入 ──→ 项目三（LangGraph 多 Agent）调用知识库

项目二 ── 框架升级 ──→ 项目三
（手写循环）              （状态机）
```

| 项目 | 定位 | 技术栈 | 测试 | 位置 |
|------|------|--------|------|------|
| **项目一** | RAG 知识库问答系统 | FastAPI · ChromaDB · DeepSeek · Redis · SSE · LangGraph | 118 条 | ✅ **本仓库**（[`rag_chat/`](rag_chat/)） |
| 项目二 | 手写 Agent 工作流引擎（事件驱动） | Python · asyncio · SSE · MCP | 22 条 | 独立仓库 |
| 项目三 | LangGraph 多 Agent 编排引擎 | LangGraph · DeepSeek · ChromaDB · Redis | 38 条 | 独立仓库 |

## 项目结构

```
AI_Agent_Project/
├── rag_chat/          # 项目一：RAG 知识库问答系统（本仓库主体）
└── README.md
```

---

## 项目一：RAG 知识库问答系统

基于 **FastAPI + ChromaDB + DeepSeek** 的检索增强生成（RAG）问答后端，对校园公开文档做知识库问答。

### 核心能力

| 能力 | 实现 |
|------|------|
| 文档解析 | TXT / Markdown / PDF 全格式解析 |
| 智能分块 | 自研递归切分器：中文标点分隔 + 重叠窗口，消融实验确定最优块大小（`CHUNK_SIZE=300 / OVERLAP=50`） |
| 混合检索 | 向量 + BM25 双路召回 + RRF 融合，长尾召回 2/3 字 n-gram + 精确匹配兜底 |
| 可选重排 | `bge-reranker-base` 重排（默认关，遇表格行值题可开，代价每问 +1.9s） |
| 流式对话 | SSE 流式推送（`thinking / sources / text / done / error` 五类事件） |
| 召回自检 | LangGraph 状态机：判定召回的片段能否支撑回答，不能则**拒答**而非编造（硬负例拒答 0/6 → 6/6） |
| 多轮记忆 | Redis LPUSH + EXPIRE 30 分钟过期 |
| 库外拦截 | 问外校（如"河北工学院分数线"）**直接拒答不检索**；**两层**校验——检索层看工具参数 + Agent 层看用户原话，堵住「LLM 把校名改写掉」的绕过路径 |
| 首句净化 | 扣住模型发起工具调用前的英文旁白，不让它流到用户屏幕（`src/stream_gate.py`） |
| 检索增强 | 查询级缓存、年份过滤、文本去重、罕见词精确兜底、同文档补块 |
| MCP 工具 | 3 个工具暴露给外部 Agent 调用 |
| 安全 | X-API-Key 鉴权（hmac 恒时比较）、历史长度/字数上限校验 |

### 评测成绩单

| 评测 | 结果 |
|------|--------|
| 6 题基准（60 分制，2026-09-16 复跑） | **60/60** |
| 召回自检探测（硬负例 6 / 无关题 4 / 正例 5） | 自检 **off**：硬负例拒答 **0/6**；**on**：硬负例 **6/6**、无关题 **4/4**、正例 **5/5** 零误拒 |
| 25 题随机 | **96%** |
| 50 题随机 | **100%** |
| 100 题随机（回归后） | **95%** |

知识库规模：**1195 篇文档 / 13 分类**。

> ⚠️ **60/60 是单次采样，不是稳定性质。** 2026-09-17 用**同一份代码**跑两遍得到 60/60 与 50/60，
> 抖动来自「回答 + 打分」两级 LLM（根因是回答级 LLM 某次把工具参数的校名改写掉了，
> 该路径已修，见上表「库外拦截」行）。**衡量检索层改动请用 `eval_guard_probe.py`（检索层、确定性），不要用 60/60。**
>
> 25/50/100 题那三套的题库与脚本从未入库、已彻底丢失，无法复跑（成绩仅存于表内）。

### 运行

```bash
cd rag_chat
python app_fastapi.py      # http://localhost:8000
python -m pytest tests/ -v # 118 条用例
```

📖 详细文档见 [rag_chat/README.md](rag_chat/README.md)

---

## 项目二：Agent 工作流引擎（独立仓库）

为理解 Agent 底层机制，**不依赖 LangChain 等框架**手写的 Agent 循环（事件驱动架构）。

### 核心能力

- 手写 for-step 事件循环 + 条件路由，将「思考-行动-观察」完全解耦
- 6 种事件类型（Thinking / ToolCall / ToolResult / Text / Done / Error）
- 7 个内置工具（时间 / 文件 / RAG / 搜索 / 天气），通过 MCP 协议暴露
- SSE + Rich 双端流式输出（TTY 自适应）
- pytest 22 条用例

> 该项目位于独立仓库，未包含在本仓库内。其架构设计由**项目三**以 LangGraph 状态机形式框架化升级。

---

## 项目三：LangGraph 多 Agent 编排引擎

基于 **LangGraph 状态机**实现「**Router 意图路由 + 三子代理**」多 Agent 架构，对接项目一知识库。

### 架构

```
                  ┌────────────────────┐
   用户问题 ──► router  │   Router 节点      │
                  │ (LLM 意图分类) │  ── kb / calc / chat
                  └────────────────────┘
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
      ┌───────────┐  ┌───────────┐  ┌────────────┐
      │ kb_agent   │  │calc_agent │  │ chat_agent │
      │ 知识库检索 │  │ 安全计算   │  │ 普通对话    │
      └───────────┘  └───────────┘  └────────────┘
        工具隔离         工具隔离        工具隔离
```

- **Router 意图路由**：LLM 分类 `kb / calc / chat`，识别不出走 chat 兜底，不拒绝用户
- **三子代理**：kb（知识库检索）/ calc（安全计算）/ chat（闲聊），各为独立 ReAct 子图，工具集按角色隔离
- 每个子代理内部：`START → agent → [should_continue] → tools → agent → ... → END`

### 核心能力

| 能力 | 实现 |
|------|------|
| 状态机 | LangGraph StateGraph（父图 + 子图嵌套 + 条件路由） |
| LLM 协议 | Anthropic Messages API（Tool Use 格式，DeepSeek 兼容接口） |
| 工具隔离 | 每个子代理只暴露自己的工具集，减少误用面 |
| 主动工作记忆 | Agent 通过 `save_memory / search_memory / clear_memory` 自己决定存什么、查什么（Redis，按会话隔离，默认保留 7 天） |
| Web 界面 | FastAPI + SSE **节点级流式**，前端实时可见「Router 分派 → 子代理推理 → 工具调用」全过程 |
| 安全计算 | AST 白名单，拒绝任意代码执行 |
| 防死循环 | `MAX_ROUNDS = 10` 兜底 |
| 鉴权 | X-API-Key（hmac 恒时比较） |

### 主动工作记忆

Agent 不依赖外部代码「被动喂历史」，而是通过工具**自己决定**记住什么、回答前查什么：

- 用户透露偏好（姓名、校区、年级…）→ Agent 主动 `save_memory`
- 回答涉及用户之前提过的信息 → 先 `search_memory`
- 用户要求忘记 → `clear_memory`

存储走 Redis（复用项目一），按会话隔离（Web 下并发用户互不串扰），默认保留 7 天（`.env` 配 `MEMORY_TTL` 可调）。Redis 挂掉时记忆工具降级返回提示，不影响 Agent 主流程。

### 工具列表

| 工具 | 说明 | 归属子代理 |
|------|------|-----------|
| `search_knowledge_base` | 向量语义检索，对接项目一 ChromaDB | kb |
| `list_knowledge_sources` | 列出知识库已有文档及块数 | kb |
| `calculate` | 安全数学表达式计算（AST 白名单） | calc |
| `save_memory` | 主动保存一条工作记忆（如用户偏好） | kb / chat |
| `search_memory` | 检索之前保存的工作记忆 | kb / chat |
| `clear_memory` | 删除记忆（'*' 清空全部） | kb / chat |

### 运行

该项目已移出本仓库（独立仓库，代码不在 `AI_Agent_Project/` 内），本仓库不提供可执行命令。
它通过 `RAG_CHAT_DIR` 指向本仓库的 `rag_chat/` 复用知识库与前端。

### 配置（`.env`，优先读 rag_chat/.env）

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_KEY` | ✅ | DeepSeek API 密钥 |
| `API_URL` | ❌ | Anthropic 兼容接口（默认 DeepSeek） |
| `MODEL_NAME` | ❌ | 模型名称 |
| `REDIS_URL` | ❌ | 记忆 / 对话持久化（默认 redis://localhost:6379） |
| `MEMORY_TTL` | ❌ | 工作记忆保留秒数（默认 7 天） |
| `WEB_PORT` | ❌ | Web 服务端口（默认 8001） |

📖 详细文档见该项目的独立仓库。

---

## 快速开始

```bash
# 1. 克隆仓库（Python 3.12+）
git clone https://github.com/AAA123489/AI_Agent_Project.git
cd AI_Agent_Project/rag_chat

# 2. 装依赖 + 配密钥
pip install -r requirements.txt
cp .env.example .env          # 填入 DeepSeek API Key

# 3. 构建知识库（三选一）
python crawl_ablation.py 通知公告 1 10   # ① 按量爬取（分类 + 页数 + 最多几篇）
# ② 前端管理面板点「刷新爬虫」，或 POST /scrape/start（全量，较慢）
# ③ 前端「上传文档」，或 POST /upload（TXT / MD / PDF / DOCX）

# 4. 启动
python app_fastapi.py         # http://localhost:8000

# 5. 跑测试（118 条，零网络）
python -m pytest tests/ -v
```

> 环境变量参考 [`rag_chat/.env.example`](rag_chat/.env.example)——`API_KEY` 必填；
> Redis 可选（不装则多轮记忆静默降级为前端会话历史）。

## License

MIT
