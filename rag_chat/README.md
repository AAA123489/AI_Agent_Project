# AI 智能知识库问答系统（RAG Chat）

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Chroma](https://img.shields.io/badge/Chroma-vector--db-FF6B6B?style=flat&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)
[![MCP](https://img.shields.io/badge/MCP-protocol-6E3FF3?style=flat&logo=anthropic&logoColor=white)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE)

基于 FastAPI + ChromaDB + DeepSeek 的 RAG 智能问答系统：爬取河南工学院官网公开信息 → 隐私脱敏 → 切分入库 → SSE 流式问答，支持 MCP 工具暴露。附带消融实验工具，可调参对比检索效果。

> ⚠️ 本系统为个人学习项目，非河南工学院官方应用。

## 核心能力

| 能力 | 说明 |
|------|------|
| 🕷️ 网站爬取 | aiohttp 异步爬虫，动易/JSP CMS 适配，SQLite 去重，限速模拟真人 |
| 🔒 隐私脱敏 | 删除个人手机号 / 邮箱，保留办公室座机（`sanitize_privacy`） |
| ✂️ 智能分块 | 自研递归切分器（段落→句子→空格），重叠窗口，参数可调 |
| 🔢 向量检索 | ChromaDB 余弦相似度检索，按文件名 + 年份 + 原文链接标注来源 |
| 💬 SSE 流式对话 | Agent 循环 + 工具调用，答案附 📎 参考来源（日期·文件名·🔗查看原文） |
| 🧠 多轮记忆 | Redis 短期记忆（可选），静默降级为前端会话历史 |
| 🧪 消融实验 | `crawl_ablation.py` 按量爬取 + `rebuild_kb.py` 参数化重建 |
| 🔧 MCP 工具 | 3 个工具暴露给外部 Agent（`mcp_server/`） |

## 技术栈

- **Web 框架**：FastAPI + Uvicorn（ASGI），SSE 流式
- **爬虫**：aiohttp + BeautifulSoup + lxml
- **向量库**：ChromaDB（本地持久化，paraphrase-multilingual-MiniLM-L12-v2 中文 Embedding）
- **LLM**：DeepSeek（Anthropic 兼容 Tool Use 格式）
- **记忆**：Redis（可选，LPUSH + EXPIRE 30 分钟）
- **前端**：原生 HTML/CSS/JS，`static/chat.html`

## 目录结构

```
rag_chat/
├── app_fastapi.py          # 【唯一入口】FastAPI SSE 后端，serve chat.html（端口 8000）
├── app_backend.py          # 核心逻辑：Agent 循环 / LLM 调用 / 工具 / 检索 / 隐私规则
├── redis_client.py         # Redis 短期记忆（可选）
├── src/
│   ├── vector_store.py     # Chroma 向量库封装（增删查、相似度检索）
│   ├── config.py           # .env 配置管理
│   └── logger.py           # 日志配置
├── campus_scraper/         # 爬虫包
│   ├── config.py           # 爬取分类 / 速率 / 日期过滤
│   ├── models.py           # ArticleMetadata 等数据模型
│   ├── storage.py          # SQLite 已爬 URL 去重
│   ├── html_parser.py      # HTML → 纯文本（动易/JSP CMS 适配）
│   ├── scraper.py          # aiohttp 异步爬虫（分页、限速、重试）
│   └── pipeline.py         # 爬取→脱敏→落盘→分块→嵌入→ChromaDB
├── text_splitter.py        # 自研递归切分器 + sanitize_privacy 隐私脱敏
├── document_parser.py      # PDF / DOCX 解析
├── rag_pipeline.py         # 单文档入库流水线
├── crawl_ablation.py       # 消融实验：按分类 + 条数爬取并脱敏落盘
├── rebuild_kb.py           # 消融实验：从 scraped_docs 按参数重建知识库
├── mcp_server/             # MCP 协议服务（暴露给外部 Agent）
├── static/chat.html        # 聊天前端（SSE 流式 + 参考来源卡片）
├── tests/                  # pytest 测试
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 环境准备

```bash
cd rag_chat
pip install -r requirements.txt
cp .env.example .env    # 填入 DeepSeek API Key
```

`.env` 配置：

```ini
API_KEY=你的DeepSeek_API密钥
API_URL=https://api.deepseek.com/anthropic/v1/messages
MODEL_NAME=deepseek-v4-flash
TEMPERATURE=0.3
REDIS_URL=redis://localhost:6379
```

> `.env` 已被 `.gitignore` 排除，不会提交。

### 2. 启动服务

```bash
python app_fastapi.py
```

访问 http://127.0.0.1:8000 即聊天前端；健康检查 http://127.0.0.1:8000/health 返回 `kb_chunks`。

### 3. 构建知识库（三选一）

- **全量爬取**：前端管理面板点「刷新爬虫」，或调用 `POST /scrape/start`（每类最多 N 页，较慢）
- **按量实验爬取**：`python crawl_ablation.py 通知公告 1 10`（分类 + 页数 + 最多几篇，自动脱敏落盘到 `scraped_docs/`）
- **上传文档**：前端「上传文档」或 `POST /upload`

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天前端页面 |
| GET | `/health` | 健康检查（含 `kb_chunks`） |
| POST | `/chat` | SSE 流式对话（Body：`user_id` / `message` / `history`） |
| POST | `/upload` | 文档上传入库（multipart `file`） |
| POST | `/scrape` | 触发爬虫（确认） |
| POST | `/scrape/start` | 实际执行爬虫 |
| GET | `/kb-stats` | 知识库统计 |

`POST /chat` 请求示例：

```json
{ "user_id": "test", "message": "河南工学院诚聘高层次人才有哪些待遇？" }
```

响应为 SSE 流：`thinking` / `sources` / `text` / `done` 四类事件，前端据此渲染思考过程、参考来源和流式答案。

## 隐私保护

- **入库前脱敏**：`sanitize_privacy` 删除个人手机号（`1[3-9]\d{9}`）和邮箱，保留办公室座机
- **回答规则**：SYSTEM_PROMPT 禁止提供个人联系方式，统一引导至官网；官方办公电话只能依据知识库原文回答
- **前端免责声明**：标注"个人学习项目，非官方应用"

## 消融实验指南

通过对比实验找出最优参数组合。**一次只改一个参数，用同一组问题对比**。

### 参数清单

| 参数 | 默认 | 改哪 | 生效方式 |
|------|------|------|---------|
| 分块大小 | 500 | `rebuild_kb.py --chunk-size` | 重建知识库 |
| 重叠窗口 | 50 | `rebuild_kb.py --chunk-overlap` | 重建知识库 |
| top_k 条数 | 5 | `app_backend.py` `_search_knowledge_base` 默认值 | 重启 |
| 生成温度 | 0.3 | `.env` 的 `TEMPERATURE` | 重启 |
| 回答上限 | 1000 | `app_backend.py` payload `max_tokens` | 重启 |
| 模型 | flash | `.env` 的 `MODEL_NAME` | 重启 |

### 流程

```bash
# 1. 重建知识库（改分块参数）
python rebuild_kb.py --chunk-size 300 --chunk-overlap 50

# 2. 启动服务，问同一组固定问题，对比参考来源与回答
python app_fastapi.py

# 3. 换参数 → 回到第 1 步（top_k / 温度 / 模型无需重建，改后重启即可）
```

### 实验提示

- 每次重建会**清空** ChromaDB，库中始终只有最新一组参数的结果
- 重建后通过 `/health` 的 `kb_chunks` 数量确认库已更新
- **对话日志按实验组分文件**：改参数后同步修改 `.env` 的 `EXPERIMENT_TAG`（如 `chunk300_overlap50`），问答记录会写入 `chat_logs_{标签}.txt`；每条记录自动附带当前模型/温度/回答上限/top_k 参数，即使忘改标签也能从内容分辨
- `CHUNK_OVERLAP` 效果低频触发（仅当块边界切断句子时显形），建议配合小 `--chunk-size`（如 200）对比
- 温度是唯一带随机性的参数，同一参数下多次提问结果会抖动，应看风格趋势而非逐字对比
- 找到最优分块参数后，同步到 `campus_scraper/pipeline.py` 的 `CHUNK_SIZE` / `CHUNK_OVERLAP`，全量爬取才会沿用

### 实验结果（2026-08-07）

测试集：6 道数字型固定题（60 分制），由外部 LLM（千问）按基准事实表逐题打分。贪心/累积消融，每步锁定当前最优再消融下一个参数。

| 阶段 | 配置 | 得分/60 |
|------|------|--------|
| ① 分块大小 | 300/50/8 | **46** 🏆 |
| | 500/50/5（基线） | 44 |
| | 800/50/3 | 32 |
| | 200/50/12 | 30 |
| | 1000/50/2 | 30 |
| ② 重叠窗口（chunk=300 固定） | 50 | **46** 🏆 |
| | 30 | 42 |
| | 60 | 36 |
| | 90 | 24 |
| | 120 | 8 |
| ③ top_k（300/50 固定） | 8 | **46** 🏆 |
| | 3 | 22 |
| | 5 | 0（疑似测试事故，相邻配置均 22~46，待重测） |
| ④ 温度（300/50/8 固定） | 0.3 | **46** 🏆 |
| | 0.4 | 36 |
| | 0.2 | 30 |
| | 0.1 | 28 |
| | 1.0 | 12 (20%) |

**最终结论：`CHUNK_SIZE=300 / CHUNK_OVERLAP=50 / top_k=8 / TEMPERATURE=0.3`，46/60 分。**
已同步到 `campus_scraper/pipeline.py`（全量爬取沿用）；`TEMPERATURE=0.3` 本就是 `.env` 默认值，无需改动。

关键发现：
- **中偏小块最优**（300 > 500 > 800≈200≈1000）：小块语义细、信息密度高，配合多取（top_k=8）能拼出完整答案
- **重叠窗口是"越大越崩"**：小块 + 大 overlap → 块间大量重复内容 → 检索信噪比急剧下降，overlap≥90 时得分率腰斩
- **top_k 与块大小强耦合**：小块需要多取补全信息，top_k=8 是 300 字块的甜点
- **温度 0.3 为甜点**：0.1~0.4 区间内 0.3 是峰值（46），两端回落（0.4=36 / 0.2=30 / 0.1=28），1.0 全开随机性在数字题上崩盘（12 分）——数字型测试题对高温极敏感，要求精确数值时务必用低温

## MCP 工具

`mcp_server/` 通过 MCP 协议暴露：知识库检索 / 文档列表 / 文档入库，供项目二（手写 Agent）等外部 Agent 调用。

## 测试

```bash
python -m pytest tests/ -v
```

## 许可证

MIT
