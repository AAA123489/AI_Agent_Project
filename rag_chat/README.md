# AI 智能知识库问答系统（RAG Chat）

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Chroma](https://img.shields.io/badge/Chroma-vector--db-FF6B6B?style=flat&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE)

基于 FastAPI + ChromaDB + DeepSeek 的 RAG 智能问答系统：爬取河南工学院官网公开信息 → 隐私脱敏 → 切分入库 → SSE 流式问答。附带消融实验工具，可调参对比检索效果。

> ⚠️ 本系统为个人学习项目，非河南工学院官方应用。

## 核心能力

| 能力 | 说明 |
|------|------|
| 🕷️ 网站爬取 | aiohttp 异步爬虫，动易/JSP CMS 适配，SQLite 去重，限速模拟真人 |
| 🔒 隐私脱敏 | 删除个人手机号 / 邮箱，保留办公室座机（`sanitize_privacy`） |
| ✂️ 智能分块 | 自研递归切分器（段落→句子→空格），重叠窗口，参数可调 |
| 📄 网页解析 | HTML → 纯文本（动易/JSP CMS 适配，按栏目配选择器）；爬虫是唯一入库通路 |
| 🔢 混合检索 | 向量 + BM25 双路召回 + **RRF 名次融合**（`k=60`）；年份过滤 / 文本去重 / 罕见词精确兜底 / 同文档补块 |
| 🕐 相对时间 | 「昨天 / 最近 / 上周 / 本月」先确定性解析成日期区间再检索（`src/time_range.py`），泛问「最近有什么通知」按日期倒序列出；不再让模型拿旧文冒充昨天 |
| 🛡️ 召回自检 | 判定召回是否支撑回答，不支撑则**拒答**而非编造；不充分时改写 query 重检（`src/recall_guard.py`） |
| 💬 SSE 流式对话 | Agent 循环 + 工具调用，答案附 📎 参考来源（日期·文件名·🔗查看原文） |
| 🧹 首句净化 | 扣住模型发起工具调用前的英文旁白（`src/stream_gate.py`），不让它流到用户屏幕 |
| 🧠 多轮记忆 | Redis 短期记忆（可选），静默降级为前端会话历史 |
| 🧪 消融实验 | `crawl_ablation.py` 按量爬取 + `rebuild_kb.py` 参数化重建 |

## 技术栈

- **Web 框架**：FastAPI + Uvicorn（ASGI），SSE 流式
- **爬虫**：aiohttp + BeautifulSoup + lxml
- **向量库**：ChromaDB（本地持久化，paraphrase-multilingual-MiniLM-L12-v2 中文 Embedding）
- **编排**：全手写（ReAct 工具循环 + 召回自检判定循环），不依赖编排框架
- **LLM**：DeepSeek（Anthropic 兼容 Tool Use 格式）
- **记忆**：Redis（可选，LPUSH + EXPIRE 30 分钟）
- **前端**：原生 HTML + ES Modules + Tailwind CSS v4（CLI 构建），无 React
- **账号 / 日志**：SQLite（`db.py`）+ bcrypt 密码哈希 + HMAC 签名 HttpOnly Cookie

## 目录结构

```
rag_chat/
├── app_fastapi.py          # 【唯一入口】FastAPI SSE 后端，serve chat.html（端口 8000）
├── app_backend.py          # 核心逻辑：Agent 循环 / LLM 调用 / 工具 / 检索 / 隐私规则
├── redis_client.py         # Redis 短期记忆（可选）
├── src/
│   ├── vector_store.py     # Chroma 向量库封装（增删查、相似度检索）
│   ├── recall_guard.py     # 召回自检（判定循环 + 改写重检环，全手写）
│   ├── stream_gate.py      # 首句净化：扣住工具调用前的英文旁白
│   └── config.py           # .env 配置管理
├── campus_scraper/         # 爬虫包
│   ├── config.py           # 爬取分类 / 速率 / 日期过滤
│   ├── models.py           # ArticleMetadata 等数据模型
│   ├── storage.py          # SQLite 已爬 URL 去重
│   ├── html_parser.py      # HTML → 纯文本（动易/JSP CMS 适配）
│   ├── scraper.py          # aiohttp 异步爬虫（分页、限速、重试）
│   └── pipeline.py         # 爬取→脱敏→落盘→分块→嵌入→ChromaDB
├── text_splitter.py        # 自研递归切分器 + sanitize_privacy 隐私脱敏
├── crawl_ablation.py       # 消融实验：按分类 + 条数爬取并脱敏落盘
├── rebuild_kb.py           # 消融实验：从 scraped_docs 按参数重建知识库
├── eval_baseline.py        # 评测：6 题基准真实链路（检索 + LLM）
├── eval_score.py           # 评测：60 分制打分器
├── eval_guard_probe.py     # 评测：召回自检能力探测（硬负例 / 无关题 / 正例）
├── eval_verify.py          # 评测：规则核对答案事实是否落在检索来源里
├── stress_retrieval.py     # 压测：检索层并发（不调 LLM）
├── auth.py                 # 注册 / 登录 / HMAC 会话 Cookie
├── db.py                   # SQLite：users / usage_daily / qa_audit
├── audit.py                # 问答审计采集（QATrace）与查询导出
├── web/
│   ├── login.html          # 登录 / 注册页（仅经路由提供）
│   └── chat.html           # 聊天页（仅登录后经 / 提供）
├── static/
│   ├── js/                 # 前端 ES Modules（api/store/render/sidebar/chat/main/login）
│   ├── css/tailwind.css    # Tailwind 构建产物（需提交）
│   ├── vendor/             # marked / lucide 本地托管（需提交）
│   └── admin.html          # 管理面板（库统计 / 爬虫 / 问答审计）
├── assets/tailwind.input.css  # Tailwind 源文件
├── scripts/copy-vendor.mjs    # 复制 marked / lucide UMD 到 static/vendor
├── tests/                  # pytest 测试
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 环境准备

```bash
cd rag_chat
pip install -r requirements.txt
npm install && npm run build   # 生成 static/css/tailwind.css 并复制 vendor
cp .env.example .env    # 填入 DeepSeek API Key，并设置 SECRET_KEY
```

`.env` 关键配置：

```ini
API_KEY=你的DeepSeek_API密钥
API_URL=https://api.deepseek.com/anthropic/v1/messages
MODEL_NAME=deepseek-v4-flash
TEMPERATURE=0.3
REDIS_URL=redis://localhost:6379

# 会话签名密钥（必填，否则重启后所有登录失效）
SECRET_KEY=用 python -c "import secrets; print(secrets.token_urlsafe(48))" 生成
DAILY_MESSAGE_QUOTA=50     # 每账号每日提问上限，0 = 不限
REGISTER_INVITE_CODE=      # 留空 = 任何人可注册
AUDIT_RETENTION_DAYS=90    # 问答审计保留天数，超期自动清理；0 = 永久保留
```

> `.env` 已被 `.gitignore` 排除，不会提交。`data/rag_app.db` 含账号与问答原文，同样已被忽略。

### 2. 启动服务

```bash
python app_fastapi.py
```

访问 http://127.0.0.1:8000 → 未登录会跳转 `/login`，注册（用户名 2-10 个英文字母、密码至少 8 位）后进入聊天页。健康检查 http://127.0.0.1:8000/health 返回 `kb_chunks`。

### 3. 构建知识库（三选一）

- **全量爬取**：前端管理面板点「刷新爬虫」，或调用 `POST /scrape/start`（每类最多 N 页，较慢）
- **按量实验爬取**：`python crawl_ablation.py 通知公告 1 10`（分类 + 页数 + 最多几篇，自动脱敏落盘到 `scraped_docs/`）
- **自动更新**：默认开启，服务运行期间每天 10:00 自动增量爬取一次；距上次成功爬取超过 `SCRAPE_CATCHUP_DAYS` 天时，启动后还会补跑一次。配置见下表。

### 自动更新知识库（定时增量爬取）

**默认开启**。服务运行期间每天到点自动跑一次「爬取 → 入库 → 清查询缓存」，不用人工干预。

爬虫本身是**增量**的（`campus_scraper/storage.py` 按 URL 去重），重复跑不会重复入库，只捞新增文章——所以定时跑是安全的，不会撑大知识库、也不需要清库重建。

| 变量 | 默认 | 说明 |
|------|------|------|
| `SCRAPE_SCHEDULE_ENABLED` | `on` | `off` 关闭定时更新 |
| `SCRAPE_SCHEDULE_HOUR` / `_MINUTE` | `10` / `0` | 每天触发时刻（本地时间） |
| `SCRAPE_SCHEDULE_PAGES` | `1` | 日常每类**最多**爬几页（上限，非固定成本） |
| `SCRAPE_CATCHUP_PAGES` | `3` | 启动补跑每类最多爬几页 |
| `SCRAPE_CATCHUP_DAYS` | `3` | 距上次成功爬取超过这么多天，启动后补跑一次；`0` = 关闭补跑 |

**页数是上限，不是固定成本。** 列表页是跨页严格新→旧的连续序列（实测：通知公告第 1/2/3 页分别是 `10-10~09-23` / `09-22~09-03` / `09-02~08-05`，无缝衔接），所以爬虫**一旦在某页遇到已爬过的 URL 就停止翻页**——说明时间线已经接上缓存，后面的页只会更旧。

这让日常增量跑完第 1 页就结束（约 3~5 分钟），省掉了其后每页 3~8 分钟的页间休息；只有真的积压时才会往后翻。

> **服务不常驻会怎样**：进程内定时器只在服务运行时触发。凌晨窗口若在关机状态，那一次就跳过——由 `SCRAPE_CATCHUP_DAYS` 补跑兜住：下次启动服务时若发现知识库太久没更新，会立刻补跑一次。所以**不必 7×24 开着**，隔几天开一次也能保持更新。
>
> 查看状态：`GET /scrape/status`（下次几点跑、上次什么时候跑的、当前是否在跑）。服务启动日志里也会打印「⏰ 下次自动更新知识库：N 小时后」。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天页；未登录 303 → `/login?next=/` |
| GET | `/login` | 登录 / 注册页；已登录 303 → `/` |
| GET | `/api/auth-config` | 公开：密码长度、是否需邀请码 |
| POST | `/api/register` | 注册（201，自动登录；重名 409） |
| POST | `/api/login` | 登录（失败统一 401「用户名或密码错误」） |
| POST | `/api/logout` | 清会话 Cookie |
| GET | `/api/me` | 当前用户 + 今日额度 |
| GET | `/health` | 健康检查（含 `kb_chunks`） |
| POST | `/chat` | SSE 流式对话（**需登录 Cookie**，Body：`message` / `history` / `conversation_id`） |
| POST | `/scrape` | 触发爬虫（确认，需 `X-API-Key`） |
| POST | `/scrape/start` | 实际执行爬虫（后台运行、立即返回；已在跑则 409） |
| GET | `/scrape/status` | 爬虫 / 自动更新状态（下次触发时刻、上次爬取时间、是否在跑） |
| GET | `/kb-stats` | 知识库统计（含 `unaccounted_chunks`） |
| GET | `/admin/audit` | 问答审计查询（需 `X-API-Key`；可按 user_id / 日期 / status 过滤） |
| GET | `/admin/audit/export` | 导出 JSONL / CSV（需 `X-API-Key`） |

`POST /chat` 请求示例（先登录拿到 `rag_session` Cookie）：

```json
{ "message": "河南工学院诚聘高层次人才有哪些待遇？", "conversation_id": "abc123" }
```

响应为 SSE 流：`thinking` / `sources` / `text` / `done` / `error` 事件，前端据此渲染思考过程、参考来源和流式答案。

> 用户身份只取自会话 Cookie；请求体里即使带 `user_id` 也会被忽略。每日额度默认 50 次/账号，超限返回 429。

### 账号与问答日志

- 用户表与问答审计同存 `data/rag_app.db`（SQLite），按 `user_id` 逻辑隔离。
- 每次问答落库：问题、答案、思考步骤、工具调用（参数 / 结果 / 耗时）、来源、错误、`request_id`、`conversation_id`、耗时。
- 审计接口需管理员 `X-API-Key`，普通用户 Cookie 无法访问；落库失败只记日志，不影响对话。
- **保留期限**：`AUDIT_RETENTION_DAYS` 默认 90 天（`0` = 永久）。服务启动时清一次，之后每 24 小时清一次；只删行，不自动 `VACUUM`。
- **管理页**：`/admin.html` 解锁后可查审计——按用户 / 日期 / 状态筛选、点行展开完整问答 / 思考 / 工具调用 / 来源、翻页、导出 CSV / JSONL。

### 首句净化（为什么回答开头不再是英文）

模型在发起工具调用前常先吐一句英文旁白（`I'll search the knowledge base for information about …`）。SSE 是**逐 delta 直发**的，等这一轮结束发现是工具轮时，英文早已显示在用户屏幕上——而这一轮的文本根本没进 `messages`（只 append 了 tool_use 块），它从头到尾都不属于回答。

`src/stream_gate.py` 的 `OpeningGate` 每轮扣住开头、直到出现**第一个汉字**：出汉字就把扣住的英文前缀丢掉、从汉字起原样放行（正文以汉字开头 → 流式几乎无延迟）；整轮没有汉字时，有 `tool_use` 就是旁白丢弃，没有就是正常英文回答、整段补发。改动前 6 题基准里 4~5 题的首句是这个旁白。

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

## 召回自检

检索不是「捞回 top_k 个块」就完事——**本校但没有的问题**（如「河南工学院食堂几点开门」）会捞回一堆无关块，LLM 拿到就会编。这一层判定「召回的片段到底能不能支撑回答」，不能就拒答。

手写判定循环（`src/recall_guard.py`，**不依赖编排框架**）：

```
循环：retrieve → grade ─┬─(sufficient)──────────────────→ format，返回
                        ├─(insufficient 且 attempt<max)──→ 下一圈（环，attempt+1）
                        └─(off_topic 或 attempt>=max)────→ refuse，返回
```

`grade` 两段式：**向量距离 ≤0.30 直接放行**（不调判官；基线 5 题里 4 题命中，Q2 运动会 0.307 会落到判官），其余交 LLM 判官给 `VERDICT: sufficient|insufficient|off_topic`。解析失败 / 判官异常**一律放行**——宁可少拒答，不可误拒答，更不可打断 SSE 流。

**开关**（`.env`，**默认开**——2026-10-16 由 `off` 翻为 `on`）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `RECALL_GUARD` | `on` | `off` 退回改造前行为（跑对照用，见下方命令） |
| `RECALL_GUARD_MAX_ATTEMPTS` | `1` | 检索轮数。`1`=不重检，`2`=允许一次改写重检（环跑一圈） |

> 为什么默认开：消融数据是支持开的——硬负例拒答 0/6 → 6/6、无关题 0/4 → 4/4、正例零误拒，
> 代价只是负例多约 0.8s（正例走快通过通道，不调判官）。原先那句"off 保持改造前行为"是
> 把它当**消融基线**，不是效果判断；写进论文的是它开着的行为，那就该默认开着。

**效果**（`eval_guard_probe.py`，硬负例 6 + 无关题 4 + 正例 5）：

| 配置 | 硬负例拒答 | 无关题拒答 | 正例放行 | 耗时（负例/正例） |
|---|---|---|---|---|
| `RECALL_GUARD=off` | **0/6** | 0/4 | 5/5 | 0.26s / 0.27s |
| `RECALL_GUARD=on` | **6/6** | 4/4 | 5/5 | 1.04s / 0.45s |

```bash
python eval_guard_probe.py --tag guard_on                      # 探测：硬负例 + 无关 + 正例（默认即 on）
RECALL_GUARD=off python eval_guard_probe.py --tag guard_off    # 对照：改造前行为（必须显式传，见下）
RECALL_GUARD=on python eval_guard_probe.py --answers           # 额外跑完整 AgentLoop，看最终回答
RECALL_GUARD=on RECALL_GUARD_MAX_ATTEMPTS=2 python eval_guard_probe.py --tag ring_on  # 消融：环开
```

> **跑 `off` 对照必须显式传环境变量。** `load_dotenv()` 默认**不覆盖**已存在的环境变量，
> 所以 `RECALL_GUARD=off` 写在命令前面必胜、写在 `.env` 里不行。
> 同理 `eval_baseline.py` 没有显式设这个开关，它现在跑的是**自检开**的条件——
> 旧那份 60/60 是 `guard=off` 时量的，要对齐得 `RECALL_GUARD=off python eval_baseline.py`。

**环默认关**是消融的结论，不是省事：干净负例上环开环关打平（环没收益），但在灰区题「招生办咨询电话」上环开会把一次正确的拒答翻成放行（两次独立运行复现）——判官第二轮不记得自己已判过不充分，对着同样无关的新片段重新判就成了 sufficient。代码保留可开关。详见 [docs/改进记录.md](docs/改进记录.md) 第 15 条。

**已知边界**：判官判的是**相关性**而非**可答性**，所以「主题词在库、具体事实不在库」的灰区题（如「现任校长是谁」，库里只有外单位领导的人名）会被放行。

## 评测

6 题基准（60 分制）可复跑，两个脚本：

| 文件 | 作用 |
|------|------|
| `eval_baseline.py` | 跑固定 6 题（真实链路：混合检索 + LLM），落盘每题回答与召回来源 |
| `eval_score.py` | 按「基准事实表 + 60 分制规则」调 DeepSeek 逐题打分，输出每题分与总分 |

```bash
python eval_baseline.py --tag topk8                              # → eval_results_baseline_topk8.json
python eval_score.py eval_results_baseline_topk8.json --out eval_score_topk8.md
```

- 检索参数可用命令行覆盖做消融：`--top-k N`、`--mode hybrid|vector`
- 基准事实表与评分规则内嵌在 `eval_score.py` 的 `FACT_TABLE` / `SCORING_PROMPT`，改题或改真值只动这一处
- 最近一次复跑：2026-09-16，hybrid / top_k=8 → **60/60**（第 6 题库外题正确拒答）

> ⚠️ **60/60 是单次采样，不是稳定性质。** 2026-09-17 用**同一份代码**跑了两遍，得到 **60/60 与 50/60**。当时那次的根因是这道题要过**两级 LLM**（回答 + 打分）：某次 LLM 把工具参数写成 `query='招生录取分数'`——**丢了校名**，库外闸门没东西可拦，返回的河南工学院分数被如实列进回答，评分 LLM 按「含基准表外数字即判幻觉」判 0。检索层两次完全一致，差别只在模型最后那段回答要不要多嘴抄数字。
> **这条路径已修**（`_guard_tool_call` 用**用户原话**再做一次库外主体校验，见 `docs/改进记录.md` 第 17 条），但「回答 + 打分」两级 LLM 本身仍有抖动。**要衡量检索层改动（如召回自检），用上面的 `eval_guard_probe.py`（检索层、确定性），不要用 60/60。**

> 25/50/100 题那三套随机评测的题库与脚本从未入库，已丢失、无法复跑。

## 测试

```bash
python -m pytest tests/ -v      # 196 条（test_recall_guard.py 召回自检、test_hybrid_retriever.py RRF 融合、test_stream_gate.py 首句净化、test_out_of_kb_guard.py 库外闸门、test_prompt_injection.py 注入防线、test_q6_rewrite_integration.py 假 LLM 端到端、test_time_range.py 相对时间解析、test_list_entry_regressions.py 列表页日期提取、test_real_page_snapshots.py 官网真实页面快照、test_html_parser_title.py 标题后缀截断、test_scraper_dead_link.py 死链页识别、test_eval_verify.py 事实核对器、test_auth.py 账号会话、test_audit.py 审计，均零网络；test_vector_store.py 读临时 Chroma）
```

两类**不进 pytest** 的测试工具（需要真实知识库/模型）：

```bash
python eval_verify.py eval_results_baseline_topk8.json   # 规则核对：日期/数字/链接能否在来源里找到
python stress_retrieval.py --workers 8 --rounds 3        # 检索层并发压测（不调 LLM）
```

`eval_verify.py` 与 LLM 打分互补：打分回答「好不好」，核对回答「有没有编」。它检查答案里的
日期、带单位数字、URL 是否全部落在检索来源中，退出码可直接用于 CI。

`stress_retrieval.py` 用线程池并发调用 `_retrieve_candidates`，报告 p50/p95/p99、
吞吐和失败用例；只读 Chroma，**不用停服务**。注意它是进程内线程压测，不覆盖
uvicorn/网络层。

## 许可证

MIT
