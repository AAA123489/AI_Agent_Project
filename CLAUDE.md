# RAG 知识库问答系统

基于 FastAPI + ChromaDB + DeepSeek 的检索增强生成问答后端，面向高校官网公开信息。

## 项目结构

```
RAG/
├── rag_chat/          # 后端代码
└── README.md
```

## 功能一览

`rag_chat/` — 基于 FastAPI + ChromaDB + DeepSeek 的检索增强生成问答后端。

| 能力 | 技术 |
|------|------|
| 网页解析 | HTML → 纯文本（动易/JSP CMS 适配），爬虫为唯一入库通路 |
| 智能分块 | 自研递归切分器（中文标点分隔 + 重叠窗口） |
| 混合检索 | ChromaDB 向量 + BM25 关键词，RRF 融合，可选 CrossEncoder 重排 |
| 召回自检 | 手写判定循环：判定召回能否支撑回答，不能则**拒答**而非编造，不充分则改写 query 重检 |
| 库外拦截 | 主体校校验（两层：工具参数 + 用户原话），外校问题直接拒答不检索 |
| 流式对话 | SSE（Server-Sent Events）流式推送 |
| 多轮记忆 | Redis LPUSH + EXPIRE 30 分钟过期 |

详见 [rag_chat/README.md](rag_chat/README.md)

### 当前状态

**知识库**：全量爬虫已完成，1195 篇文档 / 13 分类 → **5919 块 / 1194 source**。分块 `CHUNK_SIZE=300 / OVERLAP=50`（消融最优）。

> **监控口径**：`GET /kb-stats` 的 `unaccounted_chunks` 必须恒为 **0**
> （= 总数 − 13 个爬虫分类之和）。非 0 就是库里有来源不明的内容。
> 2026-09-20 曾靠这个差额才发现 5 篇测试上传的文档（378 块，含外校名单）在库里躺了半个月——
> 当时线上是 6297 块，而所有评测成绩都是 5919 块那套量的。
> **上传通路已于 2026-10 整条移除**（当时用 `DELETE /upload/{文件名}` 撤销）；详见 [docs/改进记录.md](rag_chat/docs/改进记录.md) 第 18、21 条。

**运行参数**（在 `.env`，改配置不用改代码）：
- `TEMPERATURE=0.3`、`MAX_TOKENS=1000`、`TOP_K=8`、`KB_SUBJECT_SCHOOL=河南工学院`
- `RETRIEVAL_MODE=hybrid`（向量 + BM25 RRF 融合）、`RETRIEVAL_RERANK=off`（重排默认关，遇表格行值题可临时开，代价每问 +1.9s）
- `RECALL_GUARD=off`（召回自检默认关，关掉即改造前行为）、`RECALL_GUARD_MAX_ATTEMPTS=1`（检索轮数，2 = 开改写重检环）

**评测成绩单**：

| 评测 | 结果 |
|------|------|
| 6 题基准（60 分制，2026-09-16 复跑） | **60/60** |
| 召回自检探测（硬负例 6 / 无关题 4 / 正例 5） | 自检 **off**：负例 0/6 拒答；**on**：负例 **6/6**、无关 **4/4**、正例 **5/5** 零误拒 |
| 25 题随机 | **96%** |
| 50 题随机 | **100%** |
| 100 题随机（回归后） | **95%** |

> ⚠️ **60/60 有采样噪声**：2026-09-17 用同一份代码跑两遍得到 60/60 与 50/60。当时那次的根因是 Q6 要过两级 LLM，LLM 某次把工具参数里的校名丢了 → 闸门没拦 → 回答如实列了真数字 → 评分判幻觉；**该路径已修**（`_guard_tool_call` 用用户原话二次校验，见改进记录第 17 条），但「回答 + 打分」两级 LLM 本身仍有抖动。**衡量检索层改动请用 `eval_guard_probe.py`（检索层、确定性），不要用 60/60。**
> 复跑：`cd rag_chat && python eval_baseline.py --tag topk8 && python eval_score.py eval_results_baseline_topk8.json`
> 25/50/100 题那三套的题库与脚本从未入库、已彻底丢失，无法复跑（成绩仅存于表内）。
> 详细评测过程与改造历史见 [rag_chat/docs/改进记录.md](rag_chat/docs/改进记录.md)。

## License

MIT
