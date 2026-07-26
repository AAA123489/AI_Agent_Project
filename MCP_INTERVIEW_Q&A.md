# 项目介绍
我做的是一个 AI 智能知识库问答系统，核心功能是把技术文档导进去，用户用自然语言提问，系统自动检索相关内容，交给大模型生成回答。

整个系统分两段。离线入库这块，我自研了一个递归文本切分器——按段落、换行、句号、空格的优先级逐级降级切分，保证尽量不把一个完整观点切碎。切完以后用 sentence-transformers 转成向量，存到 Chroma 向量数据库。在线问答的时候，用户的问题同样转成向量，去 Chroma 做余弦相似度检索，经过 distance 阈值过滤和来源去重，拿到最相关的 5 个片段，拼接上 Redis 里的多轮对话历史，一起组成 Prompt 发给 DeepSeek，最后用 SSE 流式推给前端。

技术栈是 FastAPI + Chroma + Redis + SQLAlchemy。有一个我自己觉得比较有技术含量的点是流式传输的断连处理——用户中途关掉页面，aiohttp 跟大模型的长连接还在跑，如果不处理会资源泄漏。我用了一个"双层 finally + CancelledError 捕获"的机制，确保无论客户端什么时候断开，上游连接一定被关闭，对话记录也一定写入数据库。

这个项目最特别的一个模块是 MCP Server。我把知识库的检索和入库能力包装成了三个标准化的 MCP 工具——语义搜索、文档入库、列出数据源。Claude Code 通过 stdio 协议直接启动我的 MCP Server，像调函数一样调这三个工具来操作知识库。相当于给 AI Agent 装了一套操作知识库的"手脚"，不需要人手动去点界面。





# MCP 面试追问 —— 5 个问题 + 参考答案

> 基于「RAG 知识库 + MCP Server」项目的面试追问实录，逐题整理。

---

## 追问 1：MCP 和 REST API 有什么区别？

### 题目

你项目里已经有一套 REST API（`POST /chat`、`GET /history`），为什么还要做 MCP Server？直接用 REST 让 Agent 调不行吗？MCP 和 REST 在你这个项目里到底有什么本质区别？

### 面试官想听到的关键词

- **通信方式**：HTTP 请求-响应 vs stdio 子进程长连接
- **发现机制**：Swagger/文档 vs `tools/list` 自动发现
- **目标用户**：人 vs AI Agent
- **工具粒度设计**：端到端黑盒 vs 原子工具自由组合

### 你想听到的好答案

分三个维度讲：

**1. 通信方式不同。** REST 走 HTTP，需要起端口、配 CORS、管理连接——它是一个网络服务。MCP 走 stdio（标准输入输出），Agent 以子进程方式启动 MCP Server，通过 stdin/stdout 通信。没有端口、没有网络开销、进程退出即清理。

**2. 发现机制不同。** REST API 需要人先看 Swagger 文档才知道有哪些端点、参数叫什么、返回什么格式。MCP 内置了 `tools/list` 协议——Agent 连上之后自动拿到工具清单，包括每个工具的 `name`、`description`、`inputSchema`，不需要预先知道任何信息。

**3. 服务的目标用户不同。** REST 的 `/chat` 是一个完整的端到端管道（检索→拼 Prompt→调 LLM→流式返回），一步到底——人用起来方便。但对 Agent 来说这是个黑盒，它没法只检索不生成，也没法检索完自己再加工。MCP 把能力拆成原子工具（`search_knowledge_base` / `ingest_file` / `list_sources`），Agent 自己决定什么时候调哪个、怎么组合、结果怎么用。

**总结一句话：** REST 是人用的 Web API，需要看文档、手动调；MCP 是给 Agent 用的工具协议，自动发现、按需组合。

### 加分关键词

> stdio 子进程、`tools/list` 自动发现、原子工具粒度、"Agent 不是来适配我的接口，而是我的工具去适配 Agent 的决策流程"

---

## 追问 2：为什么选 MCP 而不是直接用 Function Calling？

### 题目

DeepSeek 本身就支持 tool calling，你直接定义几个 function 让模型调不行吗？MCP 帮你解决了 Function Calling 解决不了的什么问题？

### 面试官想听到的关键词

- **厂商格式 vs 行业协议**
- **工具定义的复用性**：一次定义，多个 Agent 可用
- **通信方式**：Function Calling 工具定义跟在请求里 vs MCP 启动时一次性发现
- **token 消耗差异**

### 你想听到的好答案

核心区别一句话：**Function Calling 是单个模型厂商的工具调用格式，MCP 是 AI Agent 行业的工具标准协议。**

展开讲：

**1. Function Calling 是厂商锁定，MCP 是跨平台复用。** 如果我用 DeepSeek 的 Function Calling 格式定义工具，换到 Claude 就得用 Anthropic 的 tool_use 格式重新写一遍。工具逻辑没变，适配代码写两遍。MCP 一次定义，Claude Code、Claude Desktop、以及任何支持 MCP 的 Agent 都能直接调用。

**2. 工具发现机制不同。** Function Calling 的工具定义是跟在每次请求的 `tools` 参数里发给模型的——每次对话都要带完整的工具 schema，消耗 token。MCP 是 Agent 启动时通过 `tools/list` 一次性发现，之后整个会话期间随时调用，不需要重复传输工具定义。

**3. 生命周期不同。** Function Calling 的工具活在单次 API 请求-响应里，无状态。MCP Server 是一个独立进程，可以维护连接状态、缓存资源（比如我的 VectorStore 单例，加载一次 Embedding 模型，所有工具调用共享）。

**类比：** Function Calling 相当于给特定手机写的遥控 App——小米能用，OPPO 装不了。MCP 是 USB 接口——键盘插什么都能用。

### 加分关键词

> "厂商格式 vs 行业协议"、跨 Agent 复用、"一次封装，到处能用"、"tools/list 一次性发现 vs 每次请求带 tools schema 消耗 token"、进程级资源复用

---

## 追问 3：MCP Server 的工具粒度怎么设计？

### 题目

你的 MCP 拆成了 `search_knowledge_base`、`ingest_file`、`list_sources` 三个工具。这个粒度是怎么考虑的？`search_knowledge_base` 的 `top_k` 默认值为什么是 5？如果 Agent 觉得不够要再调一次——这不也是"调用次数多"吗？

### 面试官想听到的关键词

- **参数灵活性 vs 工具粒度**：两个不同维度的问题
- **默认值的来源**：来自实际管道验证，不是拍脑袋
- **Agent 学习能力**：学一次就会，不是每次都要多调
- **拆太细 vs 拆太粗**的边界判断

### 你想听到的好答案

**粒度的核心原则：一个工具做一件事，但参数给它调整的自由度。**

我的三个工具：
- `search_knowledge_base`：只管检索，不负责生成
- `ingest_file`：只管入库，不负责检索
- `list_sources`：只管查数据源，不负责其他

为什么这样拆？

- **拆太细的坏处**：如果把 `search` 拆成 `search_by_keyword`、`search_by_semantic`、`search_by_date` 三个工具，Agent 光选工具就要花两轮，这才是无意义的调用开销。
- **拆太粗的坏处**：如果只有一个 `chat` 工具（端到端），Agent 没法只检索不生成，灵活性全丢。
- **当前粒度**：每个工具有明确的单一职责，Agent 自己决定组合方式。

**关于 `top_k` 默认值 5：**

这不是拍脑袋定的。我的 `/chat` RAG 管道的实际流程是：检索 15 条 → distance 阈值过滤 → 来源去重（每文档最多 3 条）→ 取前 5 条拼 Prompt。5 是这条管道跑下来验证过的数字——够覆盖 2-3 个不同来源，又不撑爆 LLM 上下文窗口。

**为什么设最大 20？** 就是给 "5 条不够" 留的后路。Agent 第一次拿 5 条发现信息不足，可以立刻调第二次 `search_knowledge_base("xxx", top_k=10)`。这个不是"工具粒度太细"的问题——Agent 学一次就会，下次直接用更高的 top_k。

**关键区分：** 参数层面的灵活性（top_k 可调）≠ 工具层面的粒度问题。前者是给 Agent 一个旋钮——默认拧到 5，不够自己拧。后者才是真正要避免的——拆成五个工具让 Agent 选半天。默认值降低决策成本，上限保留灵活性。

### 加分关键词

> "单一职责"、"参数灵活性 ≠ 工具粒度"、"默认值来自管道验证而非拍脑袋"、"默认值降低决策成本，上限保留灵活性"、"Agent 学一次就会"

---

## 追问 4：MCP Tool 的 description 是写给谁看的？

### 题目

你的 Tool description 字段是给人看的还是给 AI 看的？你是怎么写的？有没有验证过写得好不好？

### 面试官想听到的关键词

- **给 AI 的 description vs 给人的文档**：内容完全不同
- **三大要素**：触发场景（when） + 能力边界（what it does） + 返回值摘要（what you get）
- **usage hints / prompt engineering**：用自然语言做 in-context learning
- **验证方式**：工具触发准确率间接验证，而非一定要 A/B 测试

### 你想听到的好答案

**MCP Tool 的 description 是写给 AI 看的，不是写给人看的。**

对比一下就明白了。如果是给人的 API 文档，我会写：参数列表、返回值 JSON 结构、HTTP 状态码、错误码枚举。但 MCP Tool 的 description 不需要这些——**AI 不需要知道返回值的 JSON 字段名叫什么，AI 需要知道的是"什么时候该调这个工具"。**

一个好的 MCP Tool description 应该包含三个要素：

**① 触发场景（Usage Hints）——告诉 AI 什么时候用它**

例如：`"当用户问「FastAPI 的特点」「Redis 的用法」「什么是 RAG」等需要查阅技术文档的问题时使用此工具"`

这是在用自然语言做 in-context learning（usage hints）。AI 看到这几个具体例子，就知道"遇到这类问题 → 调这个工具"。

**② 能力边界——告诉 AI 这个工具能做什么、不能做什么**

例如：`"向量语义检索"`——不是关键词匹配。AI 看到就知道：查概念、查原理用它；查精确数字可能不合适。

**③ 返回值摘要——告诉 AI 调完能拿到什么**

例如：`"返回最相关的文档片段、来源文件路径和相似度分数"`

AI 不需要知道字段名叫 `metadata.source`，它知道"我能拿到来源和相似度"就够做下一步决策了。

**关于验证：**

实话实说，我没有做严格的 A/B 测试。但我有一个间接验证：`list_sources` 的最初版本 description 里没有写 "不需要任何参数"，结果有些 Agent client 会强行传一个空参数进去。加上了这句话，它就不再传了。这验证了一个经验：**给 AI 写 description，跟给新人同事写便利贴一样——告诉他"什么时候做、做了能得到什么"，比告诉他"底层怎么实现的"有用得多。**

### 加分关键词

> "给 AI 的 description 三大要素：触发场景 + 能力边界 + 返回值摘要"、"usage hints"、"in-context learning 用自然语言做"、"不必知道 JSON schema，知道能拿到什么就够决策"、"间接验证而非一定要 A/B"

---

## 追问 5：MCP Server 冷启动延迟？

### 题目

你的 MCP Server 是 stdio 子进程，Claude Code 退出时进程就没了。下次重新启动会 spawn 新进程，重新加载 420MB 的 Embedding 模型。这个冷启动延迟你怎么解决？还是你觉得不是问题？

### 面试官想听到的关键词

- **分层次思考**：先确认问题是否存在，再给方案
- **三种方案 + 各自的代价**：预加载 / 常驻服务 / 独立 embedding 微服务
- **工程务实**：当前规模下不修是正确的
- **懒加载是 trade-off，不是 bug**

### 你想听到的好答案

**先确认问题：** 当前用的是懒加载——第一次调工具时才初始化 VectorStore，加载 420MB 的 `paraphrase-multilingual-MiniLM-L12-v2` 模型。在机械硬盘上首次调用可能卡 5-10 秒。

**三层解决方案（按复杂度递增）：**

**方案一：启动时预加载。** 在 MCP Server 的 `main()` 里，进入 `stdio_server()` 之前就初始化好 VectorStore。优点：首次工具调用不卡。缺点：每次启动都吃 420MB 内存——哪怕这次对话根本不需要知识库。

**方案二：把 MCP Server 改成常驻服务。** 不用 stdio，改用 HTTP 或 Unix socket，变成一个长期存活的进程。模型只加载一次，Agent 连上来直接用。代价：运维负担增加——进程管理、端口管理、挂了要重启。且放弃了 stdio "启动即用、退出即清理"的简洁性。

**方案三：独立 Embedding 微服务。** 把 Chroma 的 embedding_function 指向一个独立的 embedding 服务（如 TEI、Infinity），MCP Server 本身变无状态——启动毫秒级。代价：架构多一层，部署复杂。

**但关键判断是：当前项目规模下，懒加载不做优化是正确的。**

理由：
- 我一天开关 Claude Code 不超过三次，等十秒不是致命问题。
- 懒加载避免了"启动就加载 420MB 但这次对话根本不用知识库"的浪费——因为 Agent 只在真正需要知识库时才会调用 `search_knowledge_base`，那时才触发加载，这是 deliberate trade-off：启动快 vs 首次调用慢。
- 如果这个 MCP Server 要给别人用、或者集成到 CI 流水线高频调用——那冷启动就是第一个要修的问题。但目前不需要。

### 加分关键词

> "分层方案"、"预加载 vs 常驻服务 vs 独立 embedding 微服务"、"每个方案都有代价"、"懒加载是 deliberate trade-off 不是 bug"、"当前规模不需要过度设计"、"工程务实——区分需要解决的问题和暂时不构成问题的问题"

---

## 总结：MCP 面试的核心评分维度

| 维度 | 实习生合格线 | 你的目标 |
|------|-------------|----------|
| **协议理解** | 知道 MCP 是什么、跟 REST 有什么不同 | 能从通信方式、发现机制、目标用户三个维度对比 |
| **架构决策** | 知道为什么选 MCP 不选 Function Calling | "厂商格式 vs 行业协议"、跨 Agent 复用、token 消耗差异 |
| **工具设计** | 能把能力拆成工具 | 能区分参数灵活性 vs 工具粒度、默认值来自数据验证 |
| **面向 AI 的设计思维** | 工具 description 写清楚 | 知道 description 三大要素、usage hints 本质是 prompt engineering |
| **工程判断** | 能发现问题 | 能分层给方案 + 说出每种方案的代价 + 知道当前规模该不该修 |

---

> 💡 **使用建议**：不要背答案。先理解每个问题在考察什么能力，然后用自己的话重新组织一遍。当你不需要看稿也能讲出"通信方式/发现机制/目标用户"三条时，再来找我重新面试。
