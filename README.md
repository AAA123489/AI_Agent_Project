# AI Agent 项目

一个基于 Python 异步编程的轻量级 AI Agent 脚手架，支持通过配置管理、结构化日志和并发 HTTP 请求与 LLM 服务交互。

## 项目简介

本项目是一个学习型 AI Agent 基础框架，核心能力包括：

- **配置管理** — 通过 `.env` 文件管理 API 密钥、接口地址等敏感配置
- **结构化日志** — 统一的日志模块，同时输出到控制台和滚动文件
- **异步并发** — 使用 `asyncio` + `aiohttp` 实现非阻塞的 LLM 并发调用
- **错误韧性** — 捕获特定异常并返回安全字典，避免单任务失败影响其他请求

当前适配的 API：**DeepSeek（Anthropic 兼容接口）**，模型为 `deepseek-v4-pro`。

## 目录结构

```
AI_Agent_Project/
├── main.py                 # 入口脚本：并发调用 LLM 的示例
├── requirements.txt        # 项目依赖清单
├── .env                    # 环境变量（API_KEY、API_URL 等，已加入 .gitignore）
├── .gitignore              # Git 忽略规则
├── app.log                 # 运行时日志文件（滚动输出）
├── Day2_日志与配置管理.md    # Day 2 学习笔记
├── Day3_Python异步核心.md   # Day 3 学习笔记
├── Day4_项目包结构与导入重构.md # Day 4 学习笔记
└── src/                    # 核心代码包
    ├── __init__.py          # 包标记文件
    ├── config.py            # ConfigManager：读取和管理配置参数
    ├── logger.py            # 统一日志实例（RotatingFileHandler + stderr）
    ├── llm_client.py        # 异步 LLM API 客户端
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
API_KEY=你的API密钥
API_URL=https://api.deepseek.com/anthropic/v1/messages
MODEL_NAME=deepseek-v4-pro
LOG_LEVEL=INFO
```

### 5. 运行

```bash
python main.py
```

程序将并发发送 3 条测试 prompt 到 LLM，日志会同时输出到控制台和 `app.log` 文件。

## 模块说明

| 模块 | 功能 |
|------|------|
| `src/config.py` | `ConfigManager` 类：从 `.env` 和环境变量读取 `API_KEY`、`API_URL`、`LOG_LEVEL` |
| `src/logger.py` | 项目级 `logger` 实例：INFO 级别，同时输出到 stderr 和 `app.log`（500KB 滚动，保留 3 份） |
| `src/llm_client.py` | `call_llm_client(prompt)`：异步向 LLM API 发 POST 请求，自动处理错误并返回安全字典 |
| `src/file_handler.py` | `read_text_file(path)`：以 UTF-8 安全读取文本文件，失败返回空字符串 |
| `main.py` | 演示入口：用 `asyncio.gather()` 并发执行 3 个 LLM 调用 |

## 核心依赖

| 包名 | 版本 | 作用 |
|------|------|------|
| `aiohttp` | 3.14.1 | 异步 HTTP 客户端 |
| `python-dotenv` | 1.2.2 | 加载 `.env` 文件到环境变量 |

## 许可证

MIT
