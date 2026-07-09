# Day 2：带日志的配置读取工具

## 任务目标

1. 在项目根目录创建 `.env` 文件，包含假数据：`API_KEY=sk-123456789` 和 `DB_HOST=localhost`
2. 创建 `config.py`，使用 `python-dotenv` 读取环境变量，并封装为 `ConfigManager` 类
3. 创建 `main.py` 作为入口，导入 `ConfigManager`
4. 使用 `logging` 模块记录日志
5. 使用 `try...except` 捕获异常：尝试读取不存在的配置项时用 `logging.error()` 记录；成功读取 `API_KEY` 时用 `logging.info()` 记录

---

## 最终文件结构

```
AI_Agent_Project/
├── .env               # 环境变量（API_KEY, DB_HOST）
├── .gitignore
├── config.py          # ConfigManager 类
├── logger.py          # logging 日志配置
├── main.py            # 程序入口
├── file_handler.py    # Day 1 遗留：文件读取工具
├── requirements.txt   # 依赖：python-dotenv, loguru
├── app.log            # 日志输出文件
└── agent_env/         # 虚拟环境
```

---

## 各文件最终代码

### `.env`

```
API_KEY=sk-123456789
DB_HOST=localhost
```

### `logger.py`

```python
import logging
import sys
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

# 配置日志：同时输出控制台 + 写入文件（按大小轮转）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        RotatingFileHandler("app.log", maxBytes=500*1024, backupCount=3, encoding="utf-8")
    ]
)

if __name__ == "__main__":
    logger.info("日志模块测试")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.debug("调试信息")
```

### `config.py`

```python
from dotenv import load_dotenv
import os
from logger import logger

# 在模块加载时读取 .env 文件（只执行一次）
load_dotenv()

class ConfigManager:
    """
    配置管理器类，用于读取和管理配置文件中的参数。
    """
    def __init__(self):
        # 读取 API_KEY
        self.api_key = os.getenv("API_KEY")
        if self.api_key is None:
            logger.warning("API_KEY 未设置，请检查 .env 文件或环境变量。")
        else:
            logger.info("API_KEY 已成功加载。")
            logger.debug(f"API_KEY: {self.api_key}")

        # 读取 DB_HOST
        self.db_host = os.getenv("DB_HOST")
        if self.db_host is None:
            logger.warning("DB_HOST 未设置，请检查 .env 文件或环境变量。")
        else:
            logger.info("DB_HOST 已成功加载。")
            logger.debug(f"DB_HOST: {self.db_host}")

        # 读取 LOG_LEVEL（带默认值）
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # 故意读取一个不存在的配置项，验证 error 日志
        non_exist = os.getenv("NON_EXISTENT_CONFIG")
        if non_exist is None:
            logger.error("NON_EXISTENT_CONFIG 未设置，使用默认值。")

    def get_config(self):
        """返回当前的配置参数。"""
        return {
            "API_KEY": self.api_key,
            "DB_HOST": self.db_host,
            "LOG_LEVEL": self.log_level
        }
```

### `main.py`

```python
# main.py
# 导入日志模块
from logger import logger
from config import ConfigManager

if __name__ == "__main__":
    try:
        config = ConfigManager()
        logger.info("===== 程序开始运行 =====")
        config_get = config.get_config()
        logger.info(f"当前配置参数: {config_get}")
        logger.info("===== 程序运行结束 =====")
    except Exception as e:
        logger.error(f"程序运行时发生错误: {e}")
```

---

## 运行结果

```text
INFO     API_KEY 已成功加载。
INFO     DB_HOST 已成功加载。
ERROR    NON_EXISTENT_CONFIG 未设置，使用默认值。
INFO     ===== 程序开始运行 =====
INFO     当前配置参数: {'API_KEY': 'sk-123456789', 'DB_HOST': 'localhost', 'LOG_LEVEL': 'INFO'}
INFO     ===== 程序运行结束 =====
```

---

## 踩坑记录（知识点总结）

| # | 错误写法 | 报错/现象 | 正确写法 | 知识点 |
|---|----------|-----------|----------|--------|
| 1 | `from logging import logger` | `ImportError` | `import logging` + `logger = logging.getLogger(__name__)` | `logging` 是模块，没有内置 `logger` 对象；`loguru` 才有 `from loguru import logger` 的用法 |
| 2 | `logging.FileHandler("app.log", rotation="500 KB")` | `TypeError: unexpected keyword argument 'rotation'` | `RotatingFileHandler("app.log", maxBytes=500*1024, backupCount=3)` | `rotation` 是 `loguru` 的 API；标准库用 `RotatingFileHandler`，参数叫 `maxBytes` |
| 3 | `RotatingFileHandler(...)` 不导入 | `NameError` | `from logging.handlers import RotatingFileHandler` | `RotatingFileHandler` 在 `logging.handlers` 子模块里，不在 `logging` 顶层 |
| 4 | `try...except` 写在类体里而非方法里 | 异常永远捕获不到 | `try...except` 写在 `__init__` 方法内部 | 类体在定义时执行，挡不住方法调用时的异常 |
| 5 | 以为 `os.getenv()` 读不到 key 会抛异常 | `except` 块永远不触发 | 用 `if value is None` 判断返回值 | `os.getenv()` 对不存在的 key 返回 `None`，不抛异常 |
| 6 | `if not value` 判断配置是否存在 | 空字符串 `""` 被误判为未配置 | `if value is None` | `not ""` 也是 `True`，无法区分「值为空」和「key 不存在」 |
| 7 | `config = ConfigManager()` 写在模块级别 | 别人 `import main` 时也会触发初始化 | 放在 `if __name__ == "__main__":` 里面 | 模块级代码在导入时就会执行 |
| 8 | 在 Python REPL（`>>>`）里粘贴 PowerShell 命令 | `SyntaxError` | `exit()` 退出 REPL 回到终端再运行 | `>>>` 是 Python 交互模式，只能写 Python 语句；`PS` 是终端，运行 shell 命令 |

---

## 核心概念对比

### `loguru` vs `logging`

| | `loguru` | `logging` |
|------|----------|-----------|
| 导入 | `from loguru import logger` | `import logging` + `logger = logging.getLogger(__name__)` |
| 配置 | `logger.add("app.log", rotation="500 KB")` | `logging.basicConfig(handlers=[RotatingFileHandler(...)])` |
| 文件轮转 | `rotation="500 KB"` | `maxBytes=500*1024, backupCount=3` |
| 学习曲线 | 低，开箱即用 | 高，概念多（Logger / Handler / Formatter / Filter） |
| 适用场景 | 小型项目、快速原型 | 大型项目、需要精细控制 |

### `os.getenv()` 行为

```python
os.getenv("存在的KEY")       # → "对应的值"（字符串）
os.getenv("不存在的KEY")     # → None（不抛异常！）
os.getenv("KEY", "默认值")   # → KEY 不存在时返回 "默认值"
```

### `if not x` vs `if x is None`

```python
value = ""
if not value:     # True  — 空字符串是 falsy
if value is None: # False — 空字符串不是 None

value = None
if not value:     # True
if value is None: # True
```

---

## 延伸练习（选做）

1. 给 `ConfigManager` 添加 `get(key, default)` 方法，模仿 `dict.get()` 的语义
2. 学习 `logging.Formatter` / `logging.Filter`，实现更精细的日志控制
3. 对比 `loguru` 和 `logging` 的完整功能差异，理解各自的适用场景
4. 尝试用 `pytest` 为 `ConfigManager` 编写单元测试
