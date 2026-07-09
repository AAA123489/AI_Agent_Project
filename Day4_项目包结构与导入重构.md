# Day 4: 项目包结构与导入重构

## 目标

1. 将单文件代码重构为标准的 Python 包结构（`src/` + `__init__.py`）
2. 在 `main.py` 中正确使用绝对导入调用 `src` 下的模块
3. 为所有函数和类方法添加 Python 3.10+ 类型提示
4. 规范化管理虚拟环境和 `requirements.txt`

---

## 1. 包结构重构

### 迁移前
```
AI_Agent_Project/
├── main.py
├── logger.py          ← 散落在根目录
├── file_handler.py    ← 散落在根目录
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── llm_client.py
```

### 迁移后
```
AI_Agent_Project/
├── main.py            ← 唯一入口，留在包外
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   ├── logger.py      ← 迁入
│   └── file_handler.py ← 迁入
```

**原则：** 业务代码进 `src/`，入口脚本 `main.py` 留在外层。

---

## 2. 导入修复

### 致命错误 1：入口脚本使用相对导入

**错误写法（main.py）：**
```python
from .llm_client import call_llm_client  # ❌ 相对导入不能在入口脚本用
```

**报错：**
```
ImportError: attempted relative import with no known parent package
```

**原因：** 运行 `python main.py` 时，`__name__` = `"__main__"`（顶层脚本），相对导入要求 `__name__` 包含 `.`（即在包内部）。**相对导入只能在包内部模块之间使用。**

**正确写法（main.py）：**
```python
from src import llm_client              # ✅ 绝对导入（模块）
from src.logger import logger           # ✅ 绝对导入（变量）
```

---

### 致命错误 2：把「模块」当成「对象」导入

**错误写法（src/ 内部模块）：**
```python
from src import logger       # ❌ 导入的是 src/logger.py 这个"模块"
logger.info("...")           # 💥 模块没有 .info() 方法 → AttributeError
```

**正确写法：**
```python
from src.logger import logger  # ✅ 从模块中导入 logger 对象
```

**口诀：**
```
from src import logger        → 拿的是"装东西的盒子"（模块）
from src.logger import logger → 拿的是"盒子里的东西"（变量）
```

---

### 导入原则总结

| 场景 | 用什么 | 示例 |
|------|--------|------|
| 入口脚本 → src 内部 | 绝对导入 | `from src.logger import logger` |
| src 内部模块之间 | 相对导入（推荐）或绝对导入 | `from .logger import logger` |
| src 内部引用 Python 标准库/第三方库 | 绝对导入 | `import os` / `import aiohttp` |

---

## 3. 类型提示修复

### 错误 1：`self` 的类型标注

**错误：**
```python
def __init__(self: str) -> None:     # ❌ 意思是 self 是 str 类型
def get_config(self: str) -> dict:   # ❌ 同上
```

**正确：**
```python
def __init__(self) -> None:          # ✅ self 不需要标注
def get_config(self) -> dict[str, str | None]:  # ✅
```

Python 类型检查器会自动推断 `self` 的类型为当前类。

### 错误 2：返回值类型标注与实际不符

**错误：**
```python
async def main() -> dict[str, str | None]:  # ❌ 函数不返回任何东西
```

**正确：**
```python
async def main() -> None:                   # ✅ 没有 return 就是 None
```

### PEP 格式
```python
-> None      # ✅ -> 后面有空格
->dict       # ❌ -> 后面没有空格
```

---

## 4. 虚拟环境与依赖管理

### 清理过程

1. 旧环境 `agent_env/` 混入了 langchain、torch 等 150 个包，与项目无关
2. 删除旧环境，新建 `.venv/`
3. 只安装项目真正需要的：`aiohttp`、`python-dotenv`
4. 用 `pip freeze > requirements.txt` 生成精确版本

### 最终 requirements.txt（11 行，含传递依赖）

```
aiohappyeyeballs==2.7.1
aiohttp==3.14.1
aiosignal==1.4.0
attrs==26.1.0
frozenlist==1.8.0
idna==3.18
multidict==6.7.1
propcache==0.5.2
python-dotenv==1.2.2
typing_extensions==4.16.0
yarl==1.24.2
```

### .gitignore 更新

```diff
- agent_env/
+ .venv/
```

---

## 5. Git Bash × PowerShell 注意事项

| | Git Bash | PowerShell |
|---|:---|:---|
| 激活虚拟环境 | `source .venv/Scripts/activate` | `.\venv\Scripts\Activate.ps1` |
| pip 找不到 | `python -m pip` 或 `./.venv/Scripts/python.exe -m pip` | 通常直接用 `pip` |
| 路径分隔符 | `/` | `\` |
| 删除目录 | `rm -rf dir` | `Remove-Item -Recurse -Force dir` |

**建议：** 在 VS Code 中将默认终端设为 Git Bash，避免语法混乱。

---

## 关键收获

1. **相对导入只能在包内部使用**，入口脚本必须走绝对导入
2. `from package import module` vs `from package.module import name` 的区别
3. `self` 不需要类型标注，但返回值类型要如实反映函数行为
4. 一个项目一个虚拟环境，`pip freeze` 只导出当前项目的依赖
