"""pytest 自动加载：把 langgraph_agent 目录加进 sys.path，让测试能 import agent / tools。

运行方式（在 langgraph_agent 目录下）：
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
