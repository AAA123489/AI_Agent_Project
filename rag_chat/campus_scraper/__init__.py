"""
campus_scraper — 校园官网爬虫 + 知识库管线
===============================
爬取河南工学院官网公开信息 → 清洗 → 分块 → 存入 ChromaDB。

公开 API:
- run_scrape_pipeline()      完整管线
- get_knowledge_base_stats() 查询知识库统计
- ingest_uploaded_files()    处理上传文件
"""

from .pipeline import run_scrape_pipeline, get_knowledge_base_stats, ingest_uploaded_files
from .models import ArticleMetadata, KnowledgeBaseStats

__all__ = [
    "run_scrape_pipeline",
    "get_knowledge_base_stats",
    "ingest_uploaded_files",
    "ArticleMetadata",
    "KnowledgeBaseStats",
]
