"""
campus_scraper/models.py — 数据模型（dataclass）
"""

from dataclasses import dataclass, field


@dataclass
class ArticleMetadata:
    """文章元数据（解析后）"""
    title: str
    publish_date: str           # ISO 格式，如 "2026-08-05"
    category: str               # 分类名，如 "通知公告"
    source_site: str            # 来源站点，如 "河南工学院"
    url: str                    # 原文链接
    clean_text: str             # 清洗后的纯文本
    content_hash: str = ""      # 内容 MD5（用于去重）


@dataclass
class ScrapedArticle:
    """爬取到的原始文章"""
    url: str
    title: str
    publish_date: str
    category: str
    source_site: str
    html: str = ""


@dataclass
class KnowledgeBaseStats:
    """知识库统计信息"""
    total_articles: int = 0
    total_chunks: int = 0
    last_updated: str = ""              # ISO 时间
    category_counts: dict = field(default_factory=dict)
