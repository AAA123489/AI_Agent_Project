"""
campus_scraper/storage.py — sqlite3 本地缓存
===============================
记录已爬 URL，实现增量爬取去重 + 统计。
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path

# 数据库文件放在 campus_scraper 包目录下
_DB_DIR = Path(__file__).resolve().parent
_DEFAULT_DB = str(_DB_DIR / "scraper_cache.db")


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or _DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> None:
    """初始化数据库表（首次调用时自动创建）。"""
    conn = _connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scraped_articles (
            url             TEXT PRIMARY KEY,
            title           TEXT,
            publish_date    TEXT,
            category        TEXT,
            source_site     TEXT,
            content_hash    TEXT,
            scraped_at      TEXT
        )
    """)
    conn.commit()
    conn.close()


def is_scraped(url: str, db_path: str | None = None) -> bool:
    """检查 URL 是否已经爬取过。"""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT 1 FROM scraped_articles WHERE url = ?", (url,)
    ).fetchone()
    conn.close()
    return row is not None


def mark_scraped(
    url: str,
    title: str = "",
    publish_date: str = "",
    category: str = "",
    source_site: str = "",
    content_hash: str = "",
    db_path: str | None = None,
) -> None:
    """记录一条已爬 URL。"""
    conn = _connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO scraped_articles
           (url, title, publish_date, category, source_site, content_hash, scraped_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (url, title, publish_date, category, source_site, content_hash,
         datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_scraped_urls(db_path: str | None = None) -> set[str]:
    """获取所有已爬 URL 集合。"""
    conn = _connect(db_path)
    rows = conn.execute("SELECT url FROM scraped_articles").fetchall()
    conn.close()
    return {r["url"] for r in rows}


def get_recent_scraped(
    category: str,
    days: int,
    limit: int = 5,
    db_path: str | None = None,
) -> list[dict]:
    """取最近 days 天内入库、且记录了正文哈希的文章，按入库时间从旧到新。

    用于「URL 不变、正文被改」的复检（scraper 默认按 URL 去重，改字感知不到）。
    升序 = 先检查快要滑出窗口的文章，保证每篇在窗口内至少轮到一次；
    刚在本轮入库的排最后，不会当天再抓一遍。
    """
    if days <= 0 or limit <= 0:
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT url, title, publish_date, category, content_hash, scraped_at
           FROM scraped_articles
           WHERE category = ? AND scraped_at >= ? AND content_hash <> ''
           ORDER BY scraped_at ASC
           LIMIT ?""",
        (category, cutoff, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(db_path: str | None = None) -> dict:
    """获取抓取统计：总数、按分类计数、上次爬取时间。"""
    conn = _connect(db_path)
    total = conn.execute("SELECT COUNT(*) as c FROM scraped_articles").fetchone()
    by_category = conn.execute(
        "SELECT category, COUNT(*) as c FROM scraped_articles GROUP BY category"
    ).fetchall()
    last = conn.execute(
        "SELECT MAX(scraped_at) as t FROM scraped_articles"
    ).fetchone()
    conn.close()

    return {
        "total_scraped": total["c"] if total else 0,
        "by_category": {r["category"]: r["c"] for r in by_category},
        "last_scraped_at": last["t"] or "",
    }
