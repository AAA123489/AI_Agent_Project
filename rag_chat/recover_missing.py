"""recover_missing.py — 补回「已爬但未入库」的文章

## 为什么需要

2026-08-07/08 首次全量爬取时，「已爬标记」写在爬取阶段、入库却是整个分类
跑完才做（见 docs/改进记录.md 第 28 条）。中途停机留下 **207 条标记**，
正文从未进入 ChromaDB。bug 修复后标记不再提前落，但这些历史孤儿标记
会让 `is_scraped()` 继续静默跳过它们；而日常增量又「遇到已爬即停」，
翻页深度根本够不到这些老 URL——不手工补就永远缺失。

## 判据

同时满足两条才算真缺失：

1. URL 不在 Chroma 任何块的 `url` 元数据里；
2. `content_hash` 前 8 位不在任何块 id 里（id 形如
   `scrape_{日期}_{hash8}_{序号}`）。

只满足第 1 条的，多半是同一篇内容换了 URL/日期入库，**不能重复补**。

## 用法

    ../.venv/Scripts/python.exe recover_missing.py --dry-run   # 只看清单
    ../.venv/Scripts/python.exe recover_missing.py             # 全量补
    ../.venv/Scripts/python.exe recover_missing.py --limit 10  # 先试 10 篇
    ../.venv/Scripts/python.exe recover_missing.py --fix-titles  # 修复被截断的标题

`--fix-titles` 修的是另一个旧 bug：`parse_html` 曾按**第一个**连字符截站点
后缀，把标题里的学年区间「2025-2026」拦腰砍断（「关于公布2025」）。重抓
这些页面、按正确标题重切块重入库（先按 URL 删旧块）。

**必须在服务/爬虫停止时跑**（Chroma 单写者，避免锁冲突）。补完重启服务，
BM25 才会重新建索引。

按爬虫既有策略，发布时间超出 `DATE_FILTER_DAYS`（2 年）的文章不补——
它们本来也不在知识库的收录范围内。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aiohttp

from campus_scraper.config import (
    DATE_FILTER_DAYS, SCRAPER_DB_PATH, USER_AGENT,
)
from campus_scraper.pipeline import (
    _SCRAPED_DOCS_DIR, _safe_filename, _save_clean_text, _split_and_embed,
)
from campus_scraper.scraper import _random_delay, scrape_article
from src.vector_store import VectorStore

COLLECTION_NAME = "my_rag_collection"
CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db")

_DATE_CUTOFF = (datetime.now() - timedelta(days=DATE_FILTER_DAYS)).date().isoformat()


def find_missing() -> list[dict]:
    """缓存库里「有标记、库内无内容」的文章。"""
    vs = VectorStore(db_path=CHROMA_PATH, collection_name=COLLECTION_NAME)
    data = vs.collection.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    ids = data.get("ids") or []

    urls = {(m or {}).get("url", "") for m in metas}
    hashes = set()
    for doc_id in ids:
        parts = str(doc_id).split("_")
        if len(parts) >= 3 and parts[0] == "scrape":
            hashes.add(parts[2])

    conn = sqlite3.connect(SCRAPER_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT url, title, publish_date, category, source_site, content_hash "
            "FROM scraped_articles ORDER BY scraped_at, url"
        ).fetchall()
    finally:
        conn.close()

    missing = []
    for url, title, pub_date, category, site, content_hash in rows:
        if url in urls:
            continue
        if (title or "").startswith("（死链）"):
            continue  # 已确认的死链：保留标记防止反复重试，但没有内容可补
        if (content_hash or "")[:8] in hashes:
            continue  # 内容已在库里，只是换了 URL/日期
        missing.append({
            "url": url,
            "title": title or "",
            "publish_date": pub_date or "",
            "category": category or "",
            "source_site": site or "",
        })
    return missing


async def recover(items: list[dict]) -> dict:
    vs = VectorStore(db_path=CHROMA_PATH, collection_name=COLLECTION_NAME)
    present_hashes = set()
    for doc_id in vs.collection.get(include=[])["ids"]:
        parts = str(doc_id).split("_")
        if len(parts) >= 3 and parts[0] == "scrape":
            present_hashes.add(parts[2])

    stats = {"ok": 0, "chunks": 0, "dup": 0, "empty": 0, "old": 0, "error": 0}
    headers = {"User-Agent": USER_AGENT}

    async with aiohttp.ClientSession(headers=headers) as session:
        for n, item in enumerate(items, 1):
            url = item["url"]
            label = (item["title"] or url)[:45]

            if item["publish_date"] and item["publish_date"] < _DATE_CUTOFF:
                stats["old"] += 1
                print(f"[{n}/{len(items)}] ⏭ 超出保留窗口 {item['publish_date']}: {label}", flush=True)
                continue

            try:
                article = await scrape_article(
                    url=url,
                    category=item["category"],
                    source_site=item["source_site"],
                    session=session,
                    fallback_date=item["publish_date"],
                )
            except Exception as e:
                stats["error"] += 1
                print(f"[{n}/{len(items)}] ❌ 抓取失败: {label} → {e}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            if article is None:
                stats["empty"] += 1
                print(f"[{n}/{len(items)}] ⚠ 无正文/超范围，跳过: {label}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            h8 = article.content_hash[:8]
            if h8 in present_hashes:
                stats["dup"] += 1
                print(f"[{n}/{len(items)}] ↺ 内容已在库（{h8}），跳过: {label}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            try:
                _save_clean_text(article)
                chunks = _split_and_embed([article], vs)
            except Exception as e:
                stats["error"] += 1
                print(f"[{n}/{len(items)}] ❌ 入库失败: {label} → {e}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            present_hashes.add(h8)  # 同内容若在本批出现两次，第二篇直接算重复
            stats["ok"] += 1
            stats["chunks"] += chunks
            print(
                f"[{n}/{len(items)}] ✅ {chunks} 块: {article.title[:45]} "
                f"[{item['publish_date']}]",
                flush=True,
            )
            await asyncio.sleep(_random_delay())

    return stats


_TRUNCATED_TITLE_RE = re.compile(r"\d{4}$")


def _looks_truncated(source: str) -> bool:
    """「关于公布2025」这类被连字符砍断的标题：短、以完整年份结尾。"""
    return bool(source) and len(source) < 20 and bool(_TRUNCATED_TITLE_RE.search(source))


def _doc_path(title: str, category: str) -> Path:
    """与 _save_clean_text 相同的落盘路径。"""
    return _SCRAPED_DOCS_DIR / _safe_filename(category) / f"{_safe_filename(title)}.txt"


async def fix_titles(dry_run: bool = False) -> dict:
    """重抓标题被截断的文章，用正确标题重新切块入库。"""
    vs = VectorStore(db_path=CHROMA_PATH, collection_name=COLLECTION_NAME)
    metas = vs.collection.get(include=["metadatas"])["metadatas"]

    bad: dict[str, dict] = {}
    for m in metas:
        m = m or {}
        url, source = m.get("url", ""), m.get("source", "")
        if url and _looks_truncated(source):
            bad.setdefault(url, {
                "old": source,
                "category": m.get("category", ""),
                "source_site": m.get("source_site", ""),
                "publish_date": m.get("publish_date", ""),
            })

    print(f"疑似标题截断 {len(bad)} 个 URL\n")
    stats = {"fixed": 0, "chunks": 0, "skipped": 0, "error": 0}
    if dry_run:
        for url, info in bad.items():
            print(f"  {info['publish_date']}  {info['old']!r}  {url}")
        return stats

    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        for n, (url, info) in enumerate(bad.items(), 1):
            try:
                article = await scrape_article(
                    url=url,
                    category=info["category"],
                    source_site=info["source_site"],
                    session=session,
                    fallback_date=info["publish_date"],
                )
            except Exception as e:
                stats["error"] += 1
                print(f"[{n}/{len(bad)}] ❌ 抓取失败 {url}: {e}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            # 严格守卫：只有新标题确实以旧标题开头且更长，才认定是被截断
            if article is None or not (
                article.title.startswith(info["old"])
                and len(article.title) > len(info["old"])
            ):
                stats["skipped"] += 1
                print(f"[{n}/{len(bad)}] ↺ 标题未变或无法确认，跳过: {info['old']!r}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            try:
                # 先删该 URL 的旧块：标题变了，块文本前缀和块数都可能变
                vs.collection.delete(where={"url": url})
                new_path = _save_clean_text(article)
                chunks = _split_and_embed([article], vs)
            except Exception as e:
                stats["error"] += 1
                print(f"[{n}/{len(bad)}] ❌ 重建失败 {info['old']!r}: {e}", flush=True)
                await asyncio.sleep(_random_delay())
                continue

            old_path = _doc_path(info["old"], info["category"])
            if old_path.exists() and old_path != new_path:
                old_path.unlink()

            stats["fixed"] += 1
            stats["chunks"] += chunks
            print(f"[{n}/{len(bad)}] ✅ {chunks} 块: {info['old']!r} → {article.title!r}", flush=True)
            await asyncio.sleep(_random_delay())

    return stats


async def main() -> None:
    parser = argparse.ArgumentParser(description="补回「已爬但未入库」的文章")
    parser.add_argument("--limit", type=int, default=0, help="只补前 N 篇（0 = 全部）")
    parser.add_argument("--dry-run", action="store_true", help="只列清单，不抓取")
    parser.add_argument("--fix-titles", action="store_true",
                        help="修复被连字符截断的标题（重抓 + 重切块）")
    args = parser.parse_args()

    if args.fix_titles:
        stats = await fix_titles(dry_run=args.dry_run)
        if not args.dry_run:
            print("\n结果：", stats)
            if stats["fixed"]:
                print("\n完成。重启服务后 BM25 索引才会看到新标题。")
        return

    items = find_missing()
    print(f"待补 {len(items)} 篇（缓存库有标记、Chroma 无 URL 且无内容哈希）\n")

    def in_window(it: dict) -> bool:
        return not it["publish_date"] or it["publish_date"] >= _DATE_CUTOFF

    todo_all = [it for it in items if in_window(it)]
    print(f"保留窗口内 {len(todo_all)} 篇；超出 {DATE_FILTER_DAYS} 天 {len(items) - len(todo_all)} 篇按策略跳过\n")

    if args.dry_run:
        for it in items:
            mark = "" if in_window(it) else "  [超窗口]"
            print(f"  {it['publish_date'] or '无日期'}  {it['category']}  {it['title'][:50]}{mark}")
        return

    todo = todo_all[: args.limit] if args.limit else todo_all
    if not todo:
        print("没有需要补的文章。")
        return
    print(f"开始补 {len(todo)} 篇（每篇间隔 4~9 秒，预计 {len(todo) * 6.5 / 60:.0f} 分钟）...\n")

    stats = await recover(todo)
    print("\n结果：", stats)
    if stats["ok"]:
        print("\n完成。重启服务后 BM25 索引才会看到新文章（索引是进程内单例）。")


if __name__ == "__main__":
    asyncio.run(main())
