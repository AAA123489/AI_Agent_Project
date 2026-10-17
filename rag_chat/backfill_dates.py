"""backfill_dates.py — 补齐历史文档的发布日期 + date_int 元数据

## 为什么需要

2026-10 之前，部分子站（教务处、智能工程学院、车辆学院等）的详情页
不显示发布日期，或模板把日期放在解析器认不出的位置：

| 分类 | 空日期块占比 |
|---|---|
| 教务处通知 | 100% |
| 车辆与交通工程学院通知 | 100% |
| 智能工程学院通知 | 100% |
| 智能工程学院新闻 | 99% |

列表页其实有日期，只是解析器没取到。`publish_date` 为空 → 「昨天/最近」
这类问题无法按日期过滤，只能拿旧文档硬答。

本脚本只重新抓**列表页**（不抓正文），建立 URL → 日期映射，然后：
1. 给 Chroma 里同 URL 的块补 `publish_date` / `year`
2. 给所有块补 `date_int`（Chroma 数值区间过滤需要，旧数据没有）
3. 回头补 SQLite `scraped_articles.publish_date`

## 用法

    ../.venv/Scripts/python.exe backfill_dates.py             # 每类抓 3 页列表
    ../.venv/Scripts/python.exe backfill_dates.py --pages 10  # 补更老的文档
    ../.venv/Scripts/python.exe backfill_dates.py --dry-run   # 只看会改多少条

注意：必须在爬虫进程结束后跑（Chroma 单写者，避免锁冲突）。
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aiohttp

from campus_scraper.config import CATEGORIES, SCRAPER_DB_PATH
from campus_scraper.scraper import (
    filter_list_articles, resolve_next_url, scrape_list_page,
)
from src.time_range import date_to_int
from src.vector_store import VectorStore

COLLECTION_NAME = "my_rag_collection"
CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db")

# 列表页之间只做礼貌间隔，不需要爬虫那套 4~9 秒 + 3~8 分钟长休息：
# 这里只请求列表页，不抓正文，总量几十个请求。
_LIST_DELAY = 1.0


async def collect_list_dates(pages: int) -> dict[str, dict]:
    """抓各分类前 N 页列表，返回 {url: {date, title, category}}。"""
    mapping: dict[str, dict] = {}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CampusInfoBot/1.0)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        for cat in CATEGORIES:
            name = cat["name"]
            base_url = cat["base_url"]
            current = base_url + cat["list_path"]
            print(f"  📄 {name}: {current}", flush=True)

            for page in range(1, pages + 1):
                arts, pagination = await scrape_list_page(current, base_url, session)
                arts = filter_list_articles(arts, cat.get("url_contains"))
                new = 0
                for art in arts:
                    url = art.get("url")
                    if not url:
                        continue
                    old = mapping.get(url)
                    # 同一 URL 可能在不同分类重复出现；保留非空日期
                    if old is None or (not old.get("date") and art.get("date")):
                        mapping[url] = {
                            "date": art.get("date", "") or "",
                            "title": art.get("title", "") or "",
                            "category": name,
                        }
                    new += 1
                print(f"    第 {page} 页 {len(arts)} 条（累计 {len(mapping)} 个 URL）", flush=True)

                next_url = pagination.get("next_url")
                if not next_url or page >= pages:
                    break
                await asyncio.sleep(_LIST_DELAY)
                current = resolve_next_url(current, next_url)

            await asyncio.sleep(_LIST_DELAY)

    return mapping


def backfill_chroma(mapping: dict[str, dict], dry_run: bool = False) -> dict:
    """给 Chroma 块补 publish_date / year / date_int。"""
    vs = VectorStore(db_path=CHROMA_PATH, collection_name=COLLECTION_NAME)
    data = vs.collection.get(include=["metadatas"])
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []

    update_ids: list[str] = []
    update_metas: list[dict] = []
    filled = date_int_added = 0

    for doc_id, meta in zip(ids, metas):
        meta = meta or {}
        old_date = meta.get("publish_date", "") or ""
        new_date = old_date
        if not new_date:
            hit = mapping.get(meta.get("url", ""))
            if hit:
                new_date = hit.get("date", "") or ""

        di = date_to_int(new_date)
        old_di = meta.get("date_int")
        if new_date != old_date:
            filled += 1
        if di and old_di != di:
            date_int_added += 1

        if new_date != old_date or di != old_di:
            new_meta = dict(meta)
            new_meta["publish_date"] = new_date
            new_meta["date_int"] = di
            new_meta["year"] = new_date[:4]
            update_ids.append(doc_id)
            update_metas.append(new_meta)

    if not dry_run:
        for i in range(0, len(update_ids), 500):
            vs.collection.update(
                ids=update_ids[i:i + 500],
                metadatas=update_metas[i:i + 500],
            )

    return {
        "chunks": len(ids),
        "updated": len(update_ids),
        "filled_publish_date": filled,
        "date_int_added": date_int_added,
    }


def backfill_cache(mapping: dict[str, dict], dry_run: bool = False) -> int:
    """补 SQLite 缓存里空着的 publish_date。"""
    conn = sqlite3.connect(SCRAPER_DB_PATH)
    try:
        changed = 0
        for url, hit in mapping.items():
            date = hit.get("date", "")
            if not date:
                continue
            cur = conn.execute(
                "UPDATE scraped_articles SET publish_date = ? "
                "WHERE url = ? AND (publish_date IS NULL OR publish_date = '')",
                (date, url),
            )
            changed += cur.rowcount or 0
        if not dry_run:
            conn.commit()
        else:
            conn.rollback()
        return changed
    finally:
        conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="补齐历史文档发布日期")
    parser.add_argument("--pages", type=int, default=3, help="每类列表页数（默认 3）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库")
    args = parser.parse_args()

    print(f"抓取列表页（每类最多 {args.pages} 页）...")
    mapping = await collect_list_dates(args.pages)
    with_date = sum(1 for v in mapping.values() if v.get("date"))
    print(f"\n列表页共 {len(mapping)} 个 URL，其中 {with_date} 个带日期\n")

    print("补齐 Chroma 元数据...")
    result = backfill_chroma(mapping, dry_run=args.dry_run)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n补齐 SQLite 缓存...")
    cache_changed = backfill_cache(mapping, dry_run=args.dry_run)
    print(f"  更新 {cache_changed} 行")

    if args.dry_run:
        print("\n（dry-run：未写入任何数据）")
    else:
        print("\n完成。重启服务后 BM25 索引才会看到新元数据（索引是进程内单例）。")


if __name__ == "__main__":
    asyncio.run(main())
