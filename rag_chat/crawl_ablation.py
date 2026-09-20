"""
crawl_ablation.py — 消融实验专用爬取脚本
===============================
爬取指定分类的列表页 → 逐篇文章 → 隐私脱敏 → 落盘到 scraped_docs/。

用法:
    python crawl_ablation.py                    # 默认: 通知公告 1 页，前 10 篇
    python crawl_ablation.py 通知公告 1 10      # 指定分类 + 页数 + 最多几篇
    python crawl_ablation.py 学校新闻 2 20

说明:
- 落盘的 .txt 已执行 sanitize_privacy（删个人手机号/邮箱，保留办公室座机）
- 只爬取落盘，不加载 Embedding 模型（快），重建知识库走 rebuild_kb.py
"""

import asyncio
import io
import logging
import sys

if __name__ == "__main__":
    # Windows 控制台 GBK 兼容。放模块级会劫持 import 本模块者的 stdout，
    # pytest 拆捕获流时直接 I/O 崩 —— 只在作为脚本运行时改。
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

import aiohttp

from campus_scraper.config import CATEGORIES, USER_AGENT
from campus_scraper.pipeline import _save_clean_text
from campus_scraper.scraper import scrape_category


async def main() -> None:
    cat_name = sys.argv[1] if len(sys.argv) > 1 else "通知公告"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    cfg = next((c for c in CATEGORIES if c["name"] == cat_name), None)
    if cfg is None:
        print(f"❌ 未找到分类: {cat_name}")
        print("可选分类:", "、".join(c["name"] for c in CATEGORIES))
        return

    print(f"▶️  爬取分类: {cat_name}（最多 {max_pages} 页）")
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        articles = await scrape_category(cfg, session, max_pages=max_pages)

    if not articles:
        print("⚠️  未爬到文章（可能都在缓存/日期过滤外）")
        return

    total_found = len(articles)
    if total_found > limit:
        articles = articles[:limit]
        print(f"ℹ️  列表共 {total_found} 篇，本次只落盘前 {limit} 篇")

    saved = 0
    for art in articles:
        path = _save_clean_text(art)   # 内部已脱敏
        saved += 1
        print(f"  📄 {art.title[:40]} → {path.name}")

    print(f"\n🎉 完成: {saved} 篇已脱敏落盘到 scraped_docs/{cat_name}/")


if __name__ == "__main__":
    asyncio.run(main())
