"""
campus_scraper/scraper.py — 异步串行爬虫
===============================
特点：
- 串行！每次只发一个 HTTP 请求，绝不并发
- 请求间隔随机 4~9 秒（模拟真人浏览，避免防火墙告警）
- 失败自动重试（指数退避：5s → 10s）
- 分页自动遍历
"""

import asyncio
import hashlib
import logging
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import aiohttp

from .config import (
    CATEGORIES, REQUEST_DELAY_MIN, REQUEST_DELAY_MAX, REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES, RETRY_BACKOFF_BASE, DATE_FILTER_DAYS,
    MAX_LIST_PAGES, USER_AGENT,
    BATCH_REST_MIN, BATCH_REST_MAX,
)
from .models import ScrapedArticle, ArticleMetadata
from .storage import init_db, is_scraped, mark_scraped
from .html_parser import parse_html, extract_article_urls, extract_pagination

logger = logging.getLogger("campus_scraper.scraper")

# ── 输出目录（清洗后的 .txt） ──
_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "scraped_docs"

# ── 日期过滤阈值 ──
_DATE_CUTOFF = datetime.now() - timedelta(days=DATE_FILTER_DAYS)


def _random_delay() -> float:
    """生成随机延迟（模拟真人浏览节奏，避免固定间隔被防火墙识别）。"""
    return random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)


def _is_within_date_range(date_str: str) -> bool:
    """检查日期是否在 DATE_FILTER_DAYS 天内。"""
    if not date_str:
        return True  # 无法解析日期，默认保留
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d >= _DATE_CUTOFF
    except ValueError:
        return True


# ═══════════════════════════════════════════════════════
# HTTP 请求
# ═══════════════════════════════════════════════════════

async def fetch_page(
    url: str,
    session: aiohttp.ClientSession | None = None,
) -> str | None:
    """获取单页 HTML，带重试。"""
    headers = {"User-Agent": USER_AGENT}

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(headers=headers)

    try:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
                ) as resp:
                    if resp.status == 200:
                        # 尝试检测编码
                        raw = await resp.read()
                        # 动易 CMS 用 UTF-8
                        for enc in ["utf-8", "gb2312", "gbk"]:
                            try:
                                return raw.decode(enc)
                            except (UnicodeDecodeError, LookupError):
                                continue
                        return raw.decode("utf-8", errors="replace")
                    elif resp.status == 404:
                        logger.warning("404 Not Found: %s", url)
                        return None
                    else:
                        logger.warning("HTTP %d: %s (attempt %d/%d)",
                                       resp.status, url, attempt + 1, MAX_RETRIES + 1)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("请求失败: %s → %s (attempt %d/%d)",
                               url, e, attempt + 1, MAX_RETRIES + 1)

            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (attempt + 1)
                logger.info("等待 %ds 后重试...", wait)
                await asyncio.sleep(wait)

        logger.error("最终失败: %s", url)
        return None
    finally:
        if own_session and session:
            await session.close()


# ═══════════════════════════════════════════════════════
# 文章爬取
# ═══════════════════════════════════════════════════════

async def scrape_article(
    url: str,
    category: str,
    source_site: str,
    session: aiohttp.ClientSession,
) -> ArticleMetadata | None:
    """爬取单篇文章：fetch → parse → 日期过滤。"""
    html = await fetch_page(url, session)
    if not html:
        return None

    clean_text, meta = parse_html(html, url)
    meta["category"] = category
    meta["source_site"] = source_site

    # 日期过滤
    pub_date = meta.get("publish_date", "")
    if not _is_within_date_range(pub_date):
        logger.info("⏭ 跳过（超出日期范围）: %s [%s]", meta.get("title", url)[:50], pub_date)
        return None

    if not clean_text.strip():
        logger.info("⏭ 跳过（无正文内容）: %s", url)
        return None

    return ArticleMetadata(
        title=meta.get("title", ""),
        publish_date=pub_date,
        category=category,
        source_site=source_site,
        url=url,
        clean_text=clean_text,
        content_hash=meta.get("content_hash", ""),
    )


# ═══════════════════════════════════════════════════════
# 列表页 + 分类爬取
# ═══════════════════════════════════════════════════════

async def scrape_list_page(
    url: str,
    base_url: str,
    session: aiohttp.ClientSession,
) -> tuple[list[dict], dict]:
    """
    爬取一个列表页。
    返回: (articles_list, pagination_info)
    """
    html = await fetch_page(url, session)
    if not html:
        return [], {"total": 0, "current": 1, "total_pages": 1, "next_url": None}

    articles = extract_article_urls(html, base_url)
    pagination = extract_pagination(html)
    return articles, pagination


async def scrape_category(
    category_cfg: dict,
    session: aiohttp.ClientSession,
    max_pages: int = MAX_LIST_PAGES,
    dry_run: bool = False,
) -> list[ArticleMetadata]:
    """
    爬取一个分类下的所有列表页 → 逐篇文章。

    参数:
        category_cfg: CATEGORIES 中的一项
        max_pages: 最多爬几页列表
        dry_run: 如果为 True，只收集 URL 不实际抓取文章内容
    """
    name = category_cfg["name"]
    icon = category_cfg["icon"]
    base_url = category_cfg["base_url"]
    list_path = category_cfg["list_path"]
    source_site = category_cfg["site_label"]

    logger.info("%s 开始爬取: %s (%s)", icon, name, base_url + list_path)
    init_db()  # 确保数据库就绪

    all_articles: list[ArticleMetadata] = []
    current_url = base_url + list_path

    for page_num in range(1, max_pages + 1):
        logger.info("  📄 [%s] 列表页 %d: %s", name, page_num, current_url)

        # 间隔等待（对服务器友好，随机抖动模拟真人）
        if page_num > 1 or all_articles:
            await asyncio.sleep(_random_delay())

        list_articles, pagination = await scrape_list_page(
            current_url, base_url, session
        )

        if not list_articles:
            logger.info("  ⚠ [%s] 第 %d 页无文章，停止", name, page_num)
            break

        # ── 逐篇爬取 ──
        new_count = 0
        for art in list_articles:
            art_url = art["url"]
            art_title = art.get("title", "")
            art_date = art.get("date", "")

            # 日期过滤（列表页层面的快速跳过）
            if not _is_within_date_range(art_date):
                logger.debug("  ⏭ 跳过（超日期范围）: %s [%s]", art_title[:40], art_date)
                continue

            # 去重检查
            if is_scraped(art_url):
                logger.debug("  ⏭ 已爬过: %s", art_title[:40])
                continue

            if dry_run:
                logger.info("  🔍 [DRY RUN] 将爬取: %s", art_title[:50])
                mark_scraped(
                    url=art_url, title=art_title, publish_date=art_date,
                    category=name, source_site=source_site,
                )
                continue

            # 间隔等待（随机抖动）
            await asyncio.sleep(_random_delay())

            logger.info("  📝 [%s] %s", name, art_title[:60])

            article = await scrape_article(
                url=art_url, category=name, source_site=source_site,
                session=session,
            )

            if article:
                all_articles.append(article)
                mark_scraped(
                    url=art_url, title=article.title,
                    publish_date=article.publish_date,
                    category=name, source_site=source_site,
                    content_hash=article.content_hash,
                )
                new_count += 1

        logger.info("  ✅ [%s] 第 %d 页完成，本页新增 %d 篇，累计 %d 篇",
                    name, page_num, new_count, len(all_articles))

        # ── 批量间长休息（模拟真人"看完一页去干别的"，3~8 分钟） ──
        if page_num < max_pages:
            rest = random.uniform(BATCH_REST_MIN, BATCH_REST_MAX)
            logger.info("  ☕ [%s] 休息 %.0f 秒（%.1f 分钟）后翻下一页...",
                        name, rest, rest / 60)
            await asyncio.sleep(rest)

        # ── 分页：有没有下一页？ ──
        if pagination.get("next_url"):
            next_path = pagination["next_url"]
            if next_path.startswith("http"):
                current_url = next_path
            else:
                current_url = urljoin(base_url, next_path)
        else:
            logger.info("  🏁 [%s] 无更多分页，完成", name)
            break

    logger.info("🏁 [%s] 全部完成，共爬取 %d 篇", name, len(all_articles))
    return all_articles


async def scrape_all_categories(
    max_pages: int = MAX_LIST_PAGES,
    dry_run: bool = False,
) -> dict[str, list[ArticleMetadata]]:
    """
    串行爬取所有分类（一个接一个，绝不并发）。

    返回: {category_name: [ArticleMetadata, ...]}
    """
    results: dict[str, list[ArticleMetadata]] = {}

    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        for cat_cfg in CATEGORIES:
            name = cat_cfg["name"]
            try:
                articles = await scrape_category(
                    cat_cfg, session, max_pages=max_pages, dry_run=dry_run,
                )
                results[name] = articles
            except Exception as e:
                logger.exception("❌ [%s] 爬取出错: %s", name, e)
                results[name] = []

    total = sum(len(v) for v in results.values())
    logger.info("🎉 全部完成！共爬取 %d 篇文章", total)
    return results
