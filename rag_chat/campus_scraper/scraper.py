"""
campus_scraper/scraper.py — 异步串行爬虫
===============================
特点：
- 串行！每次只发一个 HTTP 请求，绝不并发
- 请求间隔随机 4~9 秒（模拟真人浏览，避免防火墙告警）
- 失败自动重试（线性退避：5s → 10s → 15s…，第 n 次重试等待 5×n 秒）
- 分页自动遍历
"""

import asyncio
import hashlib
import logging
import os
import random
import re
import ssl
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

# ── TLS 上下文：必须显式放宽到 SECLEVEL=1 ──
# hait.edu.cn 全站（含 17 个分类入口 / 13 个子站）只提供 SHA-1 密码套件，实测协商到
# `AES128-SHA`；而 Python 自带的 OpenSSL 3.0 默认安全级别是 SECLEVEL=2，会**直接拒绝
# SHA-1**，于是握手阶段就失败，报 `SSLV3_ALERT_HANDSHAKE_FAILURE`——13 个分类全军覆没。
# 2026-08 那次全量爬取还能通，是对方服务器后来改了 TLS 配置（同一个 URL 用 curl 能通，
# 因为 curl 走的是另一套 OpenSSL 配置，所以「curl 能开 ≠ 爬虫能爬」）。
# 注意：只放宽密码套件等级，**证书校验保持开启**（证书本身有效，*.hait.edu.cn，2027-03 到期）；
# 且这是只读爬取公开信息，不涉及敏感数据传输。
_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.set_ciphers("DEFAULT@SECLEVEL=1")


def _random_delay() -> float:
    """生成随机延迟（模拟真人浏览节奏，避免固定间隔被防火墙识别）。"""
    return random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)


def resolve_next_url(current_url: str, next_path: str) -> str:
    """
    基于"当前列表页 URL"拼接下一页 URL。

    动易 CMS 的下一页链接常是裸路径（如 "310.htm"、"tzgg/81.htm"），
    必须相对当前列表页目录解析（保留 /tzgg、/index 等前缀），
    不能基于站点根目录 urljoin，否则会拼成 /310.htm → 404 提前停爬。
    """
    if next_path.startswith("http"):
        return next_path
    return urljoin(current_url, next_path.lstrip("/"))


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
                    ssl=_SSL_CONTEXT,   # 见文件顶部 _SSL_CONTEXT 注释：不放宽则全站握手失败
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
        max_pages: 最多爬几页列表。**这是上限而非固定成本**——一旦某页出现
            已爬过的 URL（说明时间线已接上缓存），立即停止翻页，见下方停止判据。
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
    seen_urls: set[str] = set()   # 本轮内去重；标记已挪到入库后，见下方注释
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
        new_count = 0      # 本页成功抓到并落库的篇数（日志用）
        seen_before = 0    # 本页命中「已爬过」缓存的 URL 数（停止翻页的判据）
        for art in list_articles:
            art_url = art["url"]
            art_title = art.get("title", "")
            art_date = art.get("date", "")

            # 日期过滤（列表页层面的快速跳过）
            if not _is_within_date_range(art_date):
                logger.debug("  ⏭ 跳过（超日期范围）: %s [%s]", art_title[:40], art_date)
                continue

            # 去重检查（DB 里的标记 = 上一轮已经入库）
            if is_scraped(art_url):
                logger.debug("  ⏭ 已爬过: %s", art_title[:40])
                seen_before += 1
                continue

            # 本轮内去重：标记既然要等入库成功才落，同一轮里重复出现的 URL
            # 就不能再指望 DB 标记挡住——由这张表兜着（不计入早停判据 seen_before）
            if art_url in seen_urls:
                continue
            seen_urls.add(art_url)

            if dry_run:
                # dry_run 只收 URL、从不入库，所以这里照旧直接标记：
                # 它本来就不承诺「已入库」，不存在标记与内容脱节的问题。
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
                new_count += 1
                # ⚠ 这里**不落**「已爬」标记。标记的含义是「这篇已经进知识库」，
                # 而入库（落盘 .txt → 分块 → 嵌入）要等**整个分类**爬完才做。
                # 标记写在这里 = 提前承诺一件还没发生的事：分类跑到一半进程被杀，
                # 标记留下了、正文却只在内存里，于是这篇正文永久丢失——而且之后
                # 每轮都被 is_scraped 静默跳过，看库根本发现不了。
                # 实测 2026-10-11 两次中途停机丢了 20 篇（14 学校新闻 + 6 马院新闻）。
                # 标记统一改由 pipeline._split_and_embed 在入库成功后落。

        logger.info("  ✅ [%s] 第 %d 页完成，本页新增 %d 篇，累计 %d 篇",
                    name, page_num, new_count, len(all_articles))

        # ── 停止判据：本页出现「已爬过」的 URL → 时间线已接上缓存，停止翻页 ──
        # 依据（2026-10-11 实测）：列表页是**跨页严格新→旧**的连续序列——
        # 通知公告 p1 10-10~09-23、p2 09-22~09-03、p3 09-02~08-05；
        # 学校新闻 p1 10-10~09-18、p2 09-18~09-05、p3 09-05~08-23。
        # 页与页无缝衔接，所以本页一旦有已爬内容，后面的页只会更旧、必然也已在缓存里。
        # 页首可能有 1 篇置顶（旧文章，如通知公告 p1 头部那条 03-20），但它同样是旧内容，
        # 命中缓存只会让停止来得更早，不会漏抓。
        # 效果：max_pages 从「每天固定要付出的成本」变成「真积压时的上限」——
        # 日常增量跑完第 1 页就停，省掉其后每页 3~8 分钟的页间休息。
        if seen_before > 0:
            logger.info("  ⏭ [%s] 第 %d 页含已爬内容，已追上进度，停止翻页",
                        name, page_num)
            break

        # ── 分页：没有下一页 / 到上限，都到此为止（先判再休息，不白等） ──
        if page_num >= max_pages:
            break
        if not pagination.get("next_url"):
            logger.info("  🏁 [%s] 无更多分页，完成", name)
            break

        # ── 批量间长休息（模拟真人"看完一页去干别的"，3~8 分钟） ──
        # 只在「本页全是新内容、确实还要往后翻」时才休息
        rest = random.uniform(BATCH_REST_MIN, BATCH_REST_MAX)
        logger.info("  ☕ [%s] 休息 %.0f 秒（%.1f 分钟）后翻下一页...",
                    name, rest, rest / 60)
        await asyncio.sleep(rest)
        current_url = resolve_next_url(current_url, pagination["next_url"])

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
