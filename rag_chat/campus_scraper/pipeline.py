"""
campus_scraper/pipeline.py — 核心桥接模块
===============================
职责：
- run_scrape_pipeline()    爬取 → 清洗 → 保存 .txt → 分块 → 嵌入 → ChromaDB
- get_knowledge_base_stats()  从 ChromaDB 查询动态统计（含「来源不明」差额）
"""

import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

# ── 路径：确保能导入同目录模块 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vector_store import VectorStore
from src.time_range import date_to_int
from text_splitter import RecursiveTextSplitter, sanitize_privacy

from .config import (
    CATEGORIES, MAX_LIST_PAGES, USER_AGENT,
    SCRAPE_RECHECK_DAYS_DEFAULT, SCRAPE_RECHECK_MAX_DEFAULT,
)
from .scraper import scrape_category, scrape_article, _random_delay
from .models import ArticleMetadata, KnowledgeBaseStats
from .storage import mark_scraped, get_recent_scraped

logger = logging.getLogger("campus_scraper.pipeline")

# ── 输出目录 ──
_SCRAPED_DOCS_DIR = _PROJECT_ROOT / "scraped_docs"

# ── 分块参数 ──
# 消融实验最优（2026-08-07，6题60分制）：chunk=300 / overlap=50 / top_k=8 → 46分
# 详见 README「消融实验结果」。
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ── 知识库集合名（修复 bug：统一用 my_rag_collection） ──
COLLECTION_NAME = "my_rag_collection"

# Windows 文件名非法字符（<>:"/\|?* 与控制字符）。标题里出现 " 或 | 时
# write_text 会直接抛 OSError: Invalid argument（实测团委新闻两篇「喜报」、
# 一篇歌会），入库随之中断——知识库里静默少掉几篇。
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """把标题/分类名变成跨平台可用的文件名（去非法字符、限制长度）。"""
    return _INVALID_FILENAME_RE.sub("_", name).strip(" .")[:80]


def _save_clean_text(article: ArticleMetadata) -> Path:
    """将清洗后的文本保存到 scraped_docs/{category}/{title}.txt"""
    # 安全文件名
    safe_title = _safe_filename(article.title)
    safe_cat = _safe_filename(article.category)

    cat_dir = _SCRAPED_DOCS_DIR / safe_cat
    cat_dir.mkdir(parents=True, exist_ok=True)

    header = (
        f"标题: {article.title}\n"
        f"日期: {article.publish_date}\n"
        f"来源: {article.source_site}\n"
        f"分类: {article.category}\n"
        f"原文链接: {article.url}\n"
        f"{'─' * 60}\n\n"
    )

    file_path = cat_dir / f"{safe_title}.txt"
    # 落盘的 .txt 也执行隐私脱敏（与入库内容保持一致）
    file_path.write_text(header + sanitize_privacy(article.clean_text), encoding="utf-8")
    return file_path


def _split_and_embed(
    articles: list[ArticleMetadata],
    vector_store: VectorStore,
    *,
    mark_fn=None,
) -> int:
    """
    将文章列表分块后存入 ChromaDB，**逐篇入库成功后才落「已爬」标记**。
    返回存入的 chunk 总数。

    mark_fn: 落标记的函数（默认 storage.mark_scraped），仅测试时注入替身。
    """
    splitter = RecursiveTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0

    for article in articles:
        # 构建带元数据前缀的文本（搜索时可体现来源）
        prefix = (
            f"【来源】{article.source_site} | 【分类】{article.category} | "
            f"【日期】{article.publish_date}\n"
            f"【标题】{article.title}\n"
        )
        full_text = prefix + sanitize_privacy(article.clean_text)

        chunks = splitter.split_text(full_text)
        if not chunks:
            continue

        metadatas = []
        # 年份标签：从发布日期的 ISO 格式中提取（如 "2026-08-05" → "2026"）
        year = (article.publish_date or "")[:4]
        for _ in chunks:
            metadatas.append({
                "source": article.title,          # 标签一：文件名（文章标题）
                "category": article.category,
                "source_site": article.source_site,
                "publish_date": article.publish_date,
                "date_int": date_to_int(article.publish_date),  # 数值日期，供 Chroma 区间过滤
                "year": year,                     # 标签二：发布年份
                "url": article.url,
            })

        ids = [f"scrape_{article.publish_date}_{article.content_hash[:8]}_{i}"
               for i in range(len(chunks))]

        vector_store.save_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        total_chunks += len(chunks)

        # ── 入库成功后才落「已爬」标记（逐篇，不是逐分类）──
        # 顺序不能反：这条标记就是「这篇已经进知识库」的承诺，下一轮爬取据此跳过它。
        # 曾经标记写在爬取阶段、入库却是整个分类跑完才做，于是「标记了但没入库」
        # 的窗口横跨**整个分类**（实测一个分类 6~20 分钟）——期间中断就永久丢正文。
        # 改成逐篇落标记后，这个窗口缩到单篇的一次 save_documents 之内；
        # 真要在这里挂掉，也只是下一篇重爬一遍，不会再出现静默空洞。
        (mark_fn or mark_scraped)(
            url=article.url,
            title=article.title,
            publish_date=article.publish_date,
            category=article.category,
            source_site=article.source_site,
            content_hash=article.content_hash,
        )

    return total_chunks


# ── 内容变更复检 ──────────────────────────────────────────
# 增量爬取按 URL 去重（storage.get_scraped_urls / is_scraped），学校改了正文但
# 不改 URL 时，库里会一直是旧版本，且**没有任何征兆**。这里按天抽检最近入库的
# 文章：重抓 → 比 content_hash → 变了就删旧块、重嵌入。

def _env_int(name: str, default: int) -> int:
    """读 .env 的整数配置。在任务运行时读而不是模块导入时读——
    load_dotenv() 发生在 app 启动阶段，晚于 campus_scraper 的导入。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("环境变量 %s 不是整数，用默认值 %d", name, default)
        return default


def _remove_stale_text(old_title: str, category: str, keep_title: str) -> None:
    """标题改了 → 旧 .txt 文件删掉，避免 scraped_docs 里留两份同文不同名。"""
    old_safe = _safe_filename(old_title or "")
    if not old_safe or old_safe == _safe_filename(keep_title or ""):
        return
    path = _SCRAPED_DOCS_DIR / _safe_filename(category) / f"{old_safe}.txt"
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("旧正文文件删除失败: %s → %s", path, e)


async def _recheck_recent_changes(
    category_name: str,
    source_site: str,
    session: aiohttp.ClientSession,
    vector_store: VectorStore,
    *,
    db_path: str | None = None,
) -> tuple[int, int, int]:
    """抽检最近入库文章，返回 (复检篇数, 变更篇数, 重嵌入块数)。

    顺序很重要：**先嵌入新块、后删旧块**。反过来一旦嵌入失败，文章就只剩
    一条「已爬」标记而没有正文，和 2026-10-11 丢 20 篇是同一类静默空洞。
    嵌入成功但删旧块失败时最多留一段时间的重复块（下轮按 id 再删），不会丢内容。
    """
    days = _env_int("SCRAPE_RECHECK_DAYS", SCRAPE_RECHECK_DAYS_DEFAULT)
    limit = _env_int("SCRAPE_RECHECK_MAX", SCRAPE_RECHECK_MAX_DEFAULT)
    if days <= 0 or limit <= 0:
        return 0, 0, 0

    rows = get_recent_scraped(category_name, days=days, limit=limit, db_path=db_path)
    checked = changed = chunks_total = 0

    for row in rows:
        url = row["url"]
        # 复检也是一次真实请求，同样遵守 4~9 秒随机间隔
        await asyncio.sleep(_random_delay())
        try:
            article = await scrape_article(
                url=url,
                category=category_name,
                source_site=source_site,
                session=session,
                fallback_date=row.get("publish_date", ""),
            )
        except Exception as e:
            logger.warning("复检失败（保留旧内容）: %s → %s", url, e)
            continue

        # 抓不到 / 死链 / 超日期范围：不能拿「没有正文」去覆盖库里已有内容
        if article is None or not article.content_hash:
            continue

        checked += 1
        if article.content_hash == row.get("content_hash"):
            continue

        try:
            old_ids = vector_store.get_ids_by_url(url)
            _save_clean_text(article)
            chunks = _split_and_embed([article], vector_store)
        except Exception:
            logger.exception("正文变更重嵌入失败（保留旧块）: %s", url)
            continue

        if old_ids:
            try:
                vector_store.delete_ids(old_ids)
            except Exception:
                logger.exception("旧块删除失败（下轮复检会再试）: %s", url)

        _remove_stale_text(row.get("title", ""), category_name, article.title)
        changed += 1
        chunks_total += chunks
        logger.info(
            "🔁 [%s] 正文变更，已重嵌入 %d 块: %s",
            category_name, chunks, article.title[:50],
        )

    if checked:
        logger.info("🔎 [%s] 变更复检 %d 篇，发现变更 %d 篇",
                    category_name, checked, changed)
    return checked, changed, chunks_total


# ═══════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════

async def run_scrape_pipeline(
    max_pages: int = MAX_LIST_PAGES,
    vector_store: VectorStore | None = None,
    progress_callback=None,
) -> dict:
    """
    运行完整爬取管线：爬取 → 清洗 → 保存 → 分块 → 嵌入。

    参数:
        max_pages: 每个分类最多爬几页列表
        vector_store: 可复用的 VectorStore 实例（不传则新建）
        progress_callback: async callback(status_dict) 用于前端进度更新

    返回: {"total_articles": int, "total_chunks": int,
           "total_checked": int, "total_updated": int, "errors": list}
    total_chunks 含变更复检重嵌入的块；total_checked/total_updated 是复检口径。
    """
    if vector_store is None:
        vector_store = VectorStore(
            db_path=str(_PROJECT_ROOT / "chroma_db"),
            collection_name=COLLECTION_NAME,
        )

    errors = []
    total_articles = 0
    total_chunks = 0
    total_checked = 0
    total_updated = 0

    # ── Step 1: 逐分类爬取 → 立即清洗/保存/嵌入 ──
    # 设计：每爬完一个分类马上落盘入库（而非全部爬完再统一入库）。
    # 真正兜住中断的是「入库成功才落已爬标记」这条不变量（见 _split_and_embed）：
    # 中途被打断最多白爬一遍，不会留下"标记了但没内容"的静默空洞。
    if progress_callback:
        await progress_callback({"status": "scraping", "message": "正在爬取学校官网..."})

    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        for category_cfg in CATEGORIES:
            category_name = category_cfg["name"]
            try:
                articles = await scrape_category(
                    category_cfg, session, max_pages=max_pages,
                )
            except Exception as e:
                logger.exception("❌ [%s] 爬取出错: %s", category_name, e)
                errors.append(f"爬取出错 [{category_name}]: {e}")
                continue

            if articles:
                if progress_callback:
                    await progress_callback({
                        "status": "processing",
                        "message": f"正在处理: {category_name}（{len(articles)} 篇）...",
                    })

                # 保存清洗后的 .txt
                for article in articles:
                    try:
                        _save_clean_text(article)
                    except Exception as e:
                        logger.error("保存文件失败: %s → %s", article.title, e)
                        errors.append(f"保存失败 [{article.title}]: {e}")

                # 分块 + 嵌入
                try:
                    chunks = _split_and_embed(articles, vector_store)
                    total_chunks += chunks
                    total_articles += len(articles)
                except Exception as e:
                    logger.exception("嵌入失败: %s", e)
                    errors.append(f"嵌入失败 [{category_name}]: {e}")
            else:
                logger.info("⏭ [%s] 无新增文章，跳过入库", category_name)

            # ── 内容变更复检（与本次有无新文章无关）──
            # 新文章按 URL 去重跳过的旧文章，正文可能已被学校改过。
            try:
                checked, updated, chunks = await _recheck_recent_changes(
                    category_name,
                    category_cfg.get("site_label", ""),
                    session,
                    vector_store,
                )
                total_checked += checked
                total_updated += updated
                total_chunks += chunks
            except Exception as e:
                logger.exception("变更复检出错 [%s]: %s", category_name, e)
                errors.append(f"变更复检出错 [{category_name}]: {e}")

            if progress_callback:
                await progress_callback({
                    "status": "done_category",
                    "message": f"✅ {category_name}: 新增 {len(articles)} 篇 → 累计 {total_chunks} 块",
                })

    if progress_callback:
        await progress_callback({
            "status": "complete",
            "message": (
                f"🎉 全部完成: {total_articles} 篇新文章, {total_chunks} 文本块"
                f"，复检 {total_checked} 篇、更新 {total_updated} 篇"
            ),
        })

    return {
        "total_articles": total_articles,
        "total_chunks": total_chunks,
        "total_checked": total_checked,
        "total_updated": total_updated,
        "errors": errors,
    }


def get_knowledge_base_stats(
    vector_store: VectorStore | None = None,
) -> KnowledgeBaseStats:
    """
    查询 ChromaDB 获取知识库统计。
    不传 vector_store 则新建（延迟加载 Embedding 模型 ~420MB）。
    """
    if vector_store is None:
        vector_store = VectorStore(
            db_path=str(_PROJECT_ROOT / "chroma_db"),
            collection_name=COLLECTION_NAME,
        )

    stats = KnowledgeBaseStats()
    stats.total_chunks = vector_store.count()

    # ── 按分类统计 ──
    for cat_cfg in CATEGORIES:
        cat_name = cat_cfg["name"]
        count = vector_store.count_by_category(cat_name)
        if count > 0:
            stats.category_counts[cat_name] = count

    # ── 对不上的块数 ──
    # total_chunks 是整个 collection，category_counts 只覆盖上面那 13 个爬虫分类。
    # 两者之差 > 0 就意味着库里有"来源不明"的内容（手工写入、旧分类残留、
    # 改了 category 标签的迁移……）。这个数正常应当是 0，一旦非 0 就是知识库
    # 被污染的信号，必须查——不能让它继续当个看不见的差额存在。
    stats.unaccounted_chunks = (
        stats.total_chunks
        - sum(stats.category_counts.values())
    )

    # 去重统计文章数（按 source 字段）
    try:
        all_data = vector_store.collection.get()
        sources = set()
        for meta in (all_data.get("metadatas") or []):
            if meta and meta.get("source"):
                sources.add(meta["source"])
        stats.total_articles = len(sources)
    except Exception:
        stats.total_articles = 0

    return stats
