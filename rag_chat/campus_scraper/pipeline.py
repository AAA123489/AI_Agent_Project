"""
campus_scraper/pipeline.py — 核心桥接模块
===============================
职责：
- run_scrape_pipeline()    爬取 → 清洗 → 保存 .txt → 分块 → 嵌入 → ChromaDB
- get_knowledge_base_stats()  从 ChromaDB 查询动态统计（含「用户上传」与差额）
- ingest_uploaded_files()    处理前端上传的文件
- remove_uploaded_file()     移除一篇用户上传的文档（爬虫语料拒删）
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

# ── 路径：确保能导入同目录模块 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vector_store import VectorStore
from text_splitter import RecursiveTextSplitter, sanitize_privacy

from .config import CATEGORIES, MAX_LIST_PAGES, USER_AGENT
from .scraper import scrape_category
from .models import ArticleMetadata, KnowledgeBaseStats

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

# ── 用户上传内容的分类标签 ──
# 上传的文档和爬虫抓的文档同住在 my_rag_collection 里，靠这个 category 区分。
# 它不在 CATEGORIES 里（那是 13 个爬虫分类），所以 get_knowledge_base_stats 里
# 必须单独数一次——否则 total_chunks（整个 collection）会比 category_counts
# 之和大出一截，而没人看得出来差在哪。2026-09-20 就是靠这个差额才发现
# 有 5 篇测试上传的文档（378 块）在库里躺了半个月。
UPLOAD_CATEGORY = "用户上传"


def _save_clean_text(article: ArticleMetadata) -> Path:
    """将清洗后的文本保存到 scraped_docs/{category}/{title}.txt"""
    # 安全文件名
    safe_title = article.title.replace("/", "_").replace("\\", "_").replace(":", "：")
    safe_title = safe_title[:80]  # 限制长度
    safe_cat = article.category.replace("/", "_")

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
) -> int:
    """
    将文章列表分块后存入 ChromaDB。
    返回存入的 chunk 总数。
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

    return total_chunks


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

    返回: {"total_articles": int, "total_chunks": int, "errors": list}
    """
    if vector_store is None:
        vector_store = VectorStore(
            db_path=str(_PROJECT_ROOT / "chroma_db"),
            collection_name=COLLECTION_NAME,
        )

    errors = []
    total_articles = 0
    total_chunks = 0

    # ── Step 1: 逐分类爬取 → 立即清洗/保存/嵌入 ──
    # 设计：每爬完一个分类马上落盘入库（而非全部爬完再统一入库），
    # 这样任何时刻中断/断电，已爬内容都已安全入库，不会被"已标记未入库"丢失。
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

            if not articles:
                logger.info("⏭ [%s] 无新增文章，跳过入库", category_name)
                continue

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

            if progress_callback:
                await progress_callback({
                    "status": "done_category",
                    "message": f"✅ {category_name}: {len(articles)} 篇 → {total_chunks} 块",
                })

    if progress_callback:
        await progress_callback({
            "status": "complete",
            "message": f"🎉 全部完成: {total_articles} 篇文章, {total_chunks} 文本块",
        })

    return {
        "total_articles": total_articles,
        "total_chunks": total_chunks,
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

    # ── 用户上传（不在 CATEGORIES 里，单独数） ──
    stats.uploaded_chunks = vector_store.count_by_category(UPLOAD_CATEGORY)

    # ── 对不上的块数 ──
    # total_chunks 是整个 collection，category_counts 只覆盖上面那 13 类 + 用户上传。
    # 两者之差 > 0 就意味着库里有"来源不明"的内容（手工写入、旧分类残留、
    # 改了 category 标签的迁移……）。这个数正常应当是 0，一旦非 0 就是知识库
    # 被污染的信号，必须查——不能让它继续当个看不见的差额存在。
    stats.unaccounted_chunks = (
        stats.total_chunks
        - sum(stats.category_counts.values())
        - stats.uploaded_chunks
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


async def ingest_uploaded_files(
    file_paths: list[str],
    vector_store: VectorStore | None = None,
) -> dict:
    """
    处理用户上传的文件（PDF/DOCX/TXT/MD）。

    复用项目现有的 document_parser。

    返回: {"total_chunks": int, "files_processed": int, "errors": list}
    """
    # 延迟导入避免循环依赖
    from document_parser import parse_pdf, parse_docx

    if vector_store is None:
        vector_store = VectorStore(
            db_path=str(_PROJECT_ROOT / "chroma_db"),
            collection_name=COLLECTION_NAME,
        )

    splitter = RecursiveTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0
    files_processed = 0
    errors = []

    for fp in file_paths:
        try:
            path = Path(fp)
            ext = path.suffix.lower()

            # ── 解析文件 ──
            if ext == ".pdf":
                text, meta = parse_pdf(str(path))
            elif ext == ".docx":
                text, meta = parse_docx(str(path))
            elif ext in (".txt", ".md"):
                text = path.read_text(encoding="utf-8", errors="replace")
                meta = {"author": "unknown", "year": ""}
            else:
                errors.append(f"不支持的文件类型: {ext}")
                continue

            if not text.strip():
                errors.append(f"文件无内容: {path.name}")
                continue

            # ── 隐私脱敏（删除个人手机号/邮箱，再分块） ──
            text = sanitize_privacy(text)

            # ── 分块 ──
            chunks = splitter.split_text(text)
            if not chunks:
                continue

            # ── 元数据 ──
            metadatas = []
            for _ in chunks:
                metadatas.append({
                    "source": path.name,
                    "category": UPLOAD_CATEGORY,
                    "source_site": "本地文件",
                    "publish_date": datetime.now().strftime("%Y-%m-%d"),
                    "url": f"file://{path.absolute()}",
                    "author": meta.get("author", "unknown"),
                    "year": meta.get("year", ""),
                })

            ids = [f"upload_{path.stem}_{i}" for i in range(len(chunks))]

            # 先删旧数据（同源覆盖）
            vector_store.delete_by_source(path.name)
            vector_store.save_documents(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )
            total_chunks += len(chunks)
            files_processed += 1

        except Exception as e:
            logger.exception("文件处理失败: %s", fp)
            errors.append(f"{Path(fp).name}: {e}")

    return {
        "total_chunks": total_chunks,
        "files_processed": files_processed,
        "errors": errors,
    }


async def remove_uploaded_file(
    filename: str,
    vector_store: VectorStore | None = None,
) -> dict:
    """
    从知识库移除一篇用户上传的文档（按 source 名，即文件名）。

    安全约束：只删 category == UPLOAD_CATEGORY 的块。爬虫抓来的正文一律拒删——
    一个 DELETE 请求不该有能力抹掉学校官网的语料，否则误删就是不可逆的
    （chroma_db 没有备份）。

    返回: {"found": bool, "deleted_chunks": int, "filename": str}
    抛出: ValueError —— 该 source 存在但不是用户上传的
    """
    if vector_store is None:
        vector_store = VectorStore(
            db_path=str(_PROJECT_ROOT / "chroma_db"),
            collection_name=COLLECTION_NAME,
        )

    got = vector_store.collection.get(where={"source": filename})
    ids = got.get("ids") or []
    if not ids:
        return {"found": False, "deleted_chunks": 0, "filename": filename}

    metas = got.get("metadatas") or []
    cats = {(m or {}).get("category") for m in metas}
    if cats != {UPLOAD_CATEGORY}:
        raise ValueError(
            f"「{filename}」不是用户上传的文档"
            f"（category={sorted(c for c in cats if c)}），拒绝删除"
        )

    vector_store.collection.delete(ids=ids)
    logger.info("已移除用户上传文档 %s（%d 块）", filename, len(ids))
    return {"found": True, "deleted_chunks": len(ids), "filename": filename}
