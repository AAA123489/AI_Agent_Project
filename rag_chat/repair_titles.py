"""
repair_titles.py — 修复历史数据里的坏标题
=========================================
背景（2026-10-19）：
  两类坏标题会让 source 失去定位能力（引用来源、同源回填、题库标注全靠它）：

  1. 栏目名当标题：电缆工程学院等子站把栏目名放在 <h1>（如「通知公告」），
     真正的文章标题只在 <title>。`parse_html` 原先优先取 h1，导致 29 篇文章
     / 144 块的 source 全是栏目名。
  2. 腰斩标题：`_strip_site_suffix` 曾按第一个连字符截断（2026-10-17 已修），
     但更早由 rebuild_kb.py 入库的块仍保留「关于组织申报2025」这类半截标题。

  解析器已修复，但库里已有块不会自己变。本脚本用修好的解析器重抓这些文章，
  重写 Chroma 元数据/正文前缀与 .txt 文件。

为什么必须重抓而不是只改元数据：
  块文本里带 `【标题】...` 前缀，只改 metadata 会让正文和 source 不一致；
  重新分块+嵌入才能保证引用、检索、落盘三处一致。

顺序与安全：
  - 先抓取、解析、校验（新标题非栏目名）→ 全部通过后才删除旧块；
  - 删除旧块是必须的：块 id 由 (发布日期, 内容 hash, 序号) 生成，标题不在
    hash 里，新旧块 id 相同，直接 add 会撞 id；
  - 删除后若嵌入失败，该篇会短暂从库里消失。脚本先做一次嵌入预热，且失败时
    打印 URL；重跑脚本时用 --urls-file 指定即可恢复。

用法:
    python repair_titles.py --dry-run   # 只抓取对比，不改库
    python repair_titles.py             # 修复全部
    python repair_titles.py --limit 3   # 先试 3 篇
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campus_scraper.html_parser import (  # noqa: E402
    _looks_like_column_heading,
    parse_html,
)
from campus_scraper.models import ArticleMetadata  # noqa: E402
from campus_scraper.pipeline import (  # noqa: E402
    COLLECTION_NAME,
    _remove_stale_text,
    _save_clean_text,
    _split_and_embed,
)
from campus_scraper.scraper import fetch_page  # noqa: E402
from src.vector_store import VectorStore, _get_embedding_fn  # noqa: E402


_TRUNCATED_RE = re.compile(r"(19|20)\d{2}$")


def _source_basename(source: str) -> str:
    """rebuild_kb.py 的 source 是「分类/标题.txt」，取标题部分再判断。"""
    base = (source or "").rsplit("/", 1)[-1]
    return base[:-4] if base.endswith(".txt") else base


def _is_broken_title(source: str) -> bool:
    """source 是否栏目名 / 被腰斩 / 从正文里误抓的残句。"""
    base = _source_basename(source).strip()
    if not base:
        return False
    if _looks_like_column_heading(base):
        return True
    # 「关于组织申报2025」「关于开展2024」：连字符截断留下的半句
    if _TRUNCATED_RE.search(base):
        return True
    # 「各相关部门：」「系统提示」：标题没解析出来，抓了正文首行/页面提示
    return base.endswith(("：", ":")) or base in {"系统提示"}


def _find_affected() -> dict[str, dict]:
    """从 Chroma 找出坏标题的 URL（按 URL 去重，带分类/站点/日期）。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
    col = client.get_collection(COLLECTION_NAME)
    data = col.get(include=["metadatas"])
    out: dict[str, dict] = {}
    for m in data["metadatas"]:
        if not m or not m.get("url"):
            continue
        source = m.get("source", "")
        if _is_broken_title(source):
            out.setdefault(m["url"], {
                "url": m["url"],
                "source": source,
                "category": m.get("category", ""),
                "source_site": m.get("source_site", ""),
                "publish_date": m.get("publish_date", ""),
            })
    return out


async def _repair_one(row: dict, *, dry_run: bool, store: VectorStore | None) -> str:
    """返回状态字符串：fixed / unchanged / still-generic / fetch-failed。"""
    html = await fetch_page(row["url"])
    if not html:
        return "fetch-failed"

    clean_text, meta = parse_html(html, row["url"])
    new_title = (meta.get("title") or "").strip()
    if not new_title or _is_broken_title(new_title):
        print(f"    解析后仍是坏标题或空: {new_title!r}")
        return "still-broken"
    if new_title == row["source"]:
        return "unchanged"
    if not clean_text.strip():
        return "still-generic"

    print(f"    {row['source']} → {new_title[:60]}")
    if dry_run:
        return "fixed"

    article = ArticleMetadata(
        title=new_title,
        publish_date=meta.get("publish_date") or row["publish_date"],
        category=row["category"],
        source_site=row["source_site"],
        url=row["url"],
        clean_text=clean_text,
        content_hash=meta.get("content_hash", ""),
    )
    old_ids = store.get_ids_by_url(row["url"])
    _save_clean_text(article)          # 先落盘，失败也不至于两边都没有
    store.delete_ids(old_ids)          # 新旧 id 可能相同，必须先删
    _split_and_embed([article], store)  # 内部成功后才更新「已爬」标记
    # rebuild 来源的旧标题形如「分类/标题.txt」，落盘文件名只取标题部分
    _remove_stale_text(_source_basename(row["source"]), row["category"], new_title)
    return "fixed"


async def main() -> None:
    ap = argparse.ArgumentParser(description="修复栏目名当标题的历史数据")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--urls-file", default="", help="只修 JSON 里列出的 URL（失败重跑用）")
    args = ap.parse_args()

    affected = _find_affected()
    if args.urls_file:
        wanted = set(json.loads(Path(args.urls_file).read_text(encoding="utf-8")))
        affected = {u: v for u, v in affected.items() if u in wanted}
    rows = sorted(affected.values(), key=lambda r: r["url"])
    if args.limit:
        rows = rows[: args.limit]

    print(f"待修复 {len(rows)} 篇")
    if not rows:
        return

    store = None
    if not args.dry_run:
        store = VectorStore(
            db_path=str(PROJECT_ROOT / "chroma_db"),
            collection_name=COLLECTION_NAME,
        )
        # 预热嵌入模型：把「模型加载失败」挡在删任何旧块之前
        _get_embedding_fn()(["预热"])

    stats: dict[str, int] = {}
    failed: list[str] = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['category']} {row['url']}")
        try:
            status = await _repair_one(row, dry_run=args.dry_run, store=store)
        except Exception as e:  # 单篇失败不中断，URL 留给重跑
            print(f"    失败: {e}")
            status = "error"
        stats[status] = stats.get(status, 0) + 1
        if status in {"error", "fetch-failed"}:
            failed.append(row["url"])
        await asyncio.sleep(1)

    print("\n结果:", ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if failed:
        out = PROJECT_ROOT / "repair_failed_urls.json"
        out.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"失败 URL 已写入 {out}，可用 --urls-file 重跑")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    asyncio.run(main())
