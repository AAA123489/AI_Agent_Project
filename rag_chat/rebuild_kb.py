"""
rebuild_kb.py — 消融实验：从 scraped_docs/ 重建知识库
===============================
读取 scraped_docs/ 下所有 .txt → 解析 header（标题/日期/来源/分类/原文链接）
→ 分块 → 清空 ChromaDB → 重建入库。

用法:
    python rebuild_kb.py                       # 默认 chunk_size=500, chunk_overlap=50
    python rebuild_kb.py --chunk-size 300      # 调小块大小，观察检索差异
    python rebuild_kb.py --chunk-overlap 100   # 调重叠窗口
    python rebuild_kb.py --chunk-size 1000 --chunk-overlap 100

说明:
- 每个 .txt 的正文已在爬取时脱敏（删个人手机号/邮箱，保留座机），此处再兜底一次
- 入库 metadata 打两个标签：source=文件名（.txt 文件名），year=发布年份
- 每次运行前清空整个 collection —— 消融实验从零开始，参数改了重跑即可
"""

import argparse
import hashlib
import io
import logging
import os
import re
import sys
from pathlib import Path

# Windows 控制台 GBK 兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 离线加载嵌入模型：HF_HUB_OFFLINE 跳过 hf-mirror.com 网络校验。
# 阶段1 后期 hf-mirror 网络异常，模型加载时的 HTTP 校验挂起/segfault；
# 模型本地缓存齐全，离线加载更稳更快。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vector_store import VectorStore
from text_splitter import RecursiveTextSplitter, sanitize_privacy

COLLECTION_NAME = "my_rag_collection"
SCRAPED_DIR = _PROJECT_ROOT / "scraped_docs"


def parse_txt(path: Path) -> dict | None:
    """解析落盘 .txt 的 header，返回 {title, date, source_site, category, url, body}"""
    text = path.read_text(encoding="utf-8", errors="replace")
    header = {}
    body_lines = []
    seen_sep = False
    for line in text.splitlines():
        if seen_sep:
            body_lines.append(line)
            continue
        # 分隔线（60 个 ─）之后的都是正文
        if line.strip().startswith("─") and not line.strip().strip("─"):
            seen_sep = True
            continue
        m = re.match(r"^(标题|日期|来源|分类|原文链接):\s*(.*)$", line)
        if m:
            header[m.group(1)] = m.group(2).strip()
    if not header.get("标题"):
        return None
    return {
        "title": header["标题"],
        "date": header.get("日期", ""),
        "source_site": header.get("来源", ""),
        "category": header.get("分类", ""),
        "url": header.get("原文链接", ""),
        "body": "\n".join(body_lines).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验：从 scraped_docs/ 重建知识库")
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小（默认 500）")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="重叠窗口（默认 50）")
    args = parser.parse_args()

    files = sorted(SCRAPED_DIR.rglob("*.txt"))
    if not files:
        print("❌ scraped_docs/ 下没有 .txt，请先运行 crawl_ablation.py 爬取")
        return
    print(f"📂 找到 {len(files)} 个 .txt，开始解析...")

    docs = []
    for p in files:
        doc = parse_txt(p)
        if doc:
            doc["_path"] = p
            docs.append(doc)
    if not docs:
        print("❌ 没有解析出任何文章（检查 .txt 是否有 header）")
        return

    print(f"✅ 成功解析 {len(docs)} 篇"
          f"（chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}）")

    vs = VectorStore(
        db_path=str(_PROJECT_ROOT / "chroma_db"),
        collection_name=COLLECTION_NAME,
    )
    before = vs.count()
    vs.clear_all()

    splitter = RecursiveTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    total = 0
    for doc in docs:
        prefix = (
            f"【来源】{doc['source_site']} | 【分类】{doc['category']} | "
            f"【日期】{doc['date']}\n"
            f"【标题】{doc['title']}\n"
        )
        # 入库前再兜底脱敏一次
        body = sanitize_privacy(doc["body"])
        chunks = splitter.split_text(prefix + body)
        if not chunks:
            continue

        year = (doc["date"] or "")[:4]
        metadatas = [{
            # 标签一：相对路径（分类/文件名）。不能用文件名 stem——跨分类同名文档
            # source 会撞车，导致补块时两篇被当成一篇（如 学生处通知 vs 通知公告）。
            "source": doc["_path"].relative_to(SCRAPED_DIR).as_posix(),
            "category": doc["category"],
            "source_site": doc["source_site"],
            "publish_date": doc["date"],
            "year": year,                   # 标签二：发布年份
            "url": doc["url"],
        } for _ in chunks]

        # ID 哈希用「相对文件路径」而非标题：跨分类同名文档（标题相同）会因 sha256(title)
        # 撞 ID 被 Chroma 后写覆盖，导致整篇文档丢失（如 学生处通知 vs 通知公告 的同名通知）。
        # 用路径哈希可让不同分类下的同名文档都入库。
        rel_path = doc["_path"].relative_to(SCRAPED_DIR).as_posix()
        doc_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8]
        ids = [f"rebuild_{doc_hash}_{i}" for i in range(len(chunks))]

        vs.save_documents(chunks, metadatas, ids)
        total += len(chunks)
        print(f"  📄 {doc['_path'].stem[:36]} → {len(chunks)} 块")

    print(f"\n🎉 重建完成: {len(docs)} 篇 → {total} 块"
          f"（chunk_size={args.chunk_size}, overlap={args.chunk_overlap}）")
    print(f"   collection 总数: {vs.count()}（重建前 {before}）")


if __name__ == "__main__":
    main()
