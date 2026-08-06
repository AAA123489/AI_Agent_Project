"""
campus_scraper/html_parser.py — HTML 解析 + 清洗
===============================
适配动易 Visual SiteBuilder CMS 的静态 .htm 页面。

职责：
- parse_html(html, url) → (clean_text, metadata_dict)
  解析文章详情页，提取标题、日期、正文
- extract_article_urls(html, base_url) → [{url, title, date}]
  解析列表页，提取文章链接 + 日期
"""

import hashlib
import logging
import re
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger("campus_scraper.parser")

# ── 要移除的标签（导航/广告/脚本/样式） ──
_REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "noscript", "iframe", "object", "embed", "form",
]

# ── 要移除的 class/id 关键词（包含这些关键词的容器会被移除） ──
_REMOVE_KEYWORDS = [
    "nav", "menu", "footer", "header", "sidebar", "banner",
    "search", "toolbar", "breadcrumb", "pagination", "pagebar",
    "copyright", "hotnews", "recommend", "related", "share",
]

# ── 日期正则 ──
_DATE_PATTERNS = [
    re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?"),
    re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
]


def _normalize_date(date_str: str) -> str:
    """将各种日期格式统一为 YYYY-MM-DD。"""
    if not date_str:
        return ""
    date_str = date_str.strip().replace(" ", "")
    for pat in _DATE_PATTERNS:
        m = pat.search(date_str)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return f"{y:04d}-{mo:02d}-{d:02d}"
            except ValueError:
                return ""
    return ""


def _should_remove(tag: Tag) -> bool:
    """判断一个标签是否应该被移除（导航、广告、侧边栏等）。"""
    if tag.name is None:
        return False

    # 按标签名
    if tag.name.lower() in _REMOVE_TAGS:
        return True

    # 按 class/id 关键词
    class_str = " ".join(tag.get("class", [])) if tag.get("class") else ""
    id_str = tag.get("id", "") or ""

    combined = (class_str + " " + id_str).lower()
    for kw in _REMOVE_KEYWORDS:
        if kw in combined:
            return True
    return False


def _clean_element(elem: Tag) -> None:
    """递归移除 elem 内所有应移除的子标签。"""
    # 收集要移除的子标签（不能边遍历边修改）
    to_remove = []
    for child in elem.descendants:
        if isinstance(child, Tag) and _should_remove(child):
            to_remove.append(child)
    for child in to_remove:
        child.decompose()


def _table_to_text(table: Tag) -> str:
    """将 HTML 表格转为 pipe 格式文本。"""
    rows = table.find_all("tr")
    if not rows:
        return ""

    lines = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        cell_texts = [c.get_text(strip=True) for c in cells]
        lines.append(" | ".join(cell_texts))

    return "\n".join(lines)


def _extract_main_content(soup: BeautifulSoup) -> str:
    """从文章页面提取正文纯文本。"""
    # ── 优先定位 vsb_content（动易 CMS 正文容器） ──
    content_div = soup.find(id="vsb_content")
    if content_div is None:
        # 备选：常见正文 class
        for cls in ["article_content", "content", "article-body", "artibody",
                     "newscontent", "conN"]:
            content_div = soup.find("div", class_=re.compile(cls, re.I))
            if content_div:
                break

    if content_div is None:
        # 最终 fallback：提 body 中 <article>/<main> 或整个 body（先清理再取文本）
        body = soup.find("body")
        if body is None:
            return ""
        # 优先找 <article> 或 <main> 语义标签
        for semantic in body.find_all(["article", "main"]):
            _clean_element(semantic)
            text = semantic.get_text("\n", strip=True)
            if len(text) > 100:
                return text
        # 备选：移除导航/侧边栏/页脚后再取 body 文本
        _clean_element(body)
        return body.get_text("\n", strip=True)

    # ── 清理导航/脚本 ──
    _clean_element(content_div)

    # ── 在块级元素前插入换行标记，保证段落结构 ──
    _BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                   "li", "tr", "section", "article", "blockquote"}
    for tag in content_div.find_all(list(_BLOCK_TAGS)):
        tag.insert_before(soup.new_string("\n"))

    # ── <br> → 换行 ──
    for br in content_div.find_all("br"):
        br.replace_with(soup.new_string("\n"))

    # ── 表格 → pipe 格式 ──
    for table in content_div.find_all("table"):
        table_text = _table_to_text(table)
        table.replace_with(soup.new_string(f"\n{table_text}\n"))

    # ── 图片 → alt 文本 ──
    for img in content_div.find_all("img"):
        alt = img.get("alt", "")
        img.replace_with(soup.new_string(f"[图片: {alt}]" if alt else ""))

    # ── 获取文本（不加分隔符，行内标签自然连接） ──
    text = content_div.get_text(separator="", strip=False)

    # ── 后处理：压缩空白 ──
    text = re.sub(r"[ \t]{2,}", " ", text)           # 压缩连续空格
    text = re.sub(r"\n{3,}", "\n\n", text)           # 压缩连续空行
    text = re.sub(r"\n +", "\n", text)               # 行首空格
    text = re.sub(r" +\n", "\n", text)               # 行尾空格
    # 修复中文标点粘连数字的问题（"设有\n20\n个教学\n单位"）
    # 将孤立的数字/标点行合并回上一行
    text = re.sub(r"\n(\d{1,3})\n", r" \1 ", text)   # 孤行数字
    text = re.sub(r"\n([，。；：、！？）\)】」])\n?", r"\1\n", text)  # 孤行中文标点
    text = re.sub(r"\n([,.;:!?)])", r"\1", text)      # 孤行英文标点
    # 重新压缩
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_html(html: str, url: str = "") -> tuple[str, dict]:
    """
    解析文章详情页 HTML，返回 (clean_text, metadata_dict)。

    metadata: {
        "title": str,
        "publish_date": str (YYYY-MM-DD),
        "source_url": str,
        "category": str,
        "content_hash": str,
    }
    """
    soup = BeautifulSoup(html, "lxml")

    # ── 提取标题 ──
    title = ""
    # 方式1: doc_title / infotitle 类
    for cls in ["doc_title", "infotitle", "article_title", "news_title"]:
        title_tag = soup.find(class_=re.compile(cls, re.I))
        if title_tag:
            title = title_tag.get_text(strip=True)
            break
    # 方式2: h1
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    # 方式3: title 标签（去除后缀 "-欢迎访问河南工学院"）
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            raw = title_tag.get_text(strip=True)
            title = re.sub(r"[-—\|].*$", "", raw).strip()

    # ── 提取日期 ──
    publish_date = ""
    # 方式1: doctime / infotime 类
    for cls in ["doctime", "infotime", "article_time", "news_time", "timestyle"]:
        date_tag = soup.find(class_=re.compile(cls, re.I))
        if date_tag:
            publish_date = _normalize_date(date_tag.get_text())
            if publish_date:
                break
    # 方式2: 在 vsb_content 附近搜索
    if not publish_date:
        vsb = soup.find(id="vsb_content")
        if vsb:
            siblings = vsb.find_all_previous(string=True, limit=20)
            for s in siblings:
                d = _normalize_date(s.strip())
                if d:
                    publish_date = d
                    break

    # ── 提取正文 ──
    clean_text = _extract_main_content(soup)

    # ── 内容哈希 ──
    content_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()

    return clean_text, {
        "title": title,
        "publish_date": publish_date,
        "source_url": url,
        "content_hash": content_hash,
    }


def extract_article_urls(html: str, base_url: str = "") -> list[dict]:
    """
    从列表页 HTML 提取文章 URL、标题、日期。

    动易 CMS 列表页特征：
    - 文章链接: <a class="c{数字}" href="info/{cat}/{id}.htm" title="...">
    - 日期: <span class="timestyle{数字}">YYYY-MM-DD</span>
    - 分页: 共{n}条  {current}/{total}

    返回: [{"url": str, "title": str, "date": str}, ...]
    """
    soup = BeautifulSoup(html, "lxml")
    articles = []

    # ── 模式 1: 动易 CMS —— info/xxx/xxx.htm ──
    info_pattern = re.compile(r"info/\d+/\d+\.htm", re.I)
    # ── 模式 2: JSP 网站群 —— content.jsp?urltype=news.NewsContentUrl&...wbnewsid=xxx ──
    jsp_pattern = re.compile(r"content\.jsp\?.*wbnewsid=\d+", re.I)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        is_info = info_pattern.search(href)
        is_jsp = jsp_pattern.search(href)

        if not (is_info or is_jsp):
            continue

        full_url = urljoin(base_url, href)
        title = (a_tag.get("title") or a_tag.get_text(strip=True))

        # ── 定位日期 ──
        date_str = ""
        if is_info:
            # 动易 CMS：在 a 标签所在行中找 timestyle span
            parent_row = a_tag.find_parent("tr")
            if parent_row:
                time_span = parent_row.find("span", class_=re.compile(r"timestyle\d*", re.I))
                if time_span:
                    date_str = _normalize_date(time_span.get_text())
            if not date_str:
                nearby = a_tag.find_next_sibling(string=True)
                if nearby:
                    date_str = _normalize_date(nearby.strip())
        elif is_jsp:
            # JSP 网站群：日期通常在链接附近的 <span> 或独立文本中
            parent_row = a_tag.find_parent("tr") or a_tag.find_parent("li")
            if parent_row:
                # JSP 列表页日期常见格式：<span>2026-08-06</span> 或 <font>2026-08-06</font>
                for tag_name in ("span", "font", "div"):
                    date_tag = parent_row.find(tag_name)
                    if date_tag:
                        d = _normalize_date(date_tag.get_text())
                        if d:
                            date_str = d
                            break
            if not date_str:
                # 在 a 标签后续兄弟节点中找
                for sib in a_tag.find_next_siblings(string=True, limit=3):
                    d = _normalize_date(sib.strip())
                    if d:
                        date_str = d
                        break

        articles.append({
            "url": full_url,
            "title": title,
            "date": date_str,
        })

    return articles


def extract_pagination(html: str) -> dict:
    """
    从列表页提取分页信息。

    返回: {"total": int, "current": int, "total_pages": int, "next_url": str|None}
    """
    soup = BeautifulSoup(html, "lxml")

    result = {"total": 0, "current": 1, "total_pages": 1, "next_url": None}

    # ── 动易 CMS：查找 "共{total}条  {current}/{total_pages}" 模式 ──
    text = soup.get_text()
    pagination_pattern = re.compile(r"共\s*(\d+)\s*条\s*(\d+)\s*/\s*(\d+)")
    m = pagination_pattern.search(text)
    if m:
        result["total"] = int(m.group(1))
        result["current"] = int(m.group(2))
        result["total_pages"] = int(m.group(3))

    # ── JSP 网站群：查找 "共有{n}条" 或 "共{n}页" 或页号列表 ──
    if result["total"] == 0:
        jsp_total = re.search(r"共[有]?\s*(\d+)\s*(条|页)", text)
        if jsp_total:
            if jsp_total.group(2) == "条":
                result["total"] = int(jsp_total.group(1))
            else:
                result["total_pages"] = int(jsp_total.group(1))

    # ── 查找"下页"/"下一页"链接（动易 + JSP 通用） ──
    for a_tag in soup.find_all("a", string=re.compile(r"下页|下一页|next|>|»", re.I)):
        href = a_tag.get("href")
        if href and href != "#":
            result["next_url"] = href
            break

    # ── JSP 备选：查找页号链接中的"当前页+1" ──
    if result["next_url"] is None:
        current_page = result["current"]
        # 查找 class="current" 或 style 标记的当前页码
        current_tag = soup.find("span", class_=re.compile(r"current|active|currentpage", re.I))
        if current_tag:
            try:
                current_page = int(current_tag.get_text(strip=True))
                result["current"] = current_page
            except ValueError:
                pass
        # 查找下一页号
        next_page_tag = soup.find("a", string=str(current_page + 1))
        if next_page_tag and next_page_tag.get("href"):
            result["next_url"] = next_page_tag["href"]

    return result
