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

# class/id 里以关键词**开头**才算命中（token 边界），不做任意子串匹配：
# - 必须保留前缀匹配：动易的搜索容器是 class="searchstyle1092668"，没有分隔符；
# - 但子串匹配会把 research / unrelated / photonews（含 hotnews）误删成导航，
#   整篇正文跟着消失——0 字会被当成「无正文」静默跳过。
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# 以 share 开头但语义不是「分享组件」的词（共享/股权类），必须放行
_SHARE_FALSE_FRIENDS = {
    "shared", "shares", "sharing", "shareable", "shareholder", "shareholders",
}

# ── <form> 的例外阈值：文字量达到这个数的 form 当正文留着，不当控件删 ──
# 动易的部分模板把**整篇正文**包在 <form> 里（实测 dlxy.hait.edu.cn 的
# <div class="page_content"> 890 字，100% 落在其中的 <form> 内）。无条件删 form
# 会把文章删成 0 字，而 0 字会被 scrape_article 静默 return None、**连「已爬」标记
# 都不写**——于是这些文章每一轮都重试一次、每次都失败，永远进不了知识库。
# 判据用文字量而非 class/id 关键词：真正该删的搜索/登录表单只有几个字，
# 正文动辄几百字；而且搜索框本身另有 _REMOVE_KEYWORDS 里的 "search" 兜着
# （实测搜素容器 class="searchstyle1092668" 就是被关键词规则删掉的）。
_CONTENT_FORM_MIN_CHARS = 200

# ── 日期正则 ──
_DATE_PATTERNS = [
    re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?"),
    re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
]


_SITE_SUFFIX_MAX_LEN = 30


def _strip_site_suffix(raw: str) -> str:
    """去掉 <title> 末尾的站点后缀（如「-欢迎访问河南工学院」「-教务处（学位办）」）。

    不能按**第一个**分隔符截断：标题里的学年区间「2025-2026」也含连字符，
    按第一个截会把「关于公布2025-2026学年…的通知」砍成「关于公布2025」
    （实测教务处多篇标题被污染，引用来源里只剩半句）。
    这里只看最后一个分隔符后的尾巴，且仅当它像站点名（短、不含数字）时才截。
    """
    raw = raw.strip()
    m = re.search(r"[-—|]([^-—|]*)$", raw)
    if not m:
        return raw
    tail = m.group(1).strip()
    if not tail or len(tail) > _SITE_SUFFIX_MAX_LEN or re.search(r"\d", tail):
        return raw
    return raw[: m.start()].strip(" -—|")


# 栏目名常出现在详情页的 <h1> 里（电缆工程学院模板：<h1>通知公告</h1>，
# 真正的标题只在 <title>）。这类 h1 不能当文章标题，否则同栏目所有文章
# 共用一个 source，引用来源全变成「通知公告」，检索回填也会串文章。
_COLUMN_HEADINGS = {
    "通知公告", "学院新闻", "学校新闻", "新闻动态", "新闻中心", "最新动态",
    "学院通知", "学院动态", "学校通知", "首页", "通知", "新闻", "公告",
}
_COLUMN_SUFFIX_RE = re.compile(r"^[一-龥]{0,6}(公告|通知|新闻|动态|要闻|简报|讲座|活动)$")


def _looks_like_column_heading(text: str) -> bool:
    """判断 h1 文本是不是栏目名而不是文章标题。

    只判短文本（≤10 字）：「关于开展…活动的通知」这类完整标题不能被误判。
    """
    t = re.sub(r"\s+", "", text or "")
    if not t or len(t) > 10:
        return False
    return t in _COLUMN_HEADINGS or bool(_COLUMN_SUFFIX_RE.match(t))


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
        # <form> 例外：扛着正文的 form 不能删，见 _CONTENT_FORM_MIN_CHARS 注释
        if tag.name.lower() == "form":
            return len(tag.get_text(strip=True)) < _CONTENT_FORM_MIN_CHARS
        return True

    # 按 class/id 关键词
    class_str = " ".join(tag.get("class", [])) if tag.get("class") else ""
    id_str = tag.get("id", "") or ""

    combined = (class_str + " " + id_str).lower()
    # 先按非字母数字切 token（nav-menu → nav/menu），再对每个 token 做前缀匹配，
    # 既保住 searchstyle1092668 这类无分隔符的类名，又不再误伤 research。
    for token in _TOKEN_SPLIT_RE.split(combined):
        if not token or token in _SHARE_FALSE_FRIENDS:
            continue
        for kw in _REMOVE_KEYWORDS:
            if token.startswith(kw):
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
    # 方式2: h1（栏目页模板里 h1 是栏目名，如「通知公告」，只作为兜底）
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    # 方式3: title 标签（去除后缀 "-欢迎访问河南工学院"）
    # h1 缺失或只是栏目名时，用 <title>；仅当 <title> 明显更完整才覆盖，
    # 避免个别站点 <title> 只写站名、反而丢掉 h1 里的真标题。
    if not title or _looks_like_column_heading(title):
        title_tag = soup.find("title")
        if title_tag:
            candidate = _strip_site_suffix(title_tag.get_text(strip=True))
            if candidate and (not title or len(candidate) > len(title)):
                title = candidate

    # ── 提取日期 ──
    publish_date = ""
    # 方式1: doctime / infotime 类
    for cls in ["doctime", "infotime", "article_time", "article-date", "news_time", "timestyle"]:
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


# 列表页里「单独成链接」的日期（部分子站把日期也做成指向同一文章的 <a>）
_DATE_ONLY_RE = re.compile(
    r"^\s*(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?\s*$"
)
# 标题前缀日期：既支持 2026-09-18，也支持 2026年09月18日
_TITLE_DATE_RE = re.compile(
    r"^\s*(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?\s*(.*)$"
)


def _extract_date_from_row(a_tag: Tag) -> str:
    """在链接所在列表项（li/tr）里找发布日期。

    动易各子站模板差异很大：
    - 主站/多数学院：<span class="timestyle...">2026-10-16</span>
    - 车辆学院：日期是 li 里的裸文本节点
    - 教务处/智能工程学院：日期做成同 URL 的独立 <a>（由调用方按 URL 合并）
    这里统一扫 li/tr 的文本，避免逐站写选择器。
    """
    row = a_tag.find_parent(["li", "tr"])
    if row is not None:
        # 优先看带日期/时间语义的标签，避免标题里的年份干扰
        for tag in row.find_all(class_=re.compile(r"timestyle|date|time", re.I)):
            d = _normalize_date(tag.get_text(" ", strip=True))
            if d:
                return d
        d = _normalize_date(row.get_text(" ", strip=True))
        if d:
            return d

    parent = a_tag.parent
    if parent is not None and len(parent.get_text(" ", strip=True)) < 300:
        d = _normalize_date(parent.get_text(" ", strip=True))
        if d:
            return d

    for sib in a_tag.find_next_siblings(string=True, limit=3):
        d = _normalize_date(sib.strip())
        if d:
            return d
    return ""


def extract_article_urls(html: str, base_url: str = "") -> list[dict]:
    """
    从列表页 HTML 提取文章 URL、标题、日期。

    动易 CMS 列表页特征：
    - 文章链接: <a class="c{数字}" href="info/{cat}/{id}.htm" title="...">
    - 日期: <span class="timestyle{数字}">YYYY-MM-DD</span>
    - 分页: 共{n}条  {current}/{total}

    返回: [{"url": str, "title": str, "date": str}, ...]

    注意：同一 URL 可能出现多个 <a>（标题、图片、以及只含日期的独立链接），
    这里按 URL 合并成一条，日期互相补全，避免「一篇文章被当成两篇」。
    """
    soup = BeautifulSoup(html, "lxml")
    entries: dict[str, dict] = {}
    order: list[str] = []
    date_only: dict[str, str] = {}  # 同 URL 的「日期链接」→ 日期

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
        raw_title = (a_tag.get("title") or "").strip()
        link_text = a_tag.get_text(strip=True)
        label = raw_title or link_text

        # 日期独立成链接（教务处/智能工程学院等模板）：先记下来，稍后按 URL 合并
        if _DATE_ONLY_RE.match(label) or (not raw_title and _DATE_ONLY_RE.match(link_text)):
            d = _normalize_date(label or link_text)
            if d:
                date_only[full_url] = d
            continue

        title = label
        date_str = ""

        # 部分子站把日期直接写在链接文字里（如「2026-09-18 学院动态…」）。
        # 拆出来：既避免日期混进标题，也补上列表层的日期过滤依据。
        title_date = _TITLE_DATE_RE.match(title)
        if title_date:
            date_str = _normalize_date(title_date.group(0))
            title = title_date.group(4).strip()

        # 行内日期（span.timestyle / li 裸文本 / 紧邻文本节点）
        if not date_str:
            date_str = _extract_date_from_row(a_tag)

        entry = entries.get(full_url)
        if entry is None:
            entries[full_url] = {
                "url": full_url,
                "title": title,
                "date": date_str,
            }
            order.append(full_url)
        else:
            # 同一 URL 的多个链接：标题取更长的（标题链接 > 图片链接），日期取先拿到的
            if title and len(title) > len(entry["title"]):
                entry["title"] = title
            if date_str and not entry["date"]:
                entry["date"] = date_str

    # 用「日期链接」补全那些标题链接上没有日期的条目
    for url, d in date_only.items():
        if url in entries and not entries[url]["date"]:
            entries[url]["date"] = d

    return [entries[u] for u in order]


# 文章详情页链接。分页链接绝不会是这个形状——但「更多 >」「next」这类
# 首页区块箭头**可能**恰好指向最后一篇文章。实测材料学院首页没有分页，
# 解析器却把一篇资讯当成了「下一页」，害得爬虫多抓一次、还差点错过后续页。
_ARTICLE_HREF_RE = re.compile(r"info/\d+/\d+\.htm|wbnewsid=\d+", re.I)


def _is_article_href(href: str) -> bool:
    return bool(href and _ARTICLE_HREF_RE.search(href))


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
        if href and href != "#" and not _is_article_href(href):
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
        if next_page_tag and next_page_tag.get("href") \
                and not _is_article_href(next_page_tag["href"]):
            result["next_url"] = next_page_tag["href"]

    return result
