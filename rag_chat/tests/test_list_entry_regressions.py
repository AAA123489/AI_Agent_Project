"""
test_list_entry_regressions.py — 列表页入口的两个回归点
=====================================================
2026-10 排查「17 个分类只有 14 个有内容」时发现：

1. 外国语学院子站把日期直接写进链接文字（「2026-09-18 学院动态…」）。
   原解析器把整串当标题、丢弃日期，导致标题带日期前缀、列表层无法按日期过滤。
2. 材料学院没有可用的静态列表页，只能从首页资讯区块取文章；首页混着多个
   频道，必须按 URL 片段（info/1147/）筛出「学院新闻」，否则会把通知等
   其它栏目的内容错标成新闻。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper.html_parser import extract_article_urls, extract_pagination
from campus_scraper.scraper import filter_list_articles


def test_date_prefix_in_link_text_is_split_out():
    """链接文字里的日期要拆到 date 字段，标题不能带日期。"""
    html = """
    <html><body><ul>
      <li><a href="info/1063/7123.htm">2026-09-18 探寻新乡非遗，感受牧野文化</a></li>
      <li><a href="info/1063/7108.htm">2026-09-02 外国语学院召开教学创新会议</a></li>
    </ul></body></html>
    """

    articles = extract_article_urls(html, "https://wyx.hait.edu.cn")

    assert [a["date"] for a in articles] == ["2026-09-18", "2026-09-02"]
    assert [a["title"] for a in articles] == [
        "探寻新乡非遗，感受牧野文化",
        "外国语学院召开教学创新会议",
    ]
    assert articles[0]["url"] == "https://wyx.hait.edu.cn/info/1063/7123.htm"


def test_title_without_date_is_unchanged():
    """没有日期前缀的标题不能被误伤。"""
    html = '<a href="info/1063/7000.htm">关于2026年秋季学期选课的通知</a>'

    articles = extract_article_urls(html, "https://wyx.hait.edu.cn")

    assert articles[0]["date"] == ""
    assert articles[0]["title"] == "关于2026年秋季学期选课的通知"


def test_filter_list_articles_by_url_contains():
    """首页混多个频道时，只保留配置指定的频道链接。"""
    articles = [
        {"url": "https://clxnew.hait.edu.cn/info/1147/8461.htm"},
        {"url": "https://clxnew.hait.edu.cn/info/1149/8460.htm"},
        {"url": "https://clxnew.hait.edu.cn/info/1147/8455.htm"},
    ]

    filtered = filter_list_articles(articles, "info/1147/")

    assert [a["url"].rsplit("/", 2)[-2] for a in filtered] == ["1147", "1147"]
    assert filter_list_articles(articles, None) == articles
    assert filter_list_articles(articles, "") == articles


def test_date_only_anchor_is_merged_into_same_url():
    """教务处模板把日期做成同 URL 的独立 <a>：不能算两篇，日期要合并回标题条。"""
    html = """
    <ul>
      <li>
        <a href="info/1887/11276.htm" title="关于开展2026年教坛新秀评选工作的通知">
          关于开展2026年教坛新秀评选工作的通知
        </a>
        <a href="info/1887/11276.htm">2026年10月16日</a>
      </li>
    </ul>
    """

    articles = extract_article_urls(html, "https://jwc.hait.edu.cn")

    assert len(articles) == 1
    assert articles[0]["url"] == "https://jwc.hait.edu.cn/info/1887/11276.htm"
    assert articles[0]["title"] == "关于开展2026年教坛新秀评选工作的通知"
    assert articles[0]["date"] == "2026-10-16"


def test_date_in_plain_li_text_is_extracted():
    """车辆学院列表的日期是 li 里的裸文本，不是 span。"""
    html = """
    <ul class="text-list">
      <li>
        <a href="info/1081/4761.htm" title="2023年车辆与交通工程学院困难生认定小组名单公示">
          2023年车辆与交通工程学院困难生认定小组名单公示
        </a>
        2023-09-11
      </li>
    </ul>
    """

    articles = extract_article_urls(html, "https://qcx.hait.edu.cn")

    assert len(articles) == 1
    assert articles[0]["date"] == "2023-09-11"
    # 标题里的「2023年」是内容，不是发布日期，不能被拆走
    assert articles[0]["title"].startswith("2023年车辆与交通工程学院")


def test_article_link_is_not_mistaken_for_next_page():
    """首页的「更多 >」可能指向一篇文章——不能把它当分页链接。

    2026-10 实测：材料学院首页（无分页）的箭头链接指向 info/1147/8461.htm，
    爬虫把它当成「下一页」去抓，既白跑一次又可能让真正有分页的站点提前停爬。
    """
    html = """
    <div class="pagination">
      <a href="info/1147/8461.htm">更多 &gt;</a>
      <a href="index/xyxw/2.htm">下一页</a>
    </div>
    """

    pagination = extract_pagination(html)

    assert pagination["next_url"] == "index/xyxw/2.htm", (
        "指向文章的链接被当成下一页 → 真正的分页链接会被跳过"
    )


def test_chinese_title_date_prefix_is_split_out():
    """标题前缀是 2026年09月18日 时也要拆成 date。"""
    html = '<a href="info/1063/7123.htm">2026年09月18日 探寻新乡非遗，感受牧野文化</a>'

    articles = extract_article_urls(html, "https://wyx.hait.edu.cn")

    assert articles[0]["date"] == "2026-09-18"
    assert articles[0]["title"] == "探寻新乡非遗，感受牧野文化"
