"""
test_scraper_dead_link.py — 「死链页面不能当文章入库」
=====================================================
背景（2026-10-17）：补回历史缺失文章时，动易对不存在的文章 ID 返回 **200**
和一个「没有找到文章ID」的导航壳。原 `scrape_article` 只检查正文是否为空，
而这个壳能提取出几十个字的栏目名（「通知公告 / 当前位置 / 本站首页」），
于是被当成一篇正常文章切块入库——标题还成了「教务处（学位办、语委办）」。

这类页面既不报错、也不为空，只能靠正文里的固定文案识别。
"""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper import scraper as scraper_mod

_DEAD_HTML = """
<html><head><title>教务处（学位办、语委办）</title></head>
<body>
  <div id="vsb_content">
    通知公告 当前位置: 本站首页 &gt; 通知公告 &gt; 正文 没有找到文章ID
  </div>
</body></html>
"""

_NORMAL_HTML = """
<html><head><title>关于开展2026年教坛新秀评选工作的通知-教务处</title></head>
<body>
  <div id="vsb_content">
    各单位：为做好2026年教坛新秀评选工作，现将有关事项通知如下。
    申报范围：承担教学任务、讲师以上职称、40周岁以下。
  </div>
</body></html>
"""


def _run(monkeypatch, html: str):
    async def fake_fetch(url, session=None):
        return html

    monkeypatch.setattr(scraper_mod, "fetch_page", fake_fetch)
    return asyncio.run(
        scraper_mod.scrape_article(
            url="https://jwc.hait.edu.cn/info/1887/11128.htm",
            category="教务处通知",
            source_site="教务处",
            session=None,
        )
    )


def test_dead_link_page_is_rejected(monkeypatch):
    """「没有找到文章ID」的导航壳不能入库。"""
    assert _run(monkeypatch, _DEAD_HTML) is None


def test_normal_page_still_parses(monkeypatch):
    """正常页面不能被死链判据误伤。"""
    article = _run(monkeypatch, _NORMAL_HTML)

    assert article is not None
    assert article.title == "关于开展2026年教坛新秀评选工作的通知"
    assert "教坛新秀" in article.clean_text
