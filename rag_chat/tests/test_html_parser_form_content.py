"""
test_html_parser_form_content.py — 「正文被包在 <form> 里」回归测试
==========================================================
背景（2026-10-11）：动易的部分模板把**整篇正文**放在 <form> 内——实测
dlxy.hait.edu.cn 的 <div class="page_content">，890 个字 100% 落在其中的
<form> 里。而 _REMOVE_TAGS 原本无条件删 <form>，于是正文被清成 0 字 →
scrape_article 静默 return None → **连「已爬」标记都不写** → 这些文章每轮
重试一次、每次都失败，永远进不了知识库（那个栏目库里只有 8 篇）。

这个 bug 的可怕之处在于它**不报错**：日志里只有一句"跳过（无正文内容）"，
而失败的文章不会留下任何痕迹，所以看缓存库根本发现不了。

放这些测试是为了钉住两个方向：
  1. 扛着正文的 form 必须留下——否则整篇文章凭空消失
  2. 真正该删的短表单（查询/登录）仍要删掉——否则控件噪声混进正文
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from campus_scraper.html_parser import parse_html, _CONTENT_FORM_MIN_CHARS


def _filler(n: int) -> str:
    """造 n 个字的正文。"""
    return "河南工学院电缆工程学院学术讲座" * (n // 15 + 1)


def test_form_wrapping_article_body_is_kept():
    """正文整篇包在 <form> 里时必须照样抽出来（dlxy 真实模板形状）。

    改动前这条会拿到 0 字——文章就这么静默消失，且不留任何标记。
    """
    body = _filler(600)
    html = f"""<html><body>
      <div class="header clearfix">首页 学院概况 师资队伍</div>
      <div class="page_content">
        <form name="contentForm" method="post" action="">
          <p>高速铜缆的设计与挑战</p>
          <p>2026-10-10 16:57(浏览次数:)</p>
          <p>{body}</p>
        </form>
      </div>
    </body></html>"""

    text, _meta = parse_html(html, "https://dlxy.hait.edu.cn/info/1003/8470.htm")

    assert len(text) >= 600, f"正文被 form 吞掉了，只剩 {len(text)} 字"
    assert "高速铜缆的设计与挑战" in text
    assert _filler(600)[:30] in text


def test_short_control_form_is_still_removed():
    """短表单（查询/登录控件）仍然要删——判据是文字量，不是关键词。

    这个 form 的 name/class/id 都不含任何 _REMOVE_KEYWORDS，所以它只能靠
    文字量判据被认出来；如果哪天把阈值逻辑删了，这条会失败。
    """
    html = """<html><body>
      <div class="content">
        <form name="qform" method="get" action="/search">
          <input type="text" name="q"><input type="submit" value="查询">
        </form>
        <p>关于开展实验室安全检查的通知</p>
      </div>
    </body></html>"""

    text, _meta = parse_html(html, "https://www.hait.edu.cn/info/1.htm")

    assert "关于开展实验室安全检查的通知" in text
    assert "查询" not in text, "短控件表单没被删掉，噪声混进了正文"


def test_threshold_boundary():
    """阈值两侧的行为要分明：刚好到线留着，差一个字就删。"""
    just_over = _filler(_CONTENT_FORM_MIN_CHARS + 20)
    just_under = _filler(_CONTENT_FORM_MIN_CHARS - 60)

    html_over = f'<html><body><div class="content"><form><p>{just_over}</p></form></div></body></html>'
    html_under = f'<html><body><div class="content"><form><p>{just_under}</p></form></div></body></html>'

    text_over, _ = parse_html(html_over)
    text_under, _ = parse_html(html_under)

    assert just_over[:30] in text_over, "达到阈值的 form 被误删"
    assert just_under[:30] not in text_under, "未达阈值的 form 该删却没删"


def test_script_and_style_still_removed():
    """脚本/样式照旧要删——form 的例外不该把其它清理规则一起放水。"""
    html = """<html><body>
      <div class="content">
        <script>var secret = "脚本内容不该出现";</script>
        <style>.x { color: red; }</style>
        <p>真正的正文内容在这里，长度足够被当成正文。</p>
      </div>
    </body></html>"""

    text, _meta = parse_html(html, "https://www.hait.edu.cn/info/2.htm")

    assert "真正的正文内容在这里" in text
    assert "脚本内容不该出现" not in text
    assert "color: red" not in text


def test_plain_content_without_form_unchanged():
    """没有 form 的普通页面不受影响——这是绝大多数页面的形状。"""
    body = _filler(400)
    html = f"""<html><body>
      <div class="content">
        <p>关于2026年中秋节、国庆节假期学生安全教育管理工作的通知</p>
        <p>{body}</p>
      </div>
    </body></html>"""

    text, _meta = parse_html(html, "https://www.hait.edu.cn/info/3.htm")

    assert "关于2026年中秋节、国庆节假期学生安全教育管理工作的通知" in text
    assert body[:30] in text
