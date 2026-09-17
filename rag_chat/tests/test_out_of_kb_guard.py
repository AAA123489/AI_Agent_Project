"""测试库外主体闸门的**两层**校验（`_refuse_out_of_kb_school` + `_guard_tool_call`）。

## 钉的是哪个 bug

Q6「河北工学院的录取分数线是多少？」实测同一份代码跑两遍，一遍 60/60、一遍 50/60。
根因是这道题要过**两级 LLM**，回答那级某次把工具参数写成 `query='招生录取分数'`——
**把校名改写掉了**。闸门只看得到工具参数，于是没东西可拦，返回的本校（河南工学院）
分数被如实列进回答，评分 LLM 按「含基准表外数字即判幻觉」给 0。

`_guard_tool_call` 就是为了堵这个口子：用**用户原话**再拦一次。用户原话里校名跑不掉。

这个 bug 难得地**不需要 LLM 就能测**：闸门是纯字符串函数，输入定了输出就定了。
不 mock、不发网络请求。

注：本模块导入 app_backend（约 1 秒，reranker 惰性加载），比纯函数测试慢，但值得——
这是全项目唯一能把「LLM 改写掉校名」这条路径钉死的测试。
"""
from app_backend import _guard_tool_call, _refuse_out_of_kb_school


# ── 第一层：检索层闸门（工具参数）──


def test_工具参数带外校名时拒答():
    refuse = _refuse_out_of_kb_school("河北工学院 招生")

    assert refuse is not None
    assert "河北工学院" in refuse
    assert "【库外题拦截】" in refuse


def test_工具参数是主体校时放行():
    assert _refuse_out_of_kb_school("河南工学院食堂几点开门") is None


def test_没提任何校名时放行():
    """检索捞不到自然答"未找到"，无幻觉风险——不能因为没校名就拒答。"""
    assert _refuse_out_of_kb_school("招生录取分数") is None


# ── 第二层：用户原话闸门（本次修复）──


def test_校名被改写掉时用户原话仍能拦住():
    """Q6 的确切失败路径：LLM 把 query 改写成 '招生录取分数'（校名没了）。

    工具参数闸门放行 —— 这正是 50/60 那次发生的事；
    用户原话闸门必须拦住，否则 bug 复现。
    """
    # 工具参数已经被改写，第一层拦不住
    assert _refuse_out_of_kb_school("招生录取分数") is None

    # 用户原话拦得住
    refuse = _guard_tool_call("search_knowledge_base", "河北工学院的录取分数线是多少？")
    assert refuse is not None
    assert "河北工学院" in refuse


def test_拒答文案禁止补充本校信息():
    """光拒答不够——模型曾被诱导「作为补充说明」把本校分数列出来，文案必须堵死这条路。"""
    refuse = _guard_tool_call("search_knowledge_base", "河北工学院的录取分数线是多少？")

    assert "严禁补充" in refuse
    assert "河南工学院" in refuse


def test_主体校出现的混合问句不误拒():
    """「A和B哪个好」里主体校一出现就早退放行——不能把比较类问句也拒掉。"""
    assert _guard_tool_call("search_knowledge_base", "河南工学院和河北工学院哪个好") is None


def test_其他工具不受影响():
    """时间/计算/天气与知识库主体无关，不该被这道闸门碰。"""
    for name in ("get_current_time", "calculate", "get_weather", "get_knowledge_base_stats"):
        assert _guard_tool_call(name, "河北工学院的录取分数线是多少？") is None


def test_空问题不炸():
    assert _guard_tool_call("search_knowledge_base", "") is None
    assert _guard_tool_call("search_knowledge_base", None) is None
