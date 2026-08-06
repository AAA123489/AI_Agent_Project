"""
campus_scraper/config.py — 爬取目标、速率、日期过滤等配置常量
"""

# ── 爬取速率（随机抖动模拟真人，避免固定间隔被防火墙识别） ──
REQUEST_DELAY_MIN = 4.0              # 最小间隔
REQUEST_DELAY_MAX = 9.0              # 最大间隔
# 实际延迟 = random.uniform(MIN, MAX)，每步都不一样

# ── 批量间长休息（模拟真人"看几篇 → 歇一会 → 再来看"的节奏） ──
BATCH_REST_MIN = 180                 # 每页完成后最小休息秒数（3分钟）
BATCH_REST_MAX = 480                 # 最大休息秒数（8分钟）
# 每爬完一页列表，随机休息 3~8 分钟，再翻下一页
REQUEST_TIMEOUT_SECONDS = 30         # 单次请求超时
MAX_RETRIES = 2                      # 失败重试次数
RETRY_BACKOFF_BASE = 5               # 重试退避基数（5s → 10s）

# ── 日期过滤 ──
DATE_FILTER_DAYS = 730               # 仅保留近 2 年的内容

# ── 每类最多爬取页数（列表页） ──
MAX_LIST_PAGES = 10                  # 每类最多 10 页，约 200 篇

# ── HTTP 请求头 ──
USER_AGENT = (
    "Mozilla/5.0 (compatible; CampusInfoBot/1.0; "
    "For educational research; contact via school email)"
)

# ── 数据库路径（基于本文件位置，避免相对路径跑偏） ──
from pathlib import Path
SCRAPER_DB_PATH = str(Path(__file__).resolve().parent / "scraper_cache.db")

# ═══════════════════════════════════════════════════════
# 爬取目标
# ═══════════════════════════════════════════════════════

CATEGORIES = [
    # ── 主站 ──
    {
        "name": "通知公告",
        "icon": "📢",
        "base_url": "https://www.hait.edu.cn",
        "list_path": "/tzgg.htm",
        "is_subsite": False,
        "site_label": "河南工学院",
    },
    {
        "name": "学校新闻",
        "icon": "📰",
        "base_url": "https://www.hait.edu.cn",
        "list_path": "/xyxw.htm",
        "is_subsite": False,
        "site_label": "河南工学院",
    },
    # ── 教务处 ──
    {
        "name": "教务处通知",
        "icon": "📚",
        "base_url": "https://jwc.hait.edu.cn",
        "list_path": "/index/tzgg.htm",
        "is_subsite": True,
        "site_label": "教务处",
    },
    # ── 智能工程学院 ──
    {
        "name": "智能工程学院通知",
        "icon": "🤖",
        "base_url": "https://znxy.hait.edu.cn",
        "list_path": "/index/tzgg.htm",
        "is_subsite": True,
        "site_label": "智能工程学院",
    },
    {
        "name": "智能工程学院新闻",
        "icon": "🤖",
        "base_url": "https://znxy.hait.edu.cn",
        "list_path": "/index/xyxw.htm",
        "is_subsite": True,
        "site_label": "智能工程学院",
    },
    # ── 电缆工程学院 ──
    {
        "name": "电缆工程学院通知",
        "icon": "🔌",
        "base_url": "https://dlxy.hait.edu.cn",
        "list_path": "/index/tzgg.htm",
        "is_subsite": True,
        "site_label": "电缆工程学院",
    },
    {
        "name": "电缆工程学院新闻",
        "icon": "🔌",
        "base_url": "https://dlxy.hait.edu.cn",
        "list_path": "/index/xyxw.htm",
        "is_subsite": True,
        "site_label": "电缆工程学院",
    },
    # ── 外国语学院 ──
    {
        "name": "外国语学院通知",
        "icon": "🌐",
        "base_url": "https://wyx.hait.edu.cn",
        "list_path": "/index/tzgg.htm",
        "is_subsite": True,
        "site_label": "外国语学院",
    },
    {
        "name": "外国语学院新闻",
        "icon": "🌐",
        "base_url": "https://wyx.hait.edu.cn",
        "list_path": "/index/xyxw.htm",
        "is_subsite": True,
        "site_label": "外国语学院",
    },
    # ── 马克思主义学院 ──
    {
        "name": "马克思主义学院通知",
        "icon": "📖",
        "base_url": "https://mksxy.hait.edu.cn",
        "list_path": "/index/tzgg.htm",
        "is_subsite": True,
        "site_label": "马克思主义学院",
    },
    {
        "name": "马克思主义学院新闻",
        "icon": "📖",
        "base_url": "https://mksxy.hait.edu.cn",
        "list_path": "/index/xyxw.htm",
        "is_subsite": True,
        "site_label": "马克思主义学院",
    },
    # ── 车辆与交通工程学院 ──
    {
        "name": "车辆与交通工程学院通知",
        "icon": "🚗",
        "base_url": "https://qcx.hait.edu.cn",
        "list_path": "/index/tzgg.htm",
        "is_subsite": True,
        "site_label": "车辆与交通工程学院",
    },
    # ── 继续教育学院 ──
    {
        "name": "继续教育学院通知",
        "icon": "🎓",
        "base_url": "https://cjb.hait.edu.cn",
        "list_path": "/tzgg.htm",
        "is_subsite": True,
        "site_label": "继续教育学院",
    },
    # ── 材料科学与工程学院 ──
    {
        "name": "材料科学与工程学院新闻",
        "icon": "🔬",
        "base_url": "https://clxnew.hait.edu.cn",
        "list_path": "/index/xyxw.htm",
        "is_subsite": True,
        "site_label": "材料科学与工程学院",
    },
    # ── 学生处（JSP 网站群系统，URL 格式与动易 CMS 不同） ──
    {
        "name": "学生处通知",
        "icon": "🎓",
        "base_url": "https://xsc.hait.edu.cn",
        "list_path": "/list.jsp?urltype=tree.TreeTempUrl&wbtreeid=1044",
        "is_subsite": True,
        "site_label": "学生处",
        "cms_type": "jsp",
    },
    # ── 团委（动易 CMS，路径已修正） ──
    {
        "name": "团委通知",
        "icon": "🎯",
        "base_url": "https://tw.hait.edu.cn",
        "list_path": "/tzgg.htm",
        "is_subsite": True,
        "site_label": "团委",
    },
    {
        "name": "团委新闻",
        "icon": "🎯",
        "base_url": "https://tw.hait.edu.cn",
        "list_path": "/xwdt.htm",
        "is_subsite": True,
        "site_label": "团委",
    },
]
