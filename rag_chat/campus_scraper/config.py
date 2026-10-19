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
RETRY_BACKOFF_BASE = 5               # 重试退避基数，第 n 次重试等待 5×n 秒（线性，非指数）

# ── 日期过滤 ──
DATE_FILTER_DAYS = 730               # 仅保留近 2 年的内容

# ── 内容变更复检（URL 不变、正文被改） ──
# 增量爬取按 URL 去重：学校把名单/简章改了几个字，URL 不变，系统永远看不到，
# 库里一直是旧版本。每天从「最近入库」的文章里抽检若干篇重算正文哈希，
# 变了就删旧块重嵌。默认值只是兜底，实际以 .env 的 SCRAPE_RECHECK_DAYS /
# SCRAPE_RECHECK_MAX 为准（在爬取任务里惰性读取，保证 load_dotenv 已生效）。
SCRAPE_RECHECK_DAYS_DEFAULT = 7      # 只复检最近 7 天入库的文章；0 = 关闭
SCRAPE_RECHECK_MAX_DEFAULT = 5       # 每个分类每天最多复检 5 篇（每次都要发请求）

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
        # 2026-10 修正：/index/tzgg.htm 是旧模板页，最新只到 2023-09；
        # 站内现用 tzgg1.htm（2026 年通知都在这里）
        "name": "外国语学院通知",
        "icon": "🌐",
        "base_url": "https://wyx.hait.edu.cn",
        "list_path": "/index/tzgg1.htm",
        "is_subsite": True,
        "site_label": "外国语学院",
    },
    {
        # 2026-10 修正：/index/xyxw.htm 同样是旧模板页（最新 2023）；
        # 学院动态现挂在 xydt.htm
        "name": "外国语学院新闻",
        "icon": "🌐",
        "base_url": "https://wyx.hait.edu.cn",
        "list_path": "/index/xydt.htm",
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
        # 2026-10 修正：/index/xyxw.htm 是 VSB 动态模板，静态 HTML 里没有文章链接，
        # 爬虫会判定「第 1 页无文章」。首页的资讯区块是静态的，故改从首页取，
        # 并用 url_contains 只保留 channel 1147（学院新闻），避免混入其他栏目。
        # 首页没有分页，只能覆盖最新几篇；日常增量足够。
        "name": "材料科学与工程学院新闻",
        "icon": "🔬",
        "base_url": "https://clxnew.hait.edu.cn",
        "list_path": "/index.htm",
        "url_contains": "info/1147/",
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
