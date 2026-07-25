# 导入 SQLAlchemy 的核心组件，用于创建数据库引擎和会话
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 数据库连接地址，使用本地 SQLite 文件保存数据
SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_app.db"

# 创建数据库引擎，并关闭线程检查以适配 SQLite 的使用场景
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# 创建会话工厂，负责生成数据库会话对象
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类，后续所有数据库模型都需要继承它
Base = declarative_base()