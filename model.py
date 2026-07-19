from database import Base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

class ChatHistory(Base):
    __tablename__ = "chat_history"
    # 1. 主键 ID：整数类型，自增，并建立索引
    id = Column(Integer, primary_key=True, index=True)
    
    # 2. 用户 ID：字符串类型，建立索引（方便后续按用户查询历史记录）
    user_id = Column(String, index=True)
    
    # 3. 用户提问：字符串类型
    message = Column(String)
    
    # 4. AI 回复：字符串类型
    reply = Column(String)
    
    # 5. 时间戳：日期时间类型，默认使用数据库的当前时间
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
