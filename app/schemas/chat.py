"""Pydantic 请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=2,
        max_length=1000,
        description="用户消息，2-1000 字符",
    )
    user_id: str = Field(
        min_length=2,
        max_length=50,
        description="用户 ID，2-50 字符",
    )

    
class ChatHistoryResponse(BaseModel):
    id: int
    user_id: str
    message: str
    reply: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)