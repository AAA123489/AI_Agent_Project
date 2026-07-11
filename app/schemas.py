#此文件专门用来放 Pydantic 模型
from pydantic import BaseModel,Field

class ChatRequest(BaseModel):
    message : str = Field(
        min_length = 2,
        max_length = 1000,
        description = "用户消息，2-1000字符"
    )
    user_id : str = Field(
        min_length = 2,
        max_length = 50,
        description = "用户id，2-50字符"
    )

    