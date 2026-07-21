import json 
from src.config import ConfigManager
import redis.asyncio as redis


async def get_redis_client():
    config = ConfigManager()
    return await redis.from_url(
        config.redis_url,
        decode_responses = True,
        encoding = "utf-8",
        protocol=2
    )


async def save_message(redis_client, session_id: str, message_dict: dict):
    key = f"chat:history:{session_id}"

    json_str = json.dumps(message_dict, ensure_ascii=False)
    await redis_client.lpush(key, json_str)
    await redis_client.expire(key, 1800)


async def get_recent_messages(redis_client,session_id,count = 20):
    key = f"chat:history:{session_id}"
    messages = await redis_client.lrange(key, 0, count - 1)
    parsed_messages = []
    for msg in messages:
        msg_dict = json.loads(msg)
        parsed_messages.append(msg_dict)
    parsed_messages.reverse()
    return parsed_messages

