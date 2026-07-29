import json

import redis.asyncio as aioredis

from src.config import ConfigManager

_config = ConfigManager()


async def get_redis_client():
    return await aioredis.from_url(
        _config.redis_url,
        decode_responses=True,
        encoding="utf-8",
        protocol=2,
        socket_connect_timeout=5,
        socket_keepalive=True,
        retry_on_timeout=True,
    )


async def save_message(redis_client: aioredis.Redis, session_id: str, message_dict: dict) -> None:
    key = f"chat:history:{session_id}"
    json_str = json.dumps(message_dict, ensure_ascii=False)
    await redis_client.lpush(key, json_str)
    await redis_client.expire(key, 1800)


async def get_recent_messages(redis_client: aioredis.Redis, session_id: str, count: int = 20) -> list[dict]:
    key = f"chat:history:{session_id}"
    messages = await redis_client.lrange(key, 0, count - 1)
    parsed_messages: list[dict] = []
    for msg in messages:
        msg_dict = json.loads(msg)
        parsed_messages.append(msg_dict)
    parsed_messages.reverse()
    return parsed_messages

