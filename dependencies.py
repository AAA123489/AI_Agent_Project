import logging
from fastapi import Request, Header, HTTPException
from src.config import ConfigManager

logger = logging.getLogger(__name__)


def verify_api_key(x_api_key: str = Header(None), request: Request = None):
    config = ConfigManager()
    if not x_api_key:
        logger.warning("未提供 API Key，客户端 IP: %s", request.client.host)
        raise HTTPException(status_code=401, detail="Unauthorized")
    elif x_api_key != config.api_key:
        logger.warning("API Key 匹配失败，客户端 IP: %s", request.client.host)
        raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        logger.info("apikey验证成功")
        return x_api_key