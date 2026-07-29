"""FastAPI 可复用依赖。"""
import logging
from collections.abc import Generator

from fastapi import Header, HTTPException, Request
from sqlalchemy.orm import Session

from database import SessionLocal
from src.config import ConfigManager

logger = logging.getLogger(__name__)

# 模块级单例，避免每个请求都重新读取配置
_config = ConfigManager()


def verify_api_key(request: Request, x_api_key: str | None = Header(None)):
    client_ip = request.client.host if request.client else "unknown"
    if not x_api_key:
        logger.warning("未提供 API Key，客户端 IP: %s", client_ip)
        raise HTTPException(status_code=401, detail="Unauthorized")
    elif x_api_key != _config.api_key:
        logger.warning("API Key 匹配失败，客户端 IP: %s", client_ip)
        raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        logger.info("apikey验证成功")
        return x_api_key


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：为每个请求提供独立的数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
