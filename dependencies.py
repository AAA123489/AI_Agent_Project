'''from fastapi import Request,Header,HTTPException
from src.config import ConfigManager


def verify_api_key(x_api_key: str = Header(None)):

    config = ConfigManager()
    print(f"[DEBUG] 收到的Key: {x_api_key!r}, 期望的Key: {config.api_key!r}")
    if not x_api_key or x_api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        return x_api_key'''

from fastapi import Request, Header, HTTPException
from src.config import ConfigManager

def verify_api_key(x_api_key: str = Header(None), request: Request = None):
    config = ConfigManager()
    if not x_api_key or x_api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        return x_api_key