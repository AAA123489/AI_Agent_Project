"""公共 fixture：隔离的 SQLite、快速 bcrypt、可用的会话密钥。

所有测试都不碰真实 data/rag_app.db —— 每个用例用自己的临时目录。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    """指向临时库，并把 bcrypt 调到最低轮数（测试快且不影响正确性）。"""
    db_path = tmp_path / "test_app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-for-production")
    monkeypatch.setenv("BCRYPT_ROUNDS", "4")
    monkeypatch.setenv("MIN_PASSWORD_LEN", "8")
    monkeypatch.setenv("DAILY_MESSAGE_QUOTA", "3")
    monkeypatch.setenv("REGISTER_INVITE_CODE", "")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")

    from db import init_db

    init_db()
    return db_path


@pytest.fixture()
def client(app_env):
    """只挂 auth_router 的最小应用，避免 import app_backend 拉起 torch/Chroma。"""
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from auth import auth_router, require_user

    app = FastAPI()
    app.include_router(auth_router)

    @app.get("/api/protected")
    def protected(user: dict = Depends(require_user)):
        return {"user": user}

    @app.get("/")
    def home():
        return {"ok": True}

    with TestClient(app) as test_client:
        yield test_client
