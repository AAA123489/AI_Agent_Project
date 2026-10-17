"""
auth.py — 注册 / 登录 / 会话
==========================
- 用户存 SQLite（db.py），密码 bcrypt 哈希
- 登录态是 HMAC-SHA256 签名的 HttpOnly Cookie，不依赖 JWT 库
- 本模块不 import app_backend，测试可以脱离 Chroma / torch
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from db import connect_db, get_daily_usage

load_dotenv()

logger = logging.getLogger("auth")

_PROJECT_ROOT = Path(__file__).resolve().parent
_WEB_DIR = _PROJECT_ROOT / "web"

USERNAME_RE = re.compile(r"^[A-Za-z]{2,10}$")
SESSION_COOKIE = "rag_session"
MAX_PASSWORD_CHARS = 64
DEFAULT_SESSION_TTL_SECONDS = 7 * 86400

_EPHEMERAL_SECRET: str | None = None
_DUMMY_HASH: str | None = None


class UsernameTaken(Exception):
    """用户名（大小写不敏感）已存在。"""


# ── 环境变量（懒读取，方便测试 monkeypatch） ──

def min_password_len() -> int:
    try:
        return max(1, int(os.getenv("MIN_PASSWORD_LEN", "8")))
    except ValueError:
        return 8


def session_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS))))
    except ValueError:
        return DEFAULT_SESSION_TTL_SECONDS


def bcrypt_rounds() -> int:
    try:
        return min(31, max(4, int(os.getenv("BCRYPT_ROUNDS", "12"))))
    except ValueError:
        return 12


def daily_quota_limit() -> int:
    try:
        return max(0, int(os.getenv("DAILY_MESSAGE_QUOTA", "50")))
    except ValueError:
        return 50


def invite_required() -> bool:
    return bool((os.getenv("REGISTER_INVITE_CODE") or "").strip())


def verify_invite_code(provided: Optional[str]) -> bool:
    expected = (os.getenv("REGISTER_INVITE_CODE") or "").strip()
    if not expected:
        return True
    return hmac.compare_digest((provided or "").strip(), expected)


# ── 校验与密码 ──

def normalize_username(raw: str) -> str:
    return (raw or "").strip().lower()


def validate_username(raw: str) -> str:
    username = (raw or "").strip()
    if not USERNAME_RE.fullmatch(username):
        raise ValueError("用户名必须为 2-10 个英文字母")
    return username


def validate_password(raw: str) -> str:
    password = raw or ""
    if len(password) < min_password_len():
        raise ValueError(f"密码至少 {min_password_len()} 个字符")
    if len(password) > MAX_PASSWORD_CHARS:
        raise ValueError(f"密码最多 {MAX_PASSWORD_CHARS} 个字符")
    return password


def _prehash(password: str) -> bytes:
    """先 SHA-256 再 base64，规避 bcrypt 72 字节上限，支持任意 UTF-8 密码。"""
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt(bcrypt_rounds())).decode("ascii")


def verify_password(password: str, stored: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), stored.encode("ascii"))
    except (ValueError, TypeError):
        return False


def _dummy_hash() -> str:
    """未知用户也跑一次 bcrypt，避免用响应时间枚举账号。"""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("invalid-password")
    return _DUMMY_HASH


# ── 用户读写 ──

def _public_user(row) -> dict:
    return {"id": int(row["id"]), "username": row["username"]}


def get_user_by_id(user_id: int) -> dict | None:
    conn = connect_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(username: str, password: str) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    conn = connect_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO users(username, username_norm, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, normalize_username(username), hash_password(password), now),
        )
        conn.commit()
        user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise UsernameTaken() from exc
    finally:
        conn.close()
    return get_user_by_id(user_id)


def authenticate_user(username: str, password: str) -> dict | None:
    conn = connect_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username_norm = ?",
            (normalize_username(username),),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        verify_password(password, _dummy_hash())
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    if row["disabled"]:
        return None
    return dict(row)


def update_last_login(user_id: int) -> None:
    conn = connect_db()
    try:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), user_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── 会话 token ──

def _secret_key() -> bytes:
    global _EPHEMERAL_SECRET
    value = (os.getenv("SECRET_KEY") or "").strip()
    if value:
        return value.encode("utf-8")

    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_urlsafe(48)
        logger.warning(
            "SECRET_KEY 未配置：使用进程内随机密钥，服务重启后所有登录都会失效。"
            "部署前请在 .env 中设置固定值。"
        )
    return _EPHEMERAL_SECRET.encode("utf-8")


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def make_session_token(user: dict) -> str:
    now = int(time.time())
    payload = {
        "uid": int(user["id"]),
        "un": user["username"],
        "sv": int(user["session_version"]),
        "iat": now,
        "exp": now + session_ttl_seconds(),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = _b64e(raw)
    sig = hmac.new(_secret_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64e(sig)}"


def decode_session_token(token: str) -> dict | None:
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(_secret_key(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64d(sig), expected):
            return None
        payload = json.loads(_b64d(body).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def load_user(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    payload = decode_session_token(token)
    if not payload:
        return None

    user = get_user_by_id(int(payload.get("uid", 0)))
    if not user or user["disabled"]:
        return None
    if int(payload.get("sv", 0)) != int(user["session_version"]):
        return None
    return _public_user(user)


def optional_user(request: Request) -> dict | None:
    return load_user(request)


def require_user(request: Request) -> dict:
    user = load_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


# ── Cookie ──

def cookie_secure(request: Request) -> bool:
    mode = (os.getenv("COOKIE_SECURE", "auto") or "auto").strip().lower()
    if mode == "true":
        return True
    if mode == "false":
        return False

    if request.url.scheme == "https":
        return True
    forwarded = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return forwarded == "https"


def set_session_cookie(response, token: str, request: Request) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=session_ttl_seconds(),
        path="/",
        httponly=True,
        samesite="lax",
        secure=cookie_secure(request),
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


# ── 请求模型 ──

class RegisterRequest(BaseModel):
    username: str = ""
    password: str = ""
    invite_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


# ── 路由 ──

auth_router = APIRouter()


@auth_router.get("/login")
def login_page(request: Request):
    if optional_user(request):
        return RedirectResponse(url="/", status_code=303)

    page = _WEB_DIR / "login.html"
    if page.exists():
        return FileResponse(page, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h1>login.html not found</h1>", status_code=404)


@auth_router.get("/api/auth-config")
def auth_config():
    return {
        "invite_required": invite_required(),
        "min_password_len": min_password_len(),
        "max_password_len": MAX_PASSWORD_CHARS,
        "username_pattern": USERNAME_RE.pattern,
    }


@auth_router.post("/api/register")
def register(req: RegisterRequest, request: Request):
    try:
        username = validate_username(req.username)
        validate_password(req.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not verify_invite_code(req.invite_code):
        raise HTTPException(status_code=403, detail="邀请码无效")

    try:
        user = create_user(username, req.password)
    except UsernameTaken as exc:
        raise HTTPException(status_code=409, detail="该用户名已被注册，请换一个") from exc

    response = JSONResponse({"user": _public_user(user)}, status_code=201)
    set_session_cookie(response, make_session_token(user), request)
    return response


@auth_router.post("/api/login")
def login(req: LoginRequest, request: Request):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    update_last_login(int(user["id"]))
    response = JSONResponse({"user": _public_user(user)}, status_code=200)
    set_session_cookie(response, make_session_token(user), request)
    return response


@auth_router.post("/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    clear_session_cookie(response)
    return response


@auth_router.get("/api/me")
def me(user: dict = Depends(require_user)):
    limit = daily_quota_limit()
    used = get_daily_usage(int(user["id"]))
    return {
        "user": user,
        "quota": {
            "limit": limit,
            "used": used,
            "remaining": max(0, limit - used) if limit > 0 else None,
        },
    }
