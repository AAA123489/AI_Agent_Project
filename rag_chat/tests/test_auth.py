"""账号体系：注册、登录、会话、额度。

不 import app_fastapi（那会拉起 torch/Chroma），只挂 auth_router。
"""

import time

import pytest


def register(client, username="alice", password="password123", invite_code=None):
    return client.post(
        "/api/register",
        json={"username": username, "password": password, "invite_code": invite_code},
    )


def login(client, username="alice", password="password123"):
    return client.post("/api/login", json={"username": username, "password": password})


# ── 注册 ──


def test_注册成功并自动登录(client):
    res = register(client)
    assert res.status_code == 201
    assert res.json()["user"]["username"] == "alice"

    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "alice"


def test_用户名不区分大小写不可重复(client):
    assert register(client, "Alice").status_code == 201
    res = register(client, "alice")
    assert res.status_code == 409
    assert "已被注册" in res.json()["detail"]


@pytest.mark.parametrize(
    "username",
    ["a", "a" * 11, "alice1", "张伟", "ali ce", "alice!", ""],
)
def test_非法用户名被拒绝(client, username):
    res = register(client, username)
    assert res.status_code == 422
    assert "2-10" in res.json()["detail"]


def test_短密码被拒绝(client):
    res = register(client, password="1234567")
    assert res.status_code == 422
    assert "至少 8" in res.json()["detail"]


def test_邀请码开启时校验(client, monkeypatch):
    monkeypatch.setenv("REGISTER_INVITE_CODE", "let-me-in")

    assert register(client, "bob", invite_code=None).status_code == 403
    assert register(client, "bob", invite_code="wrong").status_code == 403
    assert register(client, "bob", invite_code="let-me-in").status_code == 201


# ── 登录 ──


def test_登录成功(client):
    register(client)
    client.post("/api/logout")

    res = login(client)
    assert res.status_code == 200
    assert client.get("/api/me").status_code == 200


def test_登录失败文案一致(client):
    register(client)
    client.post("/api/logout")

    wrong_password = login(client, password="wrong-password")
    unknown_user = login(client, username="nobody")

    assert wrong_password.status_code == 401
    assert unknown_user.status_code == 401
    assert wrong_password.json()["detail"] == unknown_user.json()["detail"]


def test_未登录访问受保护接口(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/protected").status_code == 401


def test_登出后会话失效(client):
    register(client)
    assert client.get("/api/me").status_code == 200

    assert client.post("/api/logout").status_code == 200
    assert client.get("/api/me").status_code == 401


def test_篡改签名的会话被拒绝(client):
    register(client)
    token = client.cookies.get("rag_session")
    assert token

    body, _, _sig = token.partition(".")
    client.cookies.set("rag_session", f"{body}.forged-signature")
    assert client.get("/api/me").status_code == 401


def test_过期会话被拒绝(client, monkeypatch):
    register(client)
    token = client.cookies.get("rag_session")

    import auth

    real_time = time.time
    monkeypatch.setattr(auth.time, "time", lambda: real_time() + 10**7)

    client.cookies.set("rag_session", token)
    assert client.get("/api/me").status_code == 401


def test_会话版本变化后旧cookies失效(client, app_env):
    register(client)
    token = client.cookies.get("rag_session")

    from auth import get_user_by_id
    from db import connect_db

    user = get_user_by_id(1)
    conn = connect_db()
    try:
        conn.execute(
            "UPDATE users SET session_version = session_version + 1 WHERE id = ?",
            (user["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    client.cookies.set("rag_session", token)
    assert client.get("/api/me").status_code == 401


# ── 每日额度 ──


def test_每日额度原子扣减(app_env):
    from auth import create_user
    from db import get_daily_usage, try_consume_daily_quota

    alice = create_user("alice", "password123")
    bob = create_user("bob", "password123")

    assert [try_consume_daily_quota(alice["id"], limit=2) for _ in range(3)] == [
        True,
        True,
        False,
    ]
    assert get_daily_usage(alice["id"]) == 2

    # 不同用户互不影响
    assert try_consume_daily_quota(bob["id"], limit=2) is True


def test_额度不限时永远放行(app_env):
    from auth import create_user
    from db import try_consume_daily_quota

    alice = create_user("alice", "password123")
    for _ in range(10):
        assert try_consume_daily_quota(alice["id"], limit=0) is True
