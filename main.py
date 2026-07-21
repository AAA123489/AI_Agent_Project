"""FastAPI 入口 —— 组装应用、中间件、路由。"""
import logging
from contextlib import asynccontextmanager
import uvicorn
import fastapi_cdn_host
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis_client import get_redis_client
from database import Base, engine
import src.logger  # noqa: F401  ← 导入即初始化日志配置
from app.routers.chat import router as chat_router

logger = logging.getLogger(__name__)


# ── 生命周期 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时自动创建数据库表。"""
    Base.metadata.create_all(bind=engine)
    redis_client = await get_redis_client() 
    app.state.redis = redis_client
    yield
    await app.state.redis.close()


# ── 应用实例 ──────────────────────────────────────────────
app = FastAPI(title="AI-Chat-Api", lifespan=lifespan)

# ── 中间件 ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Swagger CDN ──────────────────────────────────────────
fastapi_cdn_host.patch_docs(app)

# ── 路由注册 ──────────────────────────────────────────────
app.include_router(chat_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
