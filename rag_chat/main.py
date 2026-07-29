"""FastAPI 入口 —— 组装应用、中间件、路由。"""
import logging
from contextlib import asynccontextmanager

import fastapi_cdn_host
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import src.logger  # noqa: F401  ← 导入即初始化日志配置
from app.routers.chat import router as chat_router
from database import Base, engine
from redis_client import get_redis_client

logger = logging.getLogger(__name__)


# ── 生命周期 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时自动创建数据库表、预热向量库模型。"""
    Base.metadata.create_all(bind=engine)
    redis_client = await get_redis_client()
    app.state.redis = redis_client

    # 预热 Embedding 模型（避免首次请求时加载 420MB 模型造成超长等待）
    import asyncio as _asyncio
    from app.routers.chat import get_vector_db
    await _asyncio.to_thread(get_vector_db)
    logger.info("向量库模型预热完成")

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

# ── 静态文件 ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
