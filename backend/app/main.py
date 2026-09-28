from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.api.v1.router import api_router
from app.core.config import settings
from app.infrastructure.database.database import engine
from app.infrastructure.database.init_db import init_db
from app.agents.chat_graph import build_chat_graph
from app.llm.factory import create_chat_model
from app.services.knowledge_service import mark_interrupted_imports
from app.rag.milvus_store import MilvusBookStore


@asynccontextmanager #Python 里用来把一个异步生成器函数变成“异步上下文管理器”的装饰器。
async def lifespan(app:FastAPI):
    await init_db()
    await mark_interrupted_imports()
    async with AsyncPostgresSaver.from_conn_string(settings.PSYCOPG_DATABASE_URL) as checkpointer:
        async with AsyncPostgresStore.from_conn_string(settings.PSYCOPG_DATABASE_URL) as store:
            await checkpointer.setup()
            await store.setup()


            model =create_chat_model(
                platform="siliconflow",
                model=settings.CHAT_MODEL
            )
            app.state.store=store
            app.state.chat_graph=build_chat_graph(model=model, checkpointer=checkpointer,store=store)

            yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in settings.FRONTEND_ORIGINS.split(",") if origin.strip()],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(
        api_router,
        prefix=settings.API_V1_PREFIX,
    )

    @app.get("/health", tags=["health"])
    async def health_check():
        return {
            "status": "ok",
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
        }

    @app.get("/health/db", tags=["health"])
    async def database_check():
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT 1")
            )

            value = result.scalar_one()

        return {
            "status": "ok",
            "database": value,
        }

    @app.get("/health/milvus", tags=["health"])
    async def milvus_check():
        try:
            await MilvusBookStore().ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Milvus 暂不可用") from exc
        return {"status": "ok", "milvus": "connected"}

    return app


app = create_app()
