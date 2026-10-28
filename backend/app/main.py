"""应用装配入口：初始化数据库、模型、checkpoint、Store 和聊天图。"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
    """启动时创建共享资源，yield 后等待服务关闭，再释放连接池等资源。"""
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
            # 选择其他模型时复用同一个持久化组件，原 thread_id 和消息保持不变。
            app.state.checkpointer=checkpointer
            app.state.chat_graph=build_chat_graph(model=model, checkpointer=checkpointer,store=store)

            yield


def create_app() -> FastAPI:
    """创建应用并装配中间件、业务路由和健康检查。"""
    app = FastAPI( #创建一个FastAPI实例
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )
    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        """校验错误只返回字段和原因，避免错误响应回显用户提交的 API Key。"""
        def describe(error):
            """常见输入限制使用中文；自定义验证器的中文原因保持原样。"""
            context = error.get("ctx", {})
            messages = {
                "missing": "缺少必填字段", "string_type": "请输入文本",
                "int_parsing": "请输入整数", "int_type": "请输入整数",
                "float_parsing": "请输入有效数字", "literal_error": "该选项不受支持",
                "extra_forbidden": "包含不支持的参数",
                "greater_than_equal": f"数值不能小于 {context.get('ge')}",
                "less_than_equal": f"数值不能大于 {context.get('le')}",
                "greater_than": f"数值必须大于 {context.get('gt')}",
                "string_too_short": f"文本至少需要 {context.get('min_length')} 个字符",
                "string_too_long": f"文本不能超过 {context.get('max_length')} 个字符",
            }
            return messages.get(error["type"], error["msg"].removeprefix("Value error, "))
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(error["loc"]), "msg": describe(error), "type": error["type"]}
            for error in exc.errors()
        ]})
#给前端react开的权限
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
        """确认 HTTP 服务可响应，不代表数据库或模型服务正常。"""
        return {
            "status": "ok",
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
        }

    @app.get("/health/db", tags=["health"])
    async def database_check():
        """通过 SELECT 1 检查关系数据库连接。"""
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
        """检查向量库连接；失败返回中文 HTTP 503。"""
        try:
            await MilvusBookStore().ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Milvus 暂不可用") from exc
        return {"status": "ok", "milvus": "connected"}

    return app


app = create_app()
