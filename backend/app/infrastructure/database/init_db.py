"""初始化关系表；模型导入用于将表注册到 SQLModel 元数据。"""

import asyncio

from sqlalchemy import text
from sqlmodel import SQLModel

from app.infrastructure.database.database import engine
from app.models.user import User
from app.models.conversation import Conversation
from app.models.knowledge_book import KnowledgeBook
from app.models.model_profile import ModelProfile
from app.models.generated_artifact import GeneratedArtifact


async def init_db() -> None:
    """在连接事务中创建已注册的关系表；不会创建 Milvus 集合。"""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        # create_all does not update columns in an existing table. Earlier
        # databases made avatar NOT NULL although registration treats it as optional.
        if conn.dialect.name == "postgresql":
            await conn.execute(text("ALTER TABLE users ALTER COLUMN avatar DROP NOT NULL"))


if __name__ == "__main__":
    asyncio.run(init_db())
