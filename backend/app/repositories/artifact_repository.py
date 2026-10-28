"""生成文件元数据仓储，所有读取都限定当前用户。"""

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.generated_artifact import GeneratedArtifact


class ArtifactRepository:
    """提供按归属读取和按幂等键查询的关系库操作。"""

    async def get_owned(self, session: AsyncSession, artifact_id: str, user_id: int) -> GeneratedArtifact | None:
        """只返回当前用户的记录，避免泄露其他用户的生成结果。"""
        return (await session.exec(select(GeneratedArtifact).where(
            GeneratedArtifact.id == artifact_id, GeneratedArtifact.user_id == user_id))).first()

    async def get_by_request(self, session: AsyncSession, key: str, user_id: int) -> GeneratedArtifact | None:
        """查询同一请求的已有结果，避免重复调用工具时再次生成同一个文件。"""
        return (await session.exec(select(GeneratedArtifact).where(
            GeneratedArtifact.request_key == key, GeneratedArtifact.user_id == user_id))).first()

    async def create(self, session: AsyncSession, artifact: GeneratedArtifact) -> GeneratedArtifact:
        """提交文件元数据；调用者负责文件写入与失败补偿。"""
        session.add(artifact)
        await session.commit()
        await session.refresh(artifact)
        return artifact
