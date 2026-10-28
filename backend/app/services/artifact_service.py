"""生成并保存用户的图片与 Word，元数据入库，原文件保存在受控目录。"""

import asyncio
import hashlib
from pathlib import Path
import re

from sqlmodel.ext.asyncio.session import AsyncSession

from app.artifacts.errors import ArtifactError
from app.artifacts.images import generate_image_bytes
from app.artifacts.word import word_bytes
from app.core.config import settings
from app.infrastructure.database.database import engine
from app.models.generated_artifact import GeneratedArtifact
from app.repositories.artifact_repository import ArtifactRepository
from app.schemas.artifact import ArtifactRead


class ArtifactService:
    """业务入口只接收认证身份和结构化参数，不接受模型提供的磁盘路径。"""

    def __init__(self):
        """装配关系仓储，延迟到实际调用时才访问文件和数据库。"""
        self.repository = ArtifactRepository()

    @staticmethod
    def public(artifact: GeneratedArtifact) -> dict:
        """返回可持久化到消息中的公开附件信息，排除存储路径和用户凭据。"""
        return ArtifactRead(id=artifact.id, kind=artifact.kind, title=artifact.title,
                            filename=artifact.filename, media_type=artifact.media_type,
                            size_bytes=artifact.size_bytes,
                            download_url=f"/api/v1/files/{artifact.id}/content").model_dump()

    @staticmethod
    def storage_path(artifact: GeneratedArtifact) -> Path:
        """解析并校验存储路径，拒绝越过配置的生成文件根目录。"""
        root = settings.ARTIFACT_STORAGE_DIR.resolve()
        path = Path(artifact.storage_path).resolve()
        if not path.is_relative_to(root):
            raise ArtifactError("文件存储位置无效。")
        return path

    async def create(self, *, user_id: int, thread_id: str, request_key: str,
                     kind: str, title: str, prompt: str = "", aspect: str = "square",
                     content: str = "", sources: list[dict] | None = None) -> dict:
        """复用同一请求的结果，生成成功后保存文件；数据库提交失败时清理本次文件。"""
        key = hashlib.sha256(f"{user_id}:{thread_id}:{request_key}:{kind}".encode()).hexdigest()
        async with AsyncSession(engine) as session:
            existing = await self.repository.get_by_request(session, key, user_id)
            if existing:
                if not self.storage_path(existing).is_file():
                    raise ArtifactError("此前生成的文件已不存在，请重新提出生成请求。")
                return self.public(existing)
            if kind == "image":
                data, extension, media_type = await generate_image_bytes(prompt, aspect)
            elif kind == "word":
                data = await asyncio.to_thread(word_bytes, title, content, sources or [])
                extension = "docx"
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                raise ArtifactError("不支持的文件类型。")
            artifact_id = key[:32]
            root = settings.ARTIFACT_STORAGE_DIR.resolve()
            folder = root / str(user_id)
            await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
            path = folder / f"{artifact_id}.{extension}"
            # 展示文件名也移除 Windows 非法字符；实际磁盘路径始终使用服务器生成的 ID。
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")[:100] or "生成文件"
            record = GeneratedArtifact(id=artifact_id, user_id=user_id, thread_id=thread_id,
                request_key=key, kind=kind, title=title, filename=f"{name}.{extension}",
                storage_path=str(path), media_type=media_type, size_bytes=len(data))
            self.storage_path(record)
            try:
                await asyncio.to_thread(path.write_bytes, data)
                record = await self.repository.create(session, record)
            except BaseException:
                await asyncio.to_thread(path.unlink, missing_ok=True)
                raise
            return self.public(record)
