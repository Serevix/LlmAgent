"""生成文件下载与图片预览接口；每次读取都验证当前用户归属。"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.deps import get_current_user
from app.artifacts.errors import ArtifactError
from app.infrastructure.database.database import get_session
from app.models.user import User
from app.services.artifact_service import ArtifactService


router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{artifact_id}/content")
async def file_content(artifact_id: str, session: AsyncSession = Depends(get_session),
                       current_user: User = Depends(get_current_user)):
    """返回所属文件的二进制内容，前端通过 Bearer 请求后生成预览或下载链接。"""
    assert current_user.id is not None
    service = ArtifactService()
    artifact = await service.repository.get_owned(session, artifact_id, current_user.id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        path = service.storage_path(artifact)
    except ArtifactError:
        raise HTTPException(status_code=404, detail="文件不存在") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件已不存在，请重新生成。")
    return FileResponse(path, media_type=artifact.media_type, filename=artifact.filename,
                        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
