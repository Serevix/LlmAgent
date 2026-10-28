"""书籍接口：上传、查询状态、重试和删除；耗时导入交给后台任务。"""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.deps import get_current_user
from app.infrastructure.database.database import get_session
from app.models.user import User
from app.repositories.knowledge_book_repository import KnowledgeBookRepository
from app.schemas.knowledge import BookRead
from app.services.knowledge_service import KnowledgeService


router = APIRouter(prefix="/knowledge/books", tags=["knowledge"])


@router.get("", response_model=list[BookRead])
async def list_books(session: AsyncSession = Depends(get_session),
                     user: User = Depends(get_current_user)):
    """仅列出登录用户的书籍和导入状态。"""
    assert user.id is not None
    return await KnowledgeBookRepository().list_by_user(session, user.id)


@router.post("", response_model=BookRead, status_code=202)
async def upload_book(background: BackgroundTasks, file: UploadFile = File(...),
                      title: str | None = Form(None),
                      session: AsyncSession = Depends(get_session),
                      user: User = Depends(get_current_user)):
    """校验并保存文件，安排后台解析任务；202 只表示已接收。"""
    assert user.id is not None
    user_id = user.id
    service = KnowledgeService()
    try:
        book = await service.upload(session, user_id, file, title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background.add_task(service.process, book.id, user_id)
    return book


@router.post("/{book_id}/retry", response_model=BookRead, status_code=202)
async def retry_book(book_id: str, background: BackgroundTasks,
                     session: AsyncSession = Depends(get_session),
                     user: User = Depends(get_current_user)):
    """确认书籍归属后重置状态并安排再次导入。"""
    assert user.id is not None
    user_id = user.id
    repository = KnowledgeBookRepository()
    book = await repository.get_owned(session, book_id, user_id)
    if book is None:
        raise HTTPException(status_code=404, detail="书籍不存在")
    service = KnowledgeService()
    try:
        book = await service.retry(session, book)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background.add_task(service.process, book.id, user_id)
    return book


@router.delete("/{book_id}", status_code=204)
async def delete_book(book_id: str, session: AsyncSession = Depends(get_session),
                      user: User = Depends(get_current_user)):
    """确认归属后删除向量、关系记录和原文件，处理中书籍不可删除。"""
    assert user.id is not None
    repository = KnowledgeBookRepository()
    book = await repository.get_owned(session, book_id, user.id)
    if book is None:
        raise HTTPException(status_code=404, detail="书籍不存在")
    try:
        await KnowledgeService().delete(session, book)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
