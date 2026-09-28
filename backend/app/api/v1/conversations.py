from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.infrastructure.database.database import get_session
from app.schemas.conversation import ConversationCreate, ConversationRead
from app.services.conversation_service import ConversationService
from app.models.user import User
from app.api.v1.deps import get_current_user


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(data: ConversationCreate, session: AsyncSession = Depends(get_session), current_user: User = Depends(get_current_user)):
    assert current_user.id is not None
    return await ConversationService().create(session,current_user.id, data)


@router.get("", response_model=list[ConversationRead])
async def list_conversations(session: AsyncSession = Depends(get_session),current_user:User=Depends(get_current_user)):
    assert current_user.id is not None
    return await ConversationService().list_by_user(session, current_user.id)


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(conversation_id: int, session: AsyncSession = Depends(get_session), current_user: User = Depends(get_current_user)):
    try:
        assert current_user.id is not None
        return await ConversationService().get_owned(session, conversation_id, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
