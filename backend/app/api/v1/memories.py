from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.memory import MemoryCreate,MemoryRead
from app.api.v1.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/memories", tags=["memories"])

@router.post("", response_model=MemoryRead)
async def create_memory(data: MemoryCreate, request: Request, current_user: User = Depends(get_current_user)):
    if data.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="不能访问其他用户的记忆")
    key = uuid4().hex
    namespace = ("users", str(data.user_id), "memories")

    await request.app.state.store.aput(namespace, key, {"content": data.content})

    return MemoryRead(key=key, content=data.content)


@router.get("/{user_id}", response_model=list[MemoryRead])
async def list_memories(user_id: int, request: Request, current_user: User = Depends(get_current_user)):
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="不能访问其他用户的记忆")
    namespace = ("users", str(user_id), "memories")
    items = await request.app.state.store.asearch(namespace, limit=100)

    return [MemoryRead(key=item.key, content=item.value["content"]) for item in items]
