"""组合各业务路由；统一的 /api/v1 前缀由应用入口添加。"""

from fastapi import APIRouter

from app.api.v1.users import router as users_router
from app.api.v1.chat import router as chat_router
from app.api.v1.memories import router as memories_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.auth import router as auth_router
from app.api.v1.knowledge import router as knowledge_router
from app.api.v1.models import router as models_router
from app.api.v1.files import router as files_router


api_router = APIRouter()

api_router.include_router(users_router)
api_router.include_router(chat_router)
api_router.include_router(memories_router)
api_router.include_router(conversations_router)
api_router.include_router(auth_router)
api_router.include_router(knowledge_router)
api_router.include_router(models_router)
api_router.include_router(files_router)
