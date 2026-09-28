from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import create_access_token
from app.infrastructure.database.database import get_session
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.user import UserCreate
from app.services.auth_service import AuthService
from app.services.user_service import UserService

router=APIRouter(prefix="/auth",tags=["auth"])

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(data: UserCreate, session: AsyncSession = Depends(get_session)):
    try:
        user = await UserService().create_user(session, data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    assert user.id is not None
    return TokenResponse(access_token=create_access_token(user.id))

@router.post("/login",response_model=TokenResponse)
async def login(data:LoginRequest,session:AsyncSession=Depends(get_session)):
    try:
        user=await AuthService().authenticate(session,data.username,data.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail=str(exc)) from exc
    assert user.id is not None
    return TokenResponse(access_token=create_access_token(user.id))
