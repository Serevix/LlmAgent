"""登录请求与 token 响应的数据格式。"""

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录输入；username 字段也接受邮箱。"""
    username: str = Field(min_length=2, max_length=254)
    password: str = Field(min_length=8, max_length=128)

class TokenResponse(BaseModel):
    """认证成功返回的访问 token，前端用于后续 Bearer 请求。"""
    access_token: str
    token_type: str = "bearer"
