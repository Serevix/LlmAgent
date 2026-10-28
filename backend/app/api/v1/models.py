"""模型选择与用户参数管理接口；参数配置不影响原有会话历史接口。"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.v1.deps import get_current_user
from app.infrastructure.database.database import get_session
from app.models.user import User
from app.schemas.model_profile import ModelProfileCreate, ModelProfileRead
from app.services.model_service import ModelService, ModelSpec
from app.llm.structured import output_instructions


router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger(__name__)


class ConnectionDecision(BaseModel):
    """测试服务能否返回工作流需要的结构化数据。"""
    connected: bool = Field(description="收到连接测试时返回 true。")


@router.get("", response_model=list[ModelProfileRead])
async def list_models(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    """只列出用户自己的模型配置和服务端允许的预设。"""
    return await ModelService().list_models(session, user.id)


@router.post("", response_model=ModelProfileRead, status_code=201)
async def add_model(data: ModelProfileCreate, session: AsyncSession = Depends(get_session),
                    user: User = Depends(get_current_user)):
    """保存手填或导入的参数，不在响应中回传密钥。"""
    try:
        return await ModelService().create(session, user.id, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str, session: AsyncSession = Depends(get_session),
                       user: User = Depends(get_current_user)):
    """删除自己的配置，不修改消息、会话或书籍知识库。"""
    repository = ModelService().repository
    profile = await repository.get_owned(session, model_id, user.id)
    if profile is None:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    await repository.delete(session, profile)


@router.post("/test")
async def test_model(data: ModelProfileCreate, user: User = Depends(get_current_user)):
    """用户明确点击测试时，检查流式文字和结构化输出，两次调用不产生会话历史。"""
    spec = ModelSpec(data.model, data.base_url, data.api_key.get_secret_value().strip(), data.context_window,
                     data.max_output_tokens, data.temperature, data.top_p, data.timeout,
                     data.structured_output_method)
    try:
        async with ModelService().runtime_model(spec) as model:
            response = await model.with_structured_output(ConnectionDecision, method=spec.structured_output_method).ainvoke(
                "这是连接测试，请仅返回 connected=true。" + output_instructions(ConnectionDecision, spec.structured_output_method),
                max_tokens=128,
            )
            if not isinstance(response, ConnectionDecision) or not response.connected:
                raise ValueError("结构化输出无效")
            text = ""
            async for chunk in model.astream([HumanMessage(content="请只回答：连接成功")], max_tokens=32):
                text += chunk.text
            if not text.strip():
                raise ValueError("流式输出为空")
        return {"status": "ok", "message": "连接成功，流式和结构化输出均可使用。"}
    except Exception as exc:
        # 第三方错误体可能含用户输入或凭据，不将原始异常写入日志或返回。
        logger.warning("模型连接测试失败：%s", type(exc).__name__)
        raise HTTPException(status_code=400, detail="连接测试失败，请检查接口地址、API Key、模型名称及结构化输出方式。") from None
