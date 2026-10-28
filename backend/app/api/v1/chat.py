"""聊天接口：校验会话归属，执行工作流，将增量和最终结果转成 SSE。"""

import json
import logging
from collections.abc import AsyncIterator #AsyncIterator[str] 表示我们的异步生成器会不断产生字符串


from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse #StreamingResponse 负责边产生数据边发送 HTTP 响应。
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,ToolMessage
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.chat_graph import ChatContext
from app.agents.context import ContextBudgetError
from app.agents.retry import RetryableRetrievalError
from app.artifacts.errors import ArtifactError, ArtifactProviderError
from app.rag.reranker import RerankError
from app.infrastructure.database.database import get_session
from app.schemas.chat import ChatRequest, ChatResponse,ChatMessageRead,ChatHistoryResponse
from app.services.conversation_service import ConversationService
from app.api.v1.deps import get_current_user
from app.models.user import User
from app.services.model_service import ModelService


router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


async def resolve_model(models: ModelService, session: AsyncSession, user_id: int, model_id: str | None):
    """先检查模型配置的用户归属，让无权限或无效选择在流开始前返回 HTTP 错误。"""
    try:
        return await models.resolve(session, user_id, model_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("", response_model=ChatResponse)
async def chat(data: ChatRequest, request: Request, session: AsyncSession = Depends(get_session),current_user:User=Depends(get_current_user)):
    """校验会话归属，执行图后一次性返回最终答案和来源。"""
    service=ConversationService()
    assert current_user.id is not None
    try:
        conversation = await service.get_owned(session,data.conversation_id,user_id=current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    models = ModelService()
    spec = await resolve_model(models, session, current_user.id, data.model_id)

    try:
        async with models.runtime_graph(request.app, spec) as graph:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=data.message)]},
                config={"configurable": {"thread_id": conversation.thread_id}},
                context=ChatContext(user_id=conversation.user_id),
            )
    except (RerankError, RetryableRetrievalError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    except ContextBudgetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ArtifactProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    except ArtifactError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:
        if spec is None:
            raise
        logger.warning("所选模型聊天失败：%s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="所选模型调用失败，请检查连接、密钥及模型参数。") from None

    await service.touch(session, conversation)
    answer = result["messages"][-1]
    return ChatResponse(conversation_id=data.conversation_id, answer=str(answer.content),
                        sources=answer.additional_kwargs.get("sources", []),
                        artifacts=answer.additional_kwargs.get("artifacts", []))

@router.post("/stream")
async def chat_stream(data:ChatRequest,request: Request,session: AsyncSession = Depends(get_session),current_user:User=Depends(get_current_user)):
    """校验会话归属，组装 thread_id 和可信身份，返回 SSE 响应。"""
    service = ConversationService()
    assert current_user.id is not None
    try:
        conversation = await service.get_owned(session, data.conversation_id,current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    models = ModelService()
    spec = await resolve_model(models, session, current_user.id, data.model_id)

    config = {"configurable": {"thread_id": conversation.thread_id}}
    context = ChatContext(user_id=current_user.id)

    await service.touch(session, conversation)

    async def event_stream() -> AsyncIterator[str]:
        """将图的增量、候选来源和最终回答转换为 SSE；区分预算错误和运行异常。"""
        try:
            answer_received = False
            async with models.runtime_graph(request.app, spec) as graph:
                # custom 接收回答文字增量；updates 接收核对后的最终节点结果。
                async for mode, payload in graph.astream(
                    {"messages": [HumanMessage(content=data.message)]},
                    config=config,
                    context=context,
                    stream_mode=["custom", "updates"],
                ):
                    if mode == "custom" and payload.get("type") in ("token", "status"):
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    elif mode == "updates":
                        retrieval = payload.get("retrieve")
                        if retrieval and retrieval.get("sources"):
                            # 先发送候选来源供前端保存；最终核对结果会确认或清空，页面默认折叠详情。
                            event = {"type": "sources", "sources": retrieval["sources"], "pending": True}
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        for node in ("grounded_answer", "no_evidence", "run_artifact_tools"):
                            update = payload.get(node)
                            if not update or not update.get("messages"):
                                continue
                            answer = update["messages"][-1]
                            # 最终答案覆盖待核对文字，核对失败时同步清除候选答案及来源。
                            event = {"type": "answer", "content": str(answer.content)}
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                            sources = answer.additional_kwargs.get("sources", [])
                            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
                            artifacts = answer.additional_kwargs.get("artifacts", [])
                            if artifacts:
                                yield f"data: {json.dumps({'type': 'artifacts', 'artifacts': artifacts}, ensure_ascii=False)}\n\n"
                            answer_received = True
            if not answer_received:
                raise ValueError("聊天工作流没有返回最终答案")
            yield 'data: {"type": "done"}\n\n'
        except (RerankError, RetryableRetrievalError) as exc:
            # 这是重排服务故障，不是知识库无依据；提示用户检查配置或稍后重试。
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
        except ContextBudgetError as exc:
            # 用户可以通过缩短问题解决预算问题，不将配置或供应商原始异常暴露给前端。
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
        except ArtifactError as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            if spec is None:
                logger.exception("聊天流生成失败")
            else:
                logger.warning("所选模型聊天失败：%s", type(exc).__name__)
            message = "生成失败，请查看后端日志" if spec is None else "所选模型调用失败，请检查连接、密钥及模型参数；可在模型配置中测试连接。"
            yield f"data: {json.dumps({'type': 'error', 'content': message}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{conversation_id}/history",response_model=ChatHistoryResponse)
async def get_chat_history(conversation_id:int,request:Request,session:AsyncSession=Depends(get_session),current_user:User=Depends(get_current_user)):
    """按公开会话编号校验所有权，再从对应 checkpoint 读取完整消息。"""
    service=ConversationService()
    assert current_user.id is not None
    try:
        conversation = await service.get_owned(session, conversation_id, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404,detail=str(exc))

    graph=request.app.state.chat_graph
    state=await graph.aget_state({"configurable":{"thread_id":conversation.thread_id}})
    messages=state.values.get("messages",[])

    result:list[ChatMessageRead]=[]

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = message.type

        result.append(ChatMessageRead(role=role, content=str(message.content),
                                      sources=message.additional_kwargs.get("sources", [])
                                      if isinstance(message, AIMessage) else [],
                                      artifacts=message.additional_kwargs.get("artifacts", [])
                                      if isinstance(message, AIMessage) else []))

    return ChatHistoryResponse(conversation_id=conversation_id, messages=result)
