import json
import logging
from collections.abc import AsyncIterator #AsyncIterator[str] 表示我们的异步生成器会不断产生字符串


from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse #StreamingResponse 负责边产生数据边发送 HTTP 响应。
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,ToolMessage
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.chat_graph import ChatContext
from app.infrastructure.database.database import get_session
from app.schemas.chat import ChatRequest, ChatResponse,ChatMessageRead,ChatHistoryResponse
from app.services.conversation_service import ConversationService
from app.api.v1.deps import get_current_user
from app.models.user import User


router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(data: ChatRequest, request: Request, session: AsyncSession = Depends(get_session),current_user:User=Depends(get_current_user)):
    service=ConversationService()
    assert current_user.id is not None
    try:
        conversation = await service.get_owned(session,data.conversation_id,user_id=current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    graph = request.app.state.chat_graph

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=data.message)]},
        config={"configurable": {"thread_id": conversation.thread_id}},
        context=ChatContext(user_id=conversation.user_id),
    )

    await service.touch(session, conversation)
    answer = result["messages"][-1]
    return ChatResponse(conversation_id=data.conversation_id, answer=str(answer.content),
                        sources=answer.additional_kwargs.get("sources", []))

@router.post("/stream")
async def chat_stream(data:ChatRequest,request: Request,session: AsyncSession = Depends(get_session),current_user:User=Depends(get_current_user)):
    service = ConversationService()
    assert current_user.id is not None
    try:
        conversation = await service.get_owned(session, data.conversation_id,current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    graph = request.app.state.chat_graph

    config = {"configurable": {"thread_id": conversation.thread_id}}
    context = ChatContext(user_id=current_user.id)

    await service.touch(session, conversation)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for message, metadata in graph.astream(
                {"messages": [HumanMessage(content=data.message)]},
                config=config,
                context=context,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") not in {"model", "grounded_answer", "direct_answer"}:
                    continue
                if isinstance(message.content, str) and message.content:
                    event_data = json.dumps({"type": "token", "content": message.content}, ensure_ascii=False)
                    yield f"data: {event_data}\n\n"
            state = await graph.aget_state(config)
            sources = state.values.get("sources", [])
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception:
            logger.exception("聊天流生成失败")
            yield 'data: {"type": "error", "content": "生成失败，请查看后端日志"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{conversation_id}/history",response_model=ChatHistoryResponse)
async def get_chat_history(conversation_id:int,request:Request,session:AsyncSession=Depends(get_session),current_user:User=Depends(get_current_user)):
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
                                      if isinstance(message, AIMessage) else []))

    return ChatHistoryResponse(conversation_id=conversation_id, messages=result)
