"""验证自定义模型的隔离、密钥保护、实际选择及切换后的历史连续性。"""

import asyncio
from contextlib import asynccontextmanager
import pytest

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.chat_graph import SearchPlan, build_chat_graph
from app.agents.context import ContextLimits
from app.api.v1.models import ConnectionDecision
from app.core.config import settings
from app.infrastructure.database.database import get_session
from app.main import app
from app.memory.types import MemoryDecision
from app.models.model_profile import ModelProfile
from app.rag.grounding import GroundingDecision
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService
from app.services.model_service import ModelService, ModelSpec, credential_cipher
from app.llm.structured import output_instructions


def test_builtin_providers_use_their_own_credentials(monkeypatch):
    """两个平台分别解析凭据；公开列表不泄露密钥，缺少官方 Key 时不展示选项。"""
    monkeypatch.setattr(settings, "DeepSeeK_API_KEY", "dummy-official-secret")
    monkeypatch.setattr(settings, "DeepSeek_BASE_URL", "https://official.example/v1")
    monkeypatch.setattr(settings, "DEEPSEEK_CHAT_MODEL", "official-model")
    monkeypatch.setattr(settings, "GuiJiLiuDong_API_KEY", "dummy-silicon-secret")
    monkeypatch.setattr(settings, "GuiJiLiuDong_BASE_URL", "https://silicon.example/v1")
    monkeypatch.setattr(settings, "CHAT_MODEL_OPTIONS", "openai:extra-model")
    service = ModelService()
    models = service.builtin_models()
    official = next(item for item in models if item.id == "builtin:deepseek")
    silicon = next(item for item in models if item.model == "openai:extra-model")
    public = str([item.model_dump() for item in models])
    assert "dummy-official-secret" not in public and "dummy-silicon-secret" not in public
    assert official.model == "official-model"

    async def run():
        assert await service.resolve(None, 7, "default") is None
        spec = await service.resolve(None, 7, official.id)
        assert spec.api_key == "dummy-official-secret"
        assert spec.base_url == "https://official.example/v1"
        assert spec.model == "official-model"
        assert spec.structured_output_method == "function_calling"
        spec = await service.resolve(None, 7, silicon.id)
        assert spec.api_key == "dummy-silicon-secret"
        assert spec.base_url == "https://silicon.example/v1"
        monkeypatch.setattr(settings, "DeepSeek_BASE_URL", None)
        assert (await service.resolve(None, 7, official.id)).base_url == "https://api.deepseek.com/v1"

    asyncio.run(run())
    monkeypatch.setattr(settings, "DeepSeeK_API_KEY", " ")
    assert all(item.id != "builtin:deepseek" for item in service.builtin_models())


def test_model_configuration_and_switch_preserve_history(tmp_path, monkeypatch):
    captured = []

    class Structured:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, **kwargs):
            if self.schema is SearchPlan:
                return SearchPlan(search_query=prompt[-1].content)
            if self.schema is GroundingDecision:
                return GroundingDecision(supported=True)
            if self.schema is ConnectionDecision:
                return ConnectionDecision(connected=True)
            return MemoryDecision(should_save=False)

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def with_structured_output(self, schema, **kwargs):
            return Structured(schema)

        async def astream(self, messages, **kwargs):
            yield AIMessageChunk(content=f"由 {self.name} 回答 [1]。")

    @asynccontextmanager
    async def fake_runtime_model(self, spec):
        captured.append(spec)
        yield FakeModel(spec.model)

    async def fake_search(self, session, user_id, question):
        return [RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=user_id,
                                        title="测试教材", page=1, text="测试证据。"), score=0.9)]

    monkeypatch.setattr(ModelService, "runtime_model", fake_runtime_model)
    monkeypatch.setattr(KnowledgeService, "search", fake_search)
    monkeypatch.setattr(settings, "DeepSeeK_API_KEY", "dummy-official-secret")
    monkeypatch.setattr(settings, "DEEPSEEK_CHAT_MODEL", "official-model")

    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'models.db').as_posix()}")
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

        async def session_override():
            async with AsyncSession(engine) as session:
                yield session

        app.dependency_overrides[get_session] = session_override
        previous = {key: getattr(app.state, key, None) for key in ("chat_graph", "checkpointer", "store")}
        app.state.checkpointer = InMemorySaver()
        app.state.store = None
        app.state.chat_graph = build_chat_graph(FakeModel("default"), app.state.checkpointer, None)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                headers = []
                for username in ("alice", "bob"):
                    registered = await client.post("/api/v1/auth/register", json={
                        "username": username, "email": f"{username}@example.com", "password": "pass12345",
                    })
                    headers.append({"Authorization": f"Bearer {registered.json()['access_token']}"})
                first, second = headers
                data = {"name": "模型甲", "model": "model-a", "base_url": "https://example.com/v1",
                        "api_key": "test-secret-only", "context_window": 8192, "max_output_tokens": 512}
                created = await client.post("/api/v1/models", json=data, headers=first)
                assert created.status_code == 201, created.text
                model_a = created.json()["id"]
                assert created.json()["has_api_key"] is True
                assert "test-secret-only" not in created.text
                assert "encrypted_api_key" not in created.text and '"api_key"' not in created.text
                async with AsyncSession(engine) as session:
                    row = (await session.exec(select(ModelProfile))).one()
                    assert row.encrypted_api_key != data["api_key"]
                    assert credential_cipher().decrypt(row.encrypted_api_key.encode()).decode() == data["api_key"]
                own = await client.get("/api/v1/models", headers=first)
                other = await client.get("/api/v1/models", headers=second)
                assert model_a in [item["id"] for item in own.json()]
                assert model_a not in [item["id"] for item in other.json()]
                assert (await client.get("/api/v1/models")).status_code in (401, 403)
                assert (await client.delete(f"/api/v1/models/{model_a}", headers=second)).status_code == 404

                invalid = await client.post("/api/v1/models", json={**data, "max_output_tokens": 32768}, headers=first)
                assert invalid.status_code == 422
                assert "test-secret-only" not in invalid.text
                assert (await client.post("/api/v1/models", json={**data, "base_url": "file:///tmp/key"}, headers=first)).status_code == 422
                tested = await client.post("/api/v1/models/test", json=data, headers=first)
                assert tested.status_code == 200, tested.text
                captured.clear()

                conversation = (await client.post("/api/v1/conversations", json={"title": "历史测试"}, headers=first)).json()
                other_conversation = (await client.post("/api/v1/conversations", json={"title": "其他用户"}, headers=second)).json()
                forbidden = await client.post("/api/v1/chat/stream", headers=second, json={
                    "conversation_id": other_conversation["id"], "message": "RAG", "model_id": model_a,
                })
                assert forbidden.status_code == 404 and not captured
                response = await client.post("/api/v1/chat/stream", headers=first, json={
                    "conversation_id": conversation["id"], "message": "RAG 是什么？", "model_id": model_a,
                })
                assert '"type": "done"' in response.text
                assert "model-a" in response.text
                assert captured[0].api_key == data["api_key"]
                assert captured[0].context_window == 8192

                created_b = await client.post("/api/v1/models", headers=first, json={
                    **data, "name": "模型乙", "model": "model-b", "api_key": "", "context_window": 4096,
                })
                assert created_b.status_code == 201, created_b.text
                assert created_b.json()["has_api_key"] is False
                model_b = created_b.json()["id"]
                answer_b = await client.post("/api/v1/chat", headers=first, json={
                    "conversation_id": conversation["id"], "message": "它有什么作用？", "model_id": model_b,
                })
                assert answer_b.status_code == 200, answer_b.text
                assert "model-b" in answer_b.json()["answer"]
                history = (await client.get(f"/api/v1/chat/{conversation['id']}/history", headers=first)).json()["messages"]
                assert len(history) == 4
                assert "model-a" in history[1]["content"] and "model-b" in history[3]["content"]
                assert "test-secret-only" not in str(history)
                official_reply = await client.post("/api/v1/chat/stream", headers=first, json={
                    "conversation_id": conversation["id"], "message": "继续解释 RAG", "model_id": "builtin:deepseek",
                })
                assert '"type": "done"' in official_reply.text
                assert "official-model" in official_reply.text
                assert captured[-1].api_key == "dummy-official-secret"
                history = (await client.get(f"/api/v1/chat/{conversation['id']}/history", headers=first)).json()["messages"]
                assert len(history) == 6 and "model-a" in history[1]["content"]
                assert "dummy-official-secret" not in str(history)
                assert (await client.delete(f"/api/v1/models/{model_a}", headers=first)).status_code == 204
                unchanged = (await client.get(f"/api/v1/chat/{conversation['id']}/history", headers=first)).json()["messages"]
                assert unchanged == history
                assert (await client.post("/api/v1/chat", headers=first, json={
                    "conversation_id": conversation["id"], "message": "继续", "model_id": model_a,
                })).status_code == 404
        finally:
            app.dependency_overrides.clear()
            for key, value in previous.items():
                if value is None and hasattr(app.state, key):
                    delattr(app.state, key)
                elif value is not None:
                    setattr(app.state, key, value)
            await engine.dispose()

    asyncio.run(run())


def test_runtime_clients_close_and_window_limits_are_independent(monkeypatch):
    import app.services.model_service as module

    clients = []

    def fake_constructor(**kwargs):
        clients.extend([kwargs["http_client"], kwargs["http_async_client"]])
        assert kwargs["api_key"] == "local-no-key"
        assert kwargs["model"] == "model-a"
        return object()

    monkeypatch.setattr(module, "ChatOpenAI", fake_constructor)

    async def run():
        spec = ModelSpec("openai:model-a", "http://localhost:11434/v1", "", 4096, 512)
        async with ModelService().runtime_model(spec):
            assert not any(client.is_closed for client in clients)
        assert all(client.is_closed for client in clients)
        default = ContextLimits.from_settings()
        small = ContextLimits.from_settings(4096, 512)
        assert small.CHAT_CONTEXT_WINDOW_TOKENS == 4096
        assert small.CONTEXT_HISTORY_BUDGET_TOKENS < default.CONTEXT_HISTORY_BUDGET_TOKENS
        assert settings.CHAT_CONTEXT_WINDOW_TOKENS == default.CHAT_CONTEXT_WINDOW_TOKENS
        assert small != default

    asyncio.run(run())


def test_custom_provider_error_body_is_not_logged(monkeypatch, caplog):
    """即使第三方错误体回显密钥，节点核对失败的日志也只保留异常类型。"""
    class Structured:
        async def ainvoke(self, *args, **kwargs):
            raise ValueError("供应商错误体包含 dummy-private-secret")

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return Structured()

        async def astream(self, *args, **kwargs):
            yield AIMessageChunk(content="这是有引用的草稿 [1]。")

    async def search(self, session, user_id, question):
        return [RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=user_id,
                                        title="测试书籍", page=1, text="测试依据。"), score=0.9)]

    monkeypatch.setattr(KnowledgeService, "search", search)

    async def run():
        from langchain_core.messages import HumanMessage
        from app.agents.chat_graph import ChatContext
        graph = build_chat_graph(Model(), InMemorySaver(), None, structured_output_method="function_calling")
        result = await graph.ainvoke({"messages": [HumanMessage(content="RAG 是什么？")]},
                                    config={"configurable": {"thread_id": "private-errors"}},
                                    context=ChatContext(user_id=7))
        assert "没有足够依据" in result["messages"][-1].content
        assert "dummy-private-secret" not in caplog.text
        assert "ValueError" in caplog.text

    asyncio.run(run())


@pytest.mark.parametrize("method", ["function_calling", "json_schema", "json_mode"])
def test_real_sdk_uses_custom_parameters_for_structured_and_stream_calls(monkeypatch, method):
    """使用真实 ChatOpenAI 与 OpenAI SDK，但用本地 MockTransport 截获所有 HTTP 请求。"""
    import json
    import httpx
    import app.services.model_service as module

    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append((request, body))
        assert str(request.url) == "https://example.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer dummy-test-key"
        assert body["model"] == "custom-model"
        if body.get("stream"):
            chunk = {"id": "test-stream", "object": "chat.completion.chunk", "created": 1,
                     "model": "custom-model", "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "连接成功"}, "finish_reason": None}]}
            end = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            content = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content.encode())
        if not body.get("tools"):
            assert body["response_format"]["type"] == ("json_object" if method == "json_mode" else "json_schema")
            if method == "json_mode":
                assert "JSON" in str(body["messages"])
            return httpx.Response(200, json={"id": "json-call", "object": "chat.completion", "created": 1,
                "model": "custom-model", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"connected":true}', "refusal": None}}]})
        name = body["tools"][0]["function"]["name"]
        return httpx.Response(200, json={"id": "test-call", "object": "chat.completion", "created": 1,
            "model": "custom-model", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call-test", "type": "function",
                    "function": {"name": name, "arguments": '{"connected": true}'}}]}}]})

    async_client, sync_client = httpx.AsyncClient, httpx.Client
    class MockAsyncClient(async_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    class MockSyncClient(sync_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", MockAsyncClient)
    monkeypatch.setattr(module.httpx, "Client", MockSyncClient)

    async def run():
        spec = ModelSpec("custom-model", "https://example.com/v1", "dummy-test-key", 8192, 512,
                         temperature=0.3, top_p=0.9, structured_output_method=method)
        async with ModelService().runtime_model(spec) as model:
            result = await model.with_structured_output(ConnectionDecision, method=method).ainvoke(
                "返回 connected=true" + output_instructions(ConnectionDecision, method), max_tokens=128,
            )
            assert result.connected
            content = "".join([chunk.text async for chunk in model.astream("连接测试", max_tokens=32)])
            assert content == "连接成功"
        assert len(calls) == 2
        assert all(body["temperature"] == 0.3 and body["top_p"] == 0.9 for _, body in calls)
        assert calls[0][1].get("max_completion_tokens", calls[0][1].get("max_tokens")) == 128
        assert calls[1][1].get("max_completion_tokens", calls[1][1].get("max_tokens")) == 32

    asyncio.run(run())
