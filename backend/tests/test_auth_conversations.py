import asyncio

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.infrastructure.database.database import get_session
from app.main import app
from app.models.conversation import Conversation  # noqa: F401 - register table
from app.models.user import User  # noqa: F401 - register table


def test_register_login_and_owned_conversations(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

        async def session_override():
            async with AsyncSession(engine) as session:
                yield session

        app.dependency_overrides[get_session] = session_override
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                registered = await client.post("/api/v1/auth/register", json={
                    "username": "alice", "email": "alice@example.com", "password": "pass12345",
                })
                assert registered.status_code == 201, registered.text
                token = registered.json()["access_token"]

                duplicate = await client.post("/api/v1/auth/register", json={
                    "username": "alice", "email": "another@example.com", "password": "pass12345",
                })
                assert duplicate.status_code == 409

                login = await client.post("/api/v1/auth/login", json={
                    "username": "alice@example.com", "password": "pass12345",
                })
                assert login.status_code == 200, login.text

                unauthorized = await client.post("/api/v1/conversations", json={"title": "Private"})
                assert unauthorized.status_code in (401, 403)

                created = await client.post("/api/v1/conversations", json={"title": "LangGraph"}, headers={"Authorization": f"Bearer {token}"})
                assert created.status_code == 201, created.text
                conversation_id = created.json()["id"]

                second = await client.post("/api/v1/auth/register", json={
                    "username": "bob", "email": "bob@example.com", "password": "pass12345",
                })
                other_token = second.json()["access_token"]
                other_headers = {"Authorization": f"Bearer {other_token}"}
                assert (await client.get(f"/api/v1/conversations/{conversation_id}", headers=other_headers)).status_code == 404
                assert (await client.get("/api/v1/conversations", headers=other_headers)).json() == []

                class FakeGraph:
                    async def ainvoke(self, *args, **kwargs):
                        return {"messages": [AIMessage(content="你好", additional_kwargs={"sources": []})]}

                app.state.chat_graph = FakeGraph()
                streamed = await client.post("/api/v1/chat/stream", json={
                    "conversation_id": conversation_id, "message": "你好",
                }, headers={"Authorization": f"Bearer {token}"})
                assert streamed.status_code == 200, streamed.text
                assert streamed.headers["content-type"].startswith("text/event-stream")
                assert '"type": "token", "content": "你好"' in streamed.text
                assert '"type": "sources", "sources": []' in streamed.text
                assert '"type": "done"' in streamed.text

                blocked = await client.post("/api/v1/chat/stream", json={
                    "conversation_id": conversation_id, "message": "你好",
                }, headers=other_headers)
                assert blocked.status_code == 404
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, "chat_graph"):
                del app.state.chat_graph
            await engine.dispose()

    asyncio.run(run())
