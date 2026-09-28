import asyncio

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.agents.chat_graph import ChatContext, SearchPlan, build_chat_graph
from app.infrastructure.database.database import get_session
from app.main import app
from app.memory.types import MemoryDecision
from app.rag.evaluate import score_case
from app.rag.evaluate_answers import score_answer
from app.rag.parser import parse_book, split_book
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService


def test_parse_and_split_book():
    pages = parse_book("example.md", "第一章\n\nLangGraph 保存状态。".encode())
    chunks = split_book("a" * 32, 7, "测试书籍", pages)
    assert len(chunks) == 1
    assert chunks[0].page == -1
    assert chunks[0].user_id == 7
    assert "LangGraph" in chunks[0].text
    assert score_case(["wrong", chunks[0].id], {chunks[0].id}) == (1.0, 0.5)
    assert score_answer("Checkpoint 保存状态 [1]", [{"book_id": "book-a"}],
                        ["保存状态"], {"book-a"})["valid_citations"] is True


def test_chat_graph_uses_book_sources(monkeypatch):
    hit = RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=7,
                                   title="LangGraph 教材", page=4,
                                   text="Checkpoint 保存图的状态。"), score=0.95)

    async def fake_search(self, session, user_id, question):
        assert user_id == 7
        assert question == "Checkpoint 是什么？"
        return [hit]

    class DecisionModel:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt):
            if self.schema is SearchPlan:
                return SearchPlan(route="book_search", search_query="Checkpoint 是什么？")
            return MemoryDecision(should_save=False)

    class FakeModel:
        messages = None

        def with_structured_output(self, schema):
            return DecisionModel(schema)

        async def ainvoke(self, messages):
            self.messages = messages
            return AIMessage(content="它保存图状态 [1]。")

    monkeypatch.setattr(KnowledgeService, "search", fake_search)
    model = FakeModel()
    graph = build_chat_graph(model, InMemorySaver(), InMemoryStore())

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Checkpoint 是什么？")]},
            config={"configurable": {"thread_id": "rag-test"}},
            context=ChatContext(user_id=7),
        )
        answer = result["messages"][-1]
        assert answer.additional_kwargs["sources"][0]["book_id"] == hit.chunk.book_id
        assert any(isinstance(message, SystemMessage) and "Checkpoint 保存图的状态" in message.content
                   for message in model.messages)

    asyncio.run(run())


def test_chat_graph_abstains_without_book_evidence(monkeypatch):
    async def empty_search(self, session, user_id, question):
        return []

    class DecisionModel:
        async def ainvoke(self, prompt):
            return MemoryDecision(should_save=False)

    class FakeModel:
        def with_structured_output(self, schema):
            return DecisionModel()

        async def ainvoke(self, messages):
            raise AssertionError("没有书籍依据时不应调用回答模型")

    monkeypatch.setattr(KnowledgeService, "search", empty_search)
    graph = build_chat_graph(FakeModel(), InMemorySaver(), InMemoryStore())

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="RAG 的 checkpoint 是什么？")]},
            config={"configurable": {"thread_id": "no-evidence-test"}},
            context=ChatContext(user_id=7),
        )
        assert "没有足够依据" in result["messages"][-1].content
        assert result["messages"][-1].additional_kwargs["sources"] == []

    asyncio.run(run())


def test_chat_graph_routes_greeting_without_search(monkeypatch):
    async def unexpected_search(self, session, user_id, question):
        raise AssertionError("普通问候不应检索书籍")

    class DecisionModel:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt):
            if self.schema is SearchPlan:
                return SearchPlan(route="direct", search_query="")
            return MemoryDecision(should_save=False)

    class FakeModel:
        def with_structured_output(self, schema):
            return DecisionModel(schema)

        async def ainvoke(self, messages):
            return AIMessage(content="你好！")

    monkeypatch.setattr(KnowledgeService, "search", unexpected_search)
    graph = build_chat_graph(FakeModel(), InMemorySaver(), InMemoryStore())

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="你好")]},
            config={"configurable": {"thread_id": "direct-test"}},
            context=ChatContext(user_id=7),
        )
        assert result["route"] == "direct"
        assert result["messages"][-1].content == "你好！"

    asyncio.run(run())


def test_book_upload_requires_owner(tmp_path, monkeypatch):
    async def no_process(self, book_id, user_id):
        pass

    monkeypatch.setattr(settings, "BOOK_STORAGE_DIR", tmp_path / "books")
    monkeypatch.setattr(KnowledgeService, "process", no_process)

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
                tokens = []
                for name in ("alice", "bob"):
                    response = await client.post("/api/v1/auth/register", json={
                        "username": name, "email": f"{name}@example.com", "password": "pass12345",
                    })
                    tokens.append(response.json()["access_token"])
                headers = {"Authorization": f"Bearer {tokens[0]}"}
                uploaded = await client.post("/api/v1/knowledge/books", headers=headers,
                    files={"file": ("book.txt", b"LangGraph and RAG", "text/plain")})
                assert uploaded.status_code == 202, uploaded.text
                book_id = uploaded.json()["id"]
                assert len((await client.get("/api/v1/knowledge/books", headers=headers)).json()) == 1
                other = {"Authorization": f"Bearer {tokens[1]}"}
                assert (await client.get("/api/v1/knowledge/books", headers=other)).json() == []
                assert (await client.post(f"/api/v1/knowledge/books/{book_id}/retry", headers=other)).status_code == 404
        finally:
            app.dependency_overrides.clear()
            await engine.dispose()

    asyncio.run(run())
