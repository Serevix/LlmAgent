import asyncio

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
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
from app.rag.grounding import GroundingDecision, citations_valid
from app.rag.parser import parse_book, split_book
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService
from app.models.knowledge_book import KnowledgeBook


class StreamingFakeModel:
    """为已有测试提供异步流式接口，不访问真实模型。"""

    async def astream(self, messages, **kwargs):
        response = await self.ainvoke(messages)
        yield AIMessageChunk(content=response.content)


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
    assert score_answer("当前知识库没有足够依据回答这个问题。", [], [], set(),
                        should_abstain=True)["abstain_correct"] is True
    assert score_answer("当然可以，我知道答案。", [], [], set(),
                        should_abstain=True)["abstain_correct"] is False


def test_enumeration_search_expands_strong_short_book(monkeypatch):
    book_id = "b" * 32
    hits = [
        RetrievedChunk(BookChunk(id=str(index) * 32, book_id=book_id, user_id=7,
                                 title="机器学习算法", page=index,
                                 text=f"第 {index} 页的算法"), score=score)
        for index, score in ((1, 0.74), (2, 0.57), (3, 0.49), (4, 0.42))
    ]

    class FakeEmbedding:
        async def embed(self, texts):
            assert texts == ["4个核心算法是什么"]
            return [[0.0] * 1024]

    class FakeBooks:
        async def list_ready_ids(self, session, user_id):
            assert user_id == 7
            return [book_id]

        async def get_owned(self, session, requested_id, user_id):
            assert requested_id == book_id and user_id == 7
            return KnowledgeBook(id=book_id, user_id=7, title="机器学习算法",
                                 filename="book.pdf", storage_path="book.pdf",
                                 sha256="a" * 64, status="ready", chunk_count=4)

    class FakeVectors:
        calls = []

        async def search(self, user_id, vector, limit, book_ids):
            self.calls.append((user_id, limit, book_ids))
            return hits[:2] if len(self.calls) == 1 else list(reversed(hits))

    monkeypatch.setattr("app.services.knowledge_service.EmbeddingService", FakeEmbedding)
    service = KnowledgeService(retrieval_mode="dense")
    service.books = FakeBooks()
    service.vectors = FakeVectors()
    result = asyncio.run(service.search(None, 7, "4个核心算法是什么"))

    assert [hit.chunk.page for hit in result] == [1, 2, 3, 4]
    assert service.vectors.calls == [(7, 5, [book_id]), (7, 4, [book_id])]


def test_numbered_question_keeps_original_search_terms(monkeypatch):
    question = "6个机器学习的核心算法是什么"
    hit = RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=7,
                                   title="机器学习算法", page=1,
                                   text="一、线性回归；二、逻辑回归。"), score=0.74)

    async def fake_search(self, session, user_id, search_query):
        assert user_id == 7 and search_query == question
        return [hit]

    class DecisionModel:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, **kwargs):
            if self.schema is SearchPlan:
                raise AssertionError("没有前文的首轮问题不需要改写")
            if self.schema is GroundingDecision:
                return GroundingDecision(supported=True)
            return MemoryDecision(should_save=False)

    class FakeModel(StreamingFakeModel):
        def with_structured_output(self, schema):
            return DecisionModel(schema)

        async def ainvoke(self, messages, **kwargs):
            return AIMessage(content="线性回归和逻辑回归 [1]。")

    monkeypatch.setattr(KnowledgeService, "search", fake_search)
    graph = build_chat_graph(FakeModel(), InMemorySaver(), None)

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "numbered-list-test"}},
            context=ChatContext(user_id=7),
        )
        assert result["search_query"] == question

    asyncio.run(run())


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

        async def ainvoke(self, prompt, **kwargs):
            if self.schema is SearchPlan:
                return SearchPlan(search_query="Checkpoint 是什么？")
            if self.schema is GroundingDecision:
                return GroundingDecision(supported=True)
            return MemoryDecision(should_save=False)

    class FakeModel(StreamingFakeModel):
        messages = None

        def with_structured_output(self, schema):
            return DecisionModel(schema)

        async def ainvoke(self, messages, **kwargs):
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
        async def ainvoke(self, prompt, **kwargs):
            return MemoryDecision(should_save=False)

    class FakeModel(StreamingFakeModel):
        def with_structured_output(self, schema):
            return DecisionModel()

        async def ainvoke(self, messages, **kwargs):
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


def test_chat_graph_does_not_answer_from_model_without_book_evidence(monkeypatch):
    async def empty_search(self, session, user_id, question):
        return []

    class DecisionModel:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, **kwargs):
            if self.schema is SearchPlan:
                return SearchPlan(search_query="你好")
            return MemoryDecision(should_save=False)

    class FakeModel(StreamingFakeModel):
        def with_structured_output(self, schema):
            return DecisionModel(schema)

        async def ainvoke(self, messages, **kwargs):
            raise AssertionError("无书籍依据时不应调用回答模型")

    monkeypatch.setattr(KnowledgeService, "search", empty_search)
    graph = build_chat_graph(FakeModel(), InMemorySaver(), InMemoryStore())

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="你好")]},
            config={"configurable": {"thread_id": "direct-test"}},
            context=ChatContext(user_id=7),
        )
        assert result["route"] == "book_search"
        assert "没有足够依据" in result["messages"][-1].content

    asyncio.run(run())


def test_grounding_rejects_invalid_citations_and_unsupported_answer(monkeypatch):
    hit = RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=7,
                                   title="LangGraph 教材", page=4,
                                   text="Checkpoint 保存图的状态。"), score=0.95)

    async def fake_search(self, session, user_id, question):
        return [hit]

    class DecisionModel:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, **kwargs):
            if self.schema is SearchPlan:
                return SearchPlan(search_query="Checkpoint 是什么？")
            if self.schema is GroundingDecision:
                return GroundingDecision(supported=False)
            return MemoryDecision(should_save=False)

    class FakeModel(StreamingFakeModel):
        def with_structured_output(self, schema):
            return DecisionModel(schema)

        async def ainvoke(self, messages, **kwargs):
            return AIMessage(content="Checkpoint 会自动备份所有数据库 [1]。")

    monkeypatch.setattr(KnowledgeService, "search", fake_search)
    graph = build_chat_graph(FakeModel(), InMemorySaver(), None)

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Checkpoint 是什么？")]},
            config={"configurable": {"thread_id": "unsupported-answer-test"}},
            context=ChatContext(user_id=7),
        )
        assert "没有足够依据" in result["messages"][-1].content
        assert result["messages"][-1].additional_kwargs["sources"] == []

    asyncio.run(run())
    assert citations_valid("它保存状态 [1]。", 1)
    assert not citations_valid("它保存状态。", 1)
    assert not citations_valid("它保存状态 [2]。", 1)


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
