"""验证减少辅助模型请求的同时，保留追问、长期记忆和书籍依据核对。"""

import asyncio
from collections import Counter

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agents.call_policy import needs_memory_extraction, needs_query_rewrite
from app.agents.chat_graph import ChatContext, SearchPlan, build_chat_graph
from app.core.config import settings
from app.memory.types import MemoryDecision
from app.rag.grounding import GroundingDecision
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService


@pytest.fixture(autouse=True)
def default_call_policy(monkeypatch):
    """固定本组测试的默认策略，避免开发者自己的 .env 改变预期调用次数。"""
    monkeypatch.setattr(settings, "CHAT_QUERY_REWRITE_MODE", "auto")
    monkeypatch.setattr(settings, "CHAT_MEMORY_MODE", "selective")


@pytest.mark.parametrize("question,expected", [
    ("什么是 Milvus 的混合检索？", False),
    ("为什么 QA 抽取过程需要保存中间结果并在请求失败时重试？", False),
    ("如何实现上下文管理？", False),
    ("它有哪3种用途？", True),
    ("LangGraph 和刚才的框架有什么区别？", True),
    ("支持并发吗？", True),
    ("RAG 呢？", True),
    ("怎么用？", True),
    ("继续讲解", True),
    ("What are its limitations?", True),
])
def test_rewrite_gate_preserves_context_dependent_questions(question, expected):
    """独立问题不改写；含专业词、数量或英文指代的追问仍能触发补全。"""
    assert needs_query_rewrite(question, has_history=True) is expected
    assert not needs_query_rewrite(question, has_history=False)
    assert needs_query_rewrite(question, has_history=True, mode="always")


@pytest.mark.parametrize("message,expected", [
    ("Checkpoint 是什么？", False),
    ("这次列出3条即可。", False),
    ("记住，我喜欢先看代码再看解释。", True),
    ("我正在学习 LangGraph。", True),
    ("我的学习目标是成为大模型应用开发工程师。", True),
    ("以后回答请使用中文。", True),
    ("我是一名后端开发者。", True),
])
def test_memory_gate_only_selects_stable_information_candidates(message, expected):
    """筛选只是决定要不要请求提炼模型，不直接把用户原文存成记忆。"""
    assert needs_memory_extraction(message) is expected
    assert not needs_memory_extraction(message, "off")
    assert needs_memory_extraction(message, "always")


class CountedModel:
    """记录实际推理调用，而不是把 with_structured_output 的装配计为请求。"""

    def __init__(self, supported=True):
        """设置核对结果并初始化每种任务的请求计数。"""
        self.calls = Counter()
        self.supported = supported

    def with_structured_output(self, schema):
        """构造不同任务的模拟模型，用同一计数器观察完整工作流。"""
        parent = self

        class Structured:
            async def ainvoke(self, prompt, **kwargs):
                """模拟问题补全、依据核对或记忆提炼，并分别记录调用次数。"""
                parent.calls[schema.__name__] += 1
                if schema is SearchPlan:
                    return SearchPlan(search_query="Checkpoint 有什么作用？")
                if schema is GroundingDecision:
                    return GroundingDecision(supported=parent.supported)
                assert schema is MemoryDecision
                return MemoryDecision(should_save=True, category="preference", content="喜欢先看代码。")

        return Structured()

    async def astream(self, messages, **kwargs):
        """模拟真实的回答增量，与核对模型分开计数。"""
        self.calls["answer"] += 1
        yield AIMessageChunk(content="Checkpoint 保存图状态 [1]。")


def install_search(monkeypatch, queries, has_evidence=True):
    """替换检索外部依赖，保留实际工具调用和图的条件路由。"""
    async def search(self, session, user_id, question):
        """记录送进检索工具的问题，并按测试需要返回证据或空列表。"""
        assert user_id == 7
        queries.append(question)
        return [RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=7,
                                        title="测试书籍", page=1, text="Checkpoint 保存图状态。"),
                               score=0.9)] if has_evidence else []

    monkeypatch.setattr(KnowledgeService, "search", search)


def test_ordinary_multiturn_uses_two_calls_and_followup_still_rewrites(monkeypatch):
    """两轮独立问题各只有生成和核对；第三轮有指代，才额外请求一次改写。"""
    queries = []
    install_search(monkeypatch, queries)

    async def run():
        model, store = CountedModel(), InMemoryStore()
        graph = build_chat_graph(model, InMemorySaver(), store)
        config = {"configurable": {"thread_id": "counted-chat"}}
        for question in ("Checkpoint 是什么？", "什么是 Milvus 混合检索？"):
            await graph.ainvoke({"messages": [HumanMessage(content=question)]},
                                config=config, context=ChatContext(user_id=7))
        assert model.calls == {"answer": 2, "GroundingDecision": 2}
        result = await graph.ainvoke({"messages": [HumanMessage(content="它有什么作用？")]},
                                    config=config, context=ChatContext(user_id=7))
        assert model.calls == {"answer": 3, "GroundingDecision": 3, "SearchPlan": 1}
        assert queries == ["Checkpoint 是什么？", "什么是 Milvus 混合检索？", "Checkpoint 有什么作用？"]
        assert len(result["messages"]) == 6
        assert not await store.asearch(("users", "7", "memories"))

    asyncio.run(run())


def test_explicit_preference_is_still_extracted_and_saved(monkeypatch):
    """明确的偏好经过模型提炼后写入当前用户 Store，不能因提速丢掉记忆功能。"""
    install_search(monkeypatch, [])

    async def run():
        model, store = CountedModel(), InMemoryStore()
        graph = build_chat_graph(model, InMemorySaver(), store)
        await graph.ainvoke({"messages": [HumanMessage(content="记住，我喜欢先看代码；Checkpoint 是什么？")]},
                            config={"configurable": {"thread_id": "preference"}},
                            context=ChatContext(user_id=7))
        assert model.calls == {"answer": 1, "GroundingDecision": 1, "MemoryDecision": 1}
        saved = await store.asearch(("users", "7", "memories"))
        assert len(saved) == 1 and saved[0].value["content"] == "喜欢先看代码。"
        assert not await store.asearch(("users", "8", "memories"))

    asyncio.run(run())


@pytest.mark.parametrize("has_evidence", [True, False])
def test_abstention_is_preserved_without_unnecessary_memory_call(monkeypatch, has_evidence):
    """无检索证据不调用聊天模型；有证据但核对不通过，最终仍弃答并清空来源。"""
    install_search(monkeypatch, [], has_evidence)

    async def run():
        model = CountedModel(supported=False)
        graph = build_chat_graph(model, InMemorySaver(), InMemoryStore())
        result = await graph.ainvoke({"messages": [HumanMessage(content="Checkpoint 是什么？")]},
                                    config={"configurable": {"thread_id": "abstain"}},
                                    context=ChatContext(user_id=7))
        assert "没有足够依据" in result["messages"][-1].content
        assert result["messages"][-1].additional_kwargs["sources"] == []
        assert model.calls == ({"answer": 1, "GroundingDecision": 1} if has_evidence else {})

    asyncio.run(run())


def test_original_policies_can_be_restored_from_configuration(monkeypatch):
    """always 恢复旧调用行为，便于比较或在规则漏判复杂追问时切回原策略。"""
    monkeypatch.setattr(settings, "CHAT_QUERY_REWRITE_MODE", "always")
    monkeypatch.setattr(settings, "CHAT_MEMORY_MODE", "always")
    install_search(monkeypatch, [])

    async def run():
        model = CountedModel()
        graph = build_chat_graph(model, InMemorySaver(), InMemoryStore())
        config = {"configurable": {"thread_id": "legacy-policy"}}
        for question in ("Checkpoint 是什么？", "什么是 Milvus 混合检索？"):
            await graph.ainvoke({"messages": [HumanMessage(content=question)]},
                                config=config, context=ChatContext(user_id=7))
        assert model.calls == {"answer": 2, "GroundingDecision": 2, "MemoryDecision": 2, "SearchPlan": 1}

    asyncio.run(run())
