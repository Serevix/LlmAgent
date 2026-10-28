"""验证长对话摘要、输入预算、追问改写及完整历史保留，不调用真实模型。"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.context import (ContextBudgetError, ContextManager, clip_text,
                                ensure_input_budget, message_tokens, text_tokens)
from app.agents.chat_graph import ChatContext, SearchPlan, build_chat_graph
from app.core.config import settings
from app.rag.grounding import GroundingDecision
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService


def small_budgets(monkeypatch):
    """降低测试阈值，少量消息即可触发真实的摘要分支。"""
    for name, value in {
        "CONTEXT_HISTORY_BUDGET_TOKENS": 700,
        "CONTEXT_SUMMARY_TRIGGER_TOKENS": 700,
        "CONTEXT_RECENT_MESSAGES": 2,
        "CONTEXT_SUMMARY_BATCH_TOKENS": 350,
        "CONTEXT_SUMMARY_MAX_TOKENS": 240,
    }.items():
        monkeypatch.setattr(settings, name, value)


def test_incremental_summary_is_bounded_and_history_is_unchanged(monkeypatch):
    small_budgets(monkeypatch)

    class Model:
        prompts = []

        async def ainvoke(self, prompt, **kwargs):
            ensure_input_budget(prompt, kwargs["max_tokens"])
            self.prompts.append(prompt)
            return AIMessage(content="用户正在比较 LangGraph 与其他框架，关心 checkpoint。")

    async def run():
        model = Model()
        manager = ContextManager(model)
        history = []
        for index in range(6):
            history.extend([HumanMessage(content=f"第{index}轮：" + "学习LangGraph。" * 12),
                            AIMessage(content="这轮讨论 checkpoint。" * 8)])
        history.append(HumanMessage(content="它保存在哪里？"))
        original = [message.content for message in history]
        prepared = await manager.prepare(history)
        assert model.prompts
        assert prepared["summarized_count"] > 0
        assert "LangGraph" in prepared["conversation_summary"]
        assert message_tokens(prepared["context_messages"]) <= 700
        assert prepared["context_messages"][-1].content == "它保存在哪里？"
        assert [message.content for message in history] == original
        # 每次摘要只读取一个受限批次，而非重新塞入完整历史。
        assert all(text_tokens(prompt[-1].content) < 1100 for prompt in model.prompts)
        calls = len(model.prompts)
        again = await manager.prepare(history, prepared["conversation_summary"], prepared["summarized_count"])
        assert len(model.prompts) == calls
        assert again["summarized_count"] == prepared["summarized_count"]
        history.extend([AIMessage(content="后来的回答" * 100), HumanMessage(content="现在呢？")])
        incremental = await manager.prepare(history, again["conversation_summary"], again["summarized_count"])
        assert incremental["summarized_count"] > again["summarized_count"]
        assert all("第0轮" not in prompt[-1].content for prompt in model.prompts[calls:])

    asyncio.run(run())


def test_invalid_window_configuration_and_final_input_guard():
    from pydantic import ValidationError
    from app.core.config import Settings

    # 只在内存中验证配置，不读取或打印账号密钥。
    values = settings.model_dump()
    values["CHAT_CONTEXT_WINDOW_TOKENS"] = 4096
    with pytest.raises(ValidationError, match="预算之和超过"):
        Settings.model_validate(values)
    with pytest.raises(ContextBudgetError, match="无法放入模型窗口"):
        ensure_input_budget([HumanMessage(content="中" * settings.CHAT_CONTEXT_WINDOW_TOKENS)], 512)


def test_failed_summary_preserves_cursor_and_allows_retry(monkeypatch):
    small_budgets(monkeypatch)

    class Model:
        fail = True

        async def ainvoke(self, prompt, **kwargs):
            if self.fail:
                raise RuntimeError("模拟摘要服务不可用")
            return AIMessage(content="用户之前讨论 LangGraph。")

    async def run():
        model = Model()
        history = [HumanMessage(content="旧消息" * 100), AIMessage(content="旧回答" * 100),
                   HumanMessage(content="继续讲解。")]
        prepared = await ContextManager(model).prepare(history)
        assert prepared["summarized_count"] == 0
        assert message_tokens(prepared["context_messages"]) <= 700
        model.fail = False
        retry = await ContextManager(model).prepare(history, prepared["conversation_summary"], 0)
        assert retry["summarized_count"] > 0

    asyncio.run(run())


def test_latest_question_is_not_silently_truncated(monkeypatch):
    small_budgets(monkeypatch)

    async def run():
        with pytest.raises(ContextBudgetError, match="问题过长"):
            await ContextManager(None).prepare([HumanMessage(content="问题" * 200)])
        assert clip_text("中文内容", 7) == "中文"

    asyncio.run(run())


def test_summary_guides_followup_and_checkpoint_retains_full_history(monkeypatch):
    small_budgets(monkeypatch)
    captured = {}
    query = "LangGraph 的 checkpoint 有哪3种用途？"

    async def search(self, session, user_id, question):
        assert user_id == 7 and question == query
        # 很长的候选片段必须被预算过滤，不能切半后仍作为完整书籍证据。
        return [RetrievedChunk(BookChunk(id="a" * 32, book_id="b" * 32, user_id=7,
                                        title="过长片段", page=1, text="长" * 10000), score=0.99),
                RetrievedChunk(BookChunk(id="c" * 32, book_id="b" * 32, user_id=7,
                                        title="测试教材", page=2, text="Checkpoint 保存图状态。"), score=0.9)]

    class Structured:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, prompt, **kwargs):
            if self.schema is SearchPlan:
                captured["planning"] = prompt
                return SearchPlan(search_query=query)
            assert self.schema is GroundingDecision
            return GroundingDecision(supported=True)

    class Model:
        def with_structured_output(self, schema):
            return Structured(schema)

        async def ainvoke(self, prompt, **kwargs):
            captured.setdefault("summaries", []).append(prompt)
            return AIMessage(content="此前讨论 LangGraph 的 checkpoint，用户关心它的用途。")

        async def astream(self, messages, **kwargs):
            captured["answer"] = messages
            assert kwargs["max_tokens"] == settings.CHAT_MAX_OUTPUT_TOKENS
            ensure_input_budget(messages, kwargs["max_tokens"])
            yield AIMessageChunk(content="它保存图状态 [1]。")

    monkeypatch.setattr(KnowledgeService, "search", search)

    async def run():
        saver = InMemorySaver()
        graph = build_chat_graph(Model(), saver, None)
        config = {"configurable": {"thread_id": "long-dialog"}}
        history = [HumanMessage(content="LangGraph checkpoint" + "早期讨论" * 120),
                   AIMessage(content="旧回答" * 120), HumanMessage(content="接着解释"),
                   AIMessage(content="近期回答")]
        # 模拟没有摘要字段的旧会话，升级后仍可从原 checkpoint 继续聊天。
        await graph.aupdate_state(config, {"messages": history}, as_node="save_memory")
        result = await graph.ainvoke({"messages": [HumanMessage(content="它有哪3种用途？")]},
                                    config=config, context=ChatContext(user_id=7))
        assert result["search_query"] == query
        assert captured["summaries"]
        assert any("此前讨论 LangGraph" in msg.content for msg in captured["planning"])
        assert any("此前讨论 LangGraph" in msg.content for msg in captured["answer"])
        assert "早期讨论" * 120 not in str(captured["answer"])
        assert len(result["messages"]) == len(history) + 2
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "测试教材"
        # 用同一 saver 重建图，摘要与游标仍然存在；另一会话没有这些信息。
        restored = build_chat_graph(Model(), saver, None)
        state = await restored.aget_state(config)
        assert state.values["conversation_summary"] == result["conversation_summary"]
        assert state.values["summarized_count"] == result["summarized_count"]
        other = await restored.aget_state({"configurable": {"thread_id": "another-dialog"}})
        assert not other.values

    asyncio.run(run())
