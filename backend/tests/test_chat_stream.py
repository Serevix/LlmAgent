"""验证实时输出、最终答案替换以及聊天历史中的证据约束。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.chat_graph import NO_EVIDENCE_ANSWER, build_chat_graph
from app.api.v1.chat import chat_stream
from app.rag.grounding import GroundingDecision
from app.rag.types import BookChunk, RetrievedChunk
from app.schemas.chat import ChatRequest
from app.services.conversation_service import ConversationService
from app.services.knowledge_service import KnowledgeService


def make_graph(monkeypatch, model, *, has_evidence=True):
    async def search(self, session, user_id, question):
        assert user_id == 7
        assert question == "RAG 的 checkpoint 是什么？"
        if not has_evidence:
            return []
        return [RetrievedChunk(
            BookChunk(id="c" * 32, book_id="b" * 32, user_id=7,
                      title="测试书籍", page=1, text="Checkpoint 保存图状态。"),
            score=0.9,
        )]

    monkeypatch.setattr(KnowledgeService, "search", search)
    return build_chat_graph(model, InMemorySaver(), None)


async def make_response(monkeypatch, graph):
    async def get_owned(self, session, conversation_id, user_id):
        assert user_id == 7
        return SimpleNamespace(user_id=7, thread_id="stream-test")

    async def touch(self, session, conversation):
        pass

    monkeypatch.setattr(ConversationService, "get_owned", get_owned)
    monkeypatch.setattr(ConversationService, "touch", touch)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chat_graph=graph)))
    return await chat_stream(
        ChatRequest(conversation_id=1, message="RAG 的 checkpoint 是什么？"),
        request,
        session=None,
        current_user=SimpleNamespace(id=7),
    )


def decode_event(frame):
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[6:].strip())


def test_rerank_service_error_is_reported_instead_of_abstention(monkeypatch):
    """服务故障应返回中文 error，不得伪装成无依据的最终答案。"""
    from app.rag.reranker import RerankError

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

    graph = make_graph(monkeypatch, Model())

    async def broken_search(self, session, user_id, question):
        raise RerankError("重排服务调用失败，请检查接口。")

    monkeypatch.setattr(KnowledgeService, "search", broken_search)

    async def run():
        response = await make_response(monkeypatch, graph)
        events = [decode_event(frame) async for frame in response.body_iterator]
        assert events[-1] == {"type": "error", "content": "重排服务调用失败，请检查接口。"}
        assert not any(event["type"] in ("answer", "done") for event in events)

    asyncio.run(run())


def test_retrieval_retry_exhaustion_returns_error_without_final_answer(monkeypatch):
    """检索持续超时只尝试两次，SSE 返回安全中文错误，不发送答案或成功结束。"""
    class Model:
        def with_structured_output(self, schema, **kwargs):
            """检索失败前仅装配模型，测试中不应发生任何推理调用。"""
            return self

    graph = make_graph(monkeypatch, Model())
    calls = []

    async def broken_search(self, session, user_id, question):
        """每次读检索服务都超时，用于验证节点重试耗尽后的接口表现。"""
        calls.append(question)
        raise TimeoutError("模拟临时检索故障")

    monkeypatch.setattr(KnowledgeService, "search", broken_search)

    async def run():
        """消费完整 SSE，检查只有错误提示而没有虚假的成功结果。"""
        response = await make_response(monkeypatch, graph)
        events = [decode_event(frame) async for frame in response.body_iterator]
        assert len(calls) == 2
        assert events[-1] == {"type": "error", "content": "书籍检索服务暂时不可用，请稍后重试。"}
        assert not any(event["type"] in ("answer", "done") for event in events)

    asyncio.run(run())


@pytest.mark.parametrize("review", [True, False, "error"])
def test_tokens_arrive_before_completion_and_final_answer_is_saved(monkeypatch, review):
    async def run():
        continue_generation = asyncio.Event()

        class Model:
            generation_finished = False
            review_started = False

            def with_structured_output(self, schema):
                return self

            async def ainvoke(self, prompt, **kwargs):
                assert self.generation_finished
                self.review_started = True
                if review == "error":
                    raise RuntimeError("模拟核对服务异常")
                return GroundingDecision(supported=review)

            async def astream(self, messages, **kwargs):
                assert isinstance(messages[-1], HumanMessage)
                yield AIMessageChunk(content="Checkpoint ")
                # 消费者先收到第一段，才允许模型生成后续文字。
                await continue_generation.wait()
                yield AIMessageChunk(content=[
                    {"type": "text", "text": "保存图状态 [1]。"},
                    {"type": "reasoning", "reasoning": "内部核对信息"},
                ])
                self.generation_finished = True

        model = Model()
        graph = make_graph(monkeypatch, model)
        response = await make_response(monkeypatch, graph)
        iterator = response.body_iterator
        events = []

        async def read_first_token():
            while True:
                event = decode_event(await anext(iterator))
                events.append(event)
                if event["type"] == "token":
                    return event

        first = await asyncio.wait_for(read_first_token(), timeout=5)
        assert first["content"] == "Checkpoint "
        assert not model.generation_finished
        assert not model.review_started
        assert events[0]["type"] == "sources"
        assert events[0]["pending"] is True
        assert len(events[0]["sources"]) == 1
        continue_generation.set()
        events.extend([decode_event(frame) async for frame in iterator])

        draft = "".join(event["content"] for event in events if event["type"] == "token")
        assert draft == "Checkpoint 保存图状态 [1]。"
        assert "内部核对信息" not in json.dumps(events, ensure_ascii=False)
        final = next(event for event in events if event["type"] == "answer")
        source_events = [event for event in events if event["type"] == "sources"]
        assert len(source_events) == 2
        assert source_events[-1].get("pending", False) is False
        sources = source_events[-1]["sources"]
        assert events[-1]["type"] == "done"
        assert model.review_started

        expected = draft if review is True else NO_EVIDENCE_ANSWER
        assert final["content"] == expected
        assert bool(sources) is (review is True)
        state = await graph.aget_state({"configurable": {"thread_id": "stream-test"}})
        assert len(state.values["messages"]) == 2
        answer = state.values["messages"][-1]
        assert answer.content == expected
        assert answer.additional_kwargs["sources"] == sources

    asyncio.run(run())


def test_no_evidence_returns_final_refusal_without_calling_model(monkeypatch):
    class Model:
        def with_structured_output(self, schema):
            return self

        async def astream(self, messages, **kwargs):
            raise AssertionError("没有证据时不应调用回答模型")
            yield

    async def run():
        graph = make_graph(monkeypatch, Model(), has_evidence=False)
        response = await make_response(monkeypatch, graph)
        events = [decode_event(frame) async for frame in response.body_iterator]
        assert events == [
            {"type": "answer", "content": NO_EVIDENCE_ANSWER},
            {"type": "sources", "sources": []},
            {"type": "done"},
        ]

    asyncio.run(run())


@pytest.mark.parametrize("partial", [False, True])
def test_generation_failure_has_error_without_final_answer(monkeypatch, partial):
    class Model:
        def with_structured_output(self, schema):
            return self

        async def astream(self, messages, **kwargs):
            if partial:
                yield AIMessageChunk(content="未完成的草稿")
                raise RuntimeError("模拟模型连接中断")
            # 空流也应作为错误处理，不能发送成功结束事件。

    async def run():
        graph = make_graph(monkeypatch, Model())
        response = await make_response(monkeypatch, graph)
        events = [decode_event(frame) async for frame in response.body_iterator]
        assert events[-1]["type"] == "error"
        assert events[0]["type"] == "sources" and events[0]["pending"] is True
        assert not any(event["type"] in ("answer", "done") for event in events)
        state = await graph.aget_state({"configurable": {"thread_id": "stream-test"}})
        assert all(isinstance(message, HumanMessage) for message in state.values["messages"])

    asyncio.run(run())
