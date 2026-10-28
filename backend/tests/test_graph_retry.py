"""运行真实 LangGraph 调度，验证重试次数、错误分类和流式输出边界。"""

import asyncio
import logging
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from openai import APITimeoutError, AuthenticationError, RateLimitError

from app.agents.chat_graph import ChatContext, build_chat_graph
from app.agents.retry import (RETRIEVAL_RETRY_POLICY, RetryableRetrievalError,
                              is_transient_retrieval_error)
from app.rag.embeddings import EmbeddingService
from app.rag.grounding import GroundingDecision
from app.rag.reranker import Reranker, RerankError
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService


class Model:
    """记录生成和核对次数，确认检索重试不会重跑已经完成的其他节点。"""

    def __init__(self, *, fail_stream=False):
        """初始化任务计数，并按测试需要模拟输出中途断开。"""
        self.calls = Counter()
        self.fail_stream = fail_stream

    def with_structured_output(self, schema, **kwargs):
        """为核对提供确定结果，不进行任何外部模型请求。"""
        return self

    async def ainvoke(self, prompt, **kwargs):
        """模拟书籍依据核对；普通问题不会进入改写或记忆提炼。"""
        self.calls['review'] += 1
        return GroundingDecision(supported=True)

    async def astream(self, messages, **kwargs):
        """输出一个片段；可模拟片段已发送后连接断开。"""
        self.calls['answer'] += 1
        yield AIMessageChunk(content='Checkpoint 保存图状态 [1]。')
        if self.fail_stream:
            raise TimeoutError('模拟输出中断')


def evidence():
    """创建属于测试用户的单个证据片段。"""
    return [RetrievedChunk(BookChunk(id='c' * 32, book_id='b' * 32, user_id=7,
                                    title='测试书籍', page=1, text='Checkpoint 保存图状态。'), score=.9)]


async def invoke(graph):
    """用固定身份执行一轮独立问题，便于观察重试后状态是否重复追加。"""
    return await graph.ainvoke({'messages': [HumanMessage(content='Checkpoint 是什么？')]},
                              config={'configurable': {'thread_id': 'retry-test'}},
                              context=ChatContext(user_id=7))


def test_transient_failure_retries_once_and_only_retrieval_node(monkeypatch, caplog):
    """第一次超时第二次成功；最终生成与核对各一次，历史只追加一轮消息。"""
    calls = []

    async def search(self, session, user_id, question):
        """同时检查工具关闭了 Embedding SDK 重试，并在第一次模拟超时。"""
        assert self.embedding_max_retries == 0
        calls.append((user_id, question))
        if len(calls) == 1:
            raise TimeoutError('dummy-secret-must-not-be-logged')
        return evidence()

    monkeypatch.setattr(KnowledgeService, 'search', search)
    model = Model()
    graph = build_chat_graph(model, InMemorySaver(), None)
    caplog.set_level(logging.INFO, logger='langgraph.pregel._retry')
    result = asyncio.run(invoke(graph))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert model.calls == {'answer': 1, 'review': 1}
    assert len(result['messages']) == 2
    assert len(result['messages'][-1].additional_kwargs['sources']) == 1
    assert 'dummy-secret-must-not-be-logged' not in caplog.text
    assert RETRIEVAL_RETRY_POLICY.max_attempts == 2


def test_transient_failure_stops_after_second_attempt(monkeypatch):
    """持续故障最多执行两次，不返回假答案、不继续调用生成或核对模型。"""
    calls = []

    async def search(self, session, user_id, question):
        """每次都抛出可重试的重排故障，观察实际调度次数上限。"""
        calls.append(question)
        raise RerankError('临时不可用', retryable=True)

    monkeypatch.setattr(KnowledgeService, 'search', search)
    model = Model()
    graph = build_chat_graph(model, InMemorySaver(), None)
    with pytest.raises(RetryableRetrievalError, match='书籍检索服务暂时不可用'):
        asyncio.run(invoke(graph))
    assert len(calls) == 2 and not model.calls


@pytest.mark.parametrize('error', [ValueError('参数错误'), RerankError('密钥或格式错误'),
                                  KeyError('程序错误')])
def test_permanent_errors_are_not_retried(monkeypatch, error):
    """参数、程序和永久重排错误立即失败，不能盲目重复请求。"""
    search = AsyncMock(side_effect=error)
    monkeypatch.setattr(KnowledgeService, 'search', search)
    graph = build_chat_graph(Model(), InMemorySaver(), None)
    with pytest.raises(type(error)):
        asyncio.run(invoke(graph))
    assert search.await_count == 1


def test_empty_retrieval_is_not_a_retryable_failure(monkeypatch):
    """查不到知识属于正常空结果，直接弃答，不重复请求模型服务。"""
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(KnowledgeService, 'search', search)
    model = Model()
    result = asyncio.run(invoke(build_chat_graph(model, InMemorySaver(), None)))
    assert search.await_count == 1 and not model.calls
    assert '没有足够依据' in result['messages'][-1].content


def test_partial_answer_is_never_replayed_by_node_retry(monkeypatch):
    """回答已流出时失败，保留错误语义，不能由节点重试重复发送整段文字。"""
    monkeypatch.setattr(KnowledgeService, 'search', AsyncMock(return_value=evidence()))
    model = Model(fail_stream=True)
    graph = build_chat_graph(model, InMemorySaver(), None)

    async def run():
        """收集真实 custom 事件，直到模型中途失败。"""
        tokens = []
        with pytest.raises(TimeoutError):
            async for mode, payload in graph.astream(
                {'messages': [HumanMessage(content='Checkpoint 是什么？')]},
                config={'configurable': {'thread_id': 'stream-failure'}},
                context=ChatContext(user_id=7), stream_mode=['custom', 'updates'],
            ):
                if mode == 'custom' and payload.get('type') == 'token':
                    tokens.append(payload['content'])
        assert tokens == ['Checkpoint 保存图状态 [1]。']
        assert model.calls == {'answer': 1}

    asyncio.run(run())


@pytest.mark.parametrize('status,expected', [(400, False), (401, False), (403, False),
                                           (429, True), (500, True), (503, True)])
def test_rerank_http_status_controls_retryability(monkeypatch, status, expected):
    """通过真实 HTTPX 错误分类保留重试标志，同时隐藏服务商错误内容。"""
    monkeypatch.setattr('app.rag.reranker.settings.RERANK_API_KEY', 'dummy-key')

    def respond(request):
        """返回指定状态码，错误体故意包含不应传播的测试密钥。"""
        return httpx.Response(status, json={'error': 'dummy-key'})

    original = httpx.AsyncClient
    monkeypatch.setattr('app.rag.reranker.httpx.AsyncClient',
                        lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    with pytest.raises(RerankError) as raised:
        asyncio.run(Reranker().rerank('问题', evidence()))
    assert raised.value.retryable is expected
    assert 'dummy-key' not in str(raised.value)


def test_embedding_sdk_retry_is_disabled_only_for_graph_search(monkeypatch):
    """实际走工具与服务层，首个 Embedding 失败后由图重试，导入默认仍保留两次 SDK 重试。"""
    configured = []
    requests = []
    query = 'Checkpoint 是什么？'

    async def create(**kwargs):
        """第一轮向量请求超时，第二轮返回完整维度的合法向量。"""
        requests.append(kwargs)
        if len(requests) == 1:
            raise APITimeoutError(request=httpx.Request('POST', 'https://example.com/v1/embeddings'))
        from app.core.config import settings
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.] * settings.EMBEDDING_DIMENSION)])

    def client(**kwargs):
        """记录 SDK 重试配置，不创建真实远程客户端。"""
        configured.append(kwargs['max_retries'])
        return SimpleNamespace(embeddings=SimpleNamespace(create=create))

    monkeypatch.setattr('app.rag.embeddings.AsyncOpenAI', client)
    monkeypatch.setattr('app.rag.embeddings.settings.GuiJiLiuDong_API_KEY', 'dummy-key')
    monkeypatch.setattr('app.repositories.knowledge_book_repository.KnowledgeBookRepository.list_ready_ids',
                        AsyncMock(return_value=['b' * 32]))
    monkeypatch.setattr('app.rag.milvus_store.MilvusBookStore.hybrid_search', AsyncMock(return_value=evidence()))
    monkeypatch.setattr(Reranker, 'rerank', AsyncMock(return_value=evidence()))
    asyncio.run(invoke(build_chat_graph(Model(), InMemorySaver(), None)))
    assert len(requests) == 2 and all(request['input'] == [query] for request in requests)
    assert configured == [0, 0]
    EmbeddingService()
    assert configured[-1] == 2


def test_sdk_authentication_errors_are_permanent_and_rate_limits_transient():
    """依据 SDK 状态码选择重试，不依赖可能包含敏感内容的错误字符串。"""
    request = httpx.Request('POST', 'https://example.com/v1/embeddings')
    auth = AuthenticationError('密钥错误', response=httpx.Response(401, request=request), body=None)
    rate = RateLimitError('限流', response=httpx.Response(429, request=request), body=None)
    assert not is_transient_retrieval_error(auth)
    assert is_transient_retrieval_error(rate)
