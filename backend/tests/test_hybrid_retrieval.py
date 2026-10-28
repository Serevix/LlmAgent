"""验证混合召回、独立分数、重排协议、扩展与迁移发布；默认不访问远程服务。"""

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.rag.milvus_store import MilvusBookStore
from app.rag.reranker import Reranker, RerankError
from app.rag.types import BookChunk, RetrievedChunk
from app.services.knowledge_service import KnowledgeService


def hit(index, *, dense=.2, score=.02, user=7, expansion=False):
    """生成有独立稠密与融合分数的候选，避免测试混淆量纲。"""
    chunk = BookChunk(id=str(index)*32, book_id="b"*32, user_id=user, title="测试书籍", page=index, text=f"证据{index}")
    return RetrievedChunk(chunk, score, score_type="rrf", dense_score=dense, fusion_score=score, is_expansion=expansion)


def test_native_rrf_sends_both_permission_filters_and_keeps_score_types(monkeypatch):
    row = {"id": "1"*32, "book_id": "b"*32, "user_id": 7, "title": "书籍", "page": 1, "text": "原文", "vector": [0., 1.]}
    class Client:
        closed = False

        def has_collection(self, name):
            return True

        def describe_collection(self, name):
            return {"fields": [{"name": "vector", "params": {"dim": 2}}, {"name": "sparse"}],
                    "functions": [{"output_field_names": ["sparse"]}]}

        def hybrid_search(self, name, reqs, ranker, **kwargs):
            assert [r.anns_field for r in reqs] == ["vector", "sparse"]
            assert reqs[0].data == [[1., 0.]] and reqs[1].data == ["RunnableConfig"]
            assert all('user_id == 7' in r.expr and '"' + "b"*32 + '"' in r.expr for r in reqs)
            assert ranker.dict() == {"strategy": "rrf", "params": {"k": settings.RAG_RRF_K}}
            assert kwargs["limit"] == settings.RAG_FUSION_K
            return [[{"id": row["id"], "entity": row, "distance": .032},
                     {"id": row["id"], "entity": row, "distance": .032}]]

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 2)
    store = MilvusBookStore("test_hybrid")
    monkeypatch.setattr(store, "_client", lambda: client)
    found = store._hybrid_search(7, [1., 0.], "RunnableConfig", ["b"*32])
    assert len(found) == 1 and client.closed
    assert found[0].dense_score == 0 and found[0].fusion_score == .032
    assert found[0].score_type == "rrf" and found[0].rerank_score is None


@pytest.mark.parametrize("malformed", ["duplicate", "missing", "nan", "wrong_index", "provider_error", None])
def test_rerank_matches_response_indices_validates_and_hides_secrets(monkeypatch, malformed, caplog):
    secret = "dummy-rerank-secret"
    monkeypatch.setattr(settings, "RERANK_API_KEY", secret)
    monkeypatch.setattr(settings, "RERANK_BASE_URL", "https://example.com/v1")
    candidates = [hit(1), hit(2)]

    def respond(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://example.com/v1/rerank"
        assert request.headers["authorization"] == f"Bearer {secret}"
        assert body["documents"] == [c.chunk.text for c in candidates] and body["top_n"] == 2
        if malformed == "provider_error":
            return httpx.Response(401, json={"message": secret})
        results = [{"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .1}]
        if malformed == "duplicate": results[1]["index"] = 1
        if malformed == "missing": results.pop()
        if malformed == "nan": results[0]["relevance_score"] = float('nan')
        if malformed == "wrong_index": results[0]["index"] = 100
        if malformed == "nan":
            return httpx.Response(200, content=json.dumps({"results": results}).encode())
        return httpx.Response(200, json={"results": results})

    original = httpx.AsyncClient
    monkeypatch.setattr("app.rag.reranker.httpx.AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    if malformed:
        with pytest.raises(RerankError) as error:
            asyncio.run(Reranker().rerank("问题", candidates))
        assert secret not in str(error.value) and secret not in caplog.text
    else:
        result = asyncio.run(Reranker().rerank("问题", candidates))
        assert [h.chunk.id for h in result] == [candidates[1].chunk.id, candidates[0].chunk.id]
        assert result[0].score == .9 and result[0].score_type == "rerank"
        assert result[0].fusion_score == candidates[1].fusion_score


def make_service(monkeypatch, candidates):
    """用可观察的依赖替换外部检索，不把控制流验证伪装成真实模型效果。"""
    embedding = SimpleNamespace(embed=AsyncMock(return_value=[[1., 0.]]))
    monkeypatch.setattr("app.services.knowledge_service.EmbeddingService", lambda: embedding)
    service = KnowledgeService("hybrid_rerank")
    service.books = SimpleNamespace(list_ready_ids=AsyncMock(return_value=["b"*32]), get_owned=AsyncMock())
    service.vectors = SimpleNamespace(hybrid_search=AsyncMock(return_value=candidates), book_chunks=AsyncMock())
    return service


def test_keyword_only_candidate_reaches_rerank_and_uses_its_own_threshold(monkeypatch):
    candidates = [hit(1, dense=.9), hit(2, dense=.1)]
    service = make_service(monkeypatch, candidates)
    ranked = [replace(candidates[1], score=.95, score_type="rerank", rerank_score=.95),
              replace(candidates[0], score=.01, score_type="rerank", rerank_score=.01)]
    service.reranker.rerank = AsyncMock(return_value=ranked)
    result = asyncio.run(service.search(None, 7, "RunnableConfig 是什么？"))
    assert [r.chunk.id for r in result] == [candidates[1].chunk.id]
    service.reranker.rerank.assert_awaited_once_with("RunnableConfig 是什么？", candidates)


def test_short_book_is_merged_deduplicated_then_reranked_without_losing_context(monkeypatch):
    candidates = [hit(1, dense=.9), hit(2)]
    service = make_service(monkeypatch, candidates)
    service.books.get_owned.return_value = SimpleNamespace(id="b"*32, status="ready", chunk_count=4)
    service.vectors.book_chunks.return_value = [hit(i) for i in range(1, 5)]

    async def rank(question, merged):
        assert len(merged) == 4 and len({h.chunk.id for h in merged}) == 4
        return [replace(h, score=.9 if h.chunk.page == 1 else .01, score_type="rerank",
                        rerank_score=.9 if h.chunk.page == 1 else .01) for h in merged]

    service.reranker.rerank = rank
    result = asyncio.run(service.search(None, 7, "4个术语分别是什么？", limit=1))
    assert [h.chunk.page for h in result] == [1, 2, 3, 4]
    assert all(h.is_expansion for h in result)
    assert result[1].score < settings.RERANK_MIN_SCORE  # 明确标记为同书上下文。


def test_rerank_failure_does_not_silently_return_rrf_candidates(monkeypatch):
    service = make_service(monkeypatch, [hit(1)])
    service.reranker.rerank = AsyncMock(side_effect=RerankError("重排不可用"))
    with pytest.raises(RerankError):
        asyncio.run(service.search(None, 7, "普通问题"))


@pytest.mark.parametrize("corrupt", [False, True])
def test_migration_only_publishes_after_content_verification(monkeypatch, corrupt):
    from app.rag.migrate_index import migrate
    row = {"id": "1"*32, "book_id": "b"*32, "user_id": 7, "title": "书", "page": 1,
           "text": "原文", "vector": [1., 0.]}

    class Iterator:
        def __init__(self, rows): self.rows = copy.deepcopy(rows)
        def next(self):
            result, self.rows = self.rows, []
            return result
        def close(self): pass

    class Client:
        def __init__(self): self.rows = {"source": [row]}; self.aliases = {}
        def has_collection(self, name): return name in self.rows or name in self.aliases
        def describe_collection(self, name): return {"fields": [{"name": "vector", "params": {"dim": 2}}]}
        def query_iterator(self, name, **kwargs): return Iterator(self.rows[name])
        def insert(self, name, data):
            self.rows[name].extend(copy.deepcopy(data))
            if corrupt: self.rows[name][0]["text"] = "损坏"
        def flush(self, name): pass
        def load_collection(self, name): pass
        def create_alias(self, collection_name, alias): self.aliases[alias] = collection_name

    monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 2)
    monkeypatch.setattr("app.rag.migrate_index.ensure_backend_stopped", lambda: None)
    monkeypatch.setattr(MilvusBookStore, "_ensure_collection", lambda self, client: client.rows.setdefault(self.collection, []))
    monkeypatch.setattr(MilvusBookStore, "_validate_hybrid_schema", lambda *args: None)
    monkeypatch.setattr(MilvusBookStore, "_hybrid_search", lambda *args: [hit(1)])
    client = Client()
    if corrupt:
        with pytest.raises(ValueError, match="校验失败"):
            migrate("source", "target", client=client)
        assert not client.aliases
    else:
        result = migrate("source", "target", client=client)
        assert result["verified"] and result["chunks"] == 1
        assert client.rows[client.aliases["target"]] == [row]
        with pytest.raises(ValueError, match="已存在"):
            migrate("source", "target", client=client)
    assert client.rows["source"] == [row]
