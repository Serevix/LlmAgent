"""向量存储层：建立集合、写入切片、按用户检索及删除书籍向量。"""

import asyncio
import json
import math

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from app.core.config import settings
from app.rag.types import BookChunk, RetrievedChunk


class MilvusBookStore:
    """负责书籍向量及切片存取；书籍所有权与导入状态仍以 PostgreSQL 为准。"""

    def __init__(self, collection_name: str | None = None):
        """默认使用新混合集合/别名；迁移和隔离测试可显式指定目标集合。"""
        self.collection = collection_name or settings.MILVUS_HYBRID_COLLECTION
        self.last_collection = self.collection

    def _client(self) -> MilvusClient:
        """连接配置的 Milvus 地址；连接失败向业务层传播异常。"""
        return MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN or "",
            timeout=10,
        )

    def _ping(self) -> None:
        """同步检查集合列表，结束时关闭客户端。"""
        client = self._client()
        try:
            client.list_collections(timeout=10)
        finally:
            client.close()

    async def ping(self) -> None:
        """在线程中执行同步健康检查，避免阻塞事件循环。"""
        await asyncio.to_thread(self._ping)

    def _ensure_collection(self, client: MilvusClient) -> None:
        """创建稠密向量、中文分词及 BM25 稀疏索引，不修改旧集合结构。"""
        name = self.collection
        if client.has_collection(name):
            self._validate_hybrid_schema(client, name)
            return
        if name == settings.MILVUS_HYBRID_COLLECTION and client.has_collection(settings.MILVUS_COLLECTION):
            count = client.query(settings.MILVUS_COLLECTION, filter="", output_fields=["count(*)"],
                                 consistency_level="Strong")[0]["count(*)"]
            if count:
                raise ValueError("检测到旧知识库，请先停止后端并执行 python -m app.rag.migrate_index，再导入文件。")

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        # 主键由切片生成；保存原文和位置，检索结果才能支持来源引用。
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)
        schema.add_field("user_id", DataType.INT64)
        schema.add_field("book_id", DataType.VARCHAR, max_length=32)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("page", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=4096, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["lowercase"]})
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(name="book_bm25", input_field_names=["text"],
                                     output_field_names=["sparse"], function_type=FunctionType.BM25))
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        client.create_collection(name, schema=schema, index_params=indexes)
        client.load_collection(name)

    def _validate_hybrid_schema(self, client: MilvusClient, name: str) -> None:
        """已有集合须具备匹配的稠密维度和 BM25 函数，拒绝误连旧索引。"""
        description = client.describe_collection(name)
        fields = {field["name"]: field for field in description["fields"]}
        dimension = int(fields.get("vector", {}).get("params", {}).get("dim", 0))
        if dimension != settings.EMBEDDING_DIMENSION or "sparse" not in fields:
            raise ValueError("混合索引字段或向量维度不匹配，请迁移到新的集合名称。")
        if not any("sparse" in fn.get("output_field_names", []) for fn in description.get("functions", [])):
            raise ValueError("集合缺少 BM25 函数，请执行混合索引迁移。")

    @staticmethod
    def _filter(user_id: int, book_ids: list[str]) -> str:
        """两路检索使用同一权限过滤；用 JSON 编码书籍 ID，避免拼接过滤表达式。"""
        return f"user_id == {int(user_id)} and book_id in {json.dumps(book_ids)}"

    @staticmethod
    def _chunk(row: dict) -> BookChunk:
        """将 Milvus 行转换为带原始 ID 的片段，查询和迁移均不重新生成编号。"""
        return BookChunk(id=str(row["id"]), book_id=row["book_id"], user_id=int(row["user_id"]),
                         title=row["title"], page=int(row["page"]), text=row["text"])

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        """为融合候选单独计算余弦信号，仅用于短书强命中判断，不替代 RRF 分数。"""
        if len(left) != len(right):
            raise ValueError("查询向量与存储向量维度不一致")
        denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
        return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0

    def _insert(self, chunks: list[BookChunk], vectors: list[list[float]]) -> None:
        """每批写入一百个切片及其对应向量。"""
        client = self._client()
        try:
            self._ensure_collection(client)
            for start in range(0, len(chunks), 100):
                rows = [
                    {"id": chunk.id, "vector": vector, "user_id": chunk.user_id,
                     "book_id": chunk.book_id, "title": chunk.title, "page": chunk.page,
                     "text": chunk.text}
                    for chunk, vector in zip(chunks[start:start + 100], vectors[start:start + 100])
                ]
                client.insert(self.collection, data=rows)
        finally:
            client.close()

    async def insert(self, chunks: list[BookChunk], vectors: list[list[float]]) -> None:
        """先检查切片与向量数量相等，再在线程中写入。"""
        if len(chunks) != len(vectors):
            raise ValueError("文本段和向量数量不一致")
        await asyncio.to_thread(self._insert, chunks, vectors)

    def _search(self, user_id: int, vector: list[float], limit: int,
                book_ids: list[str]) -> list[RetrievedChunk]:
        """按用户和允许的书籍过滤向量检索，返回带分数的切片。"""
        if not book_ids:
            return []
        client = self._client()
        try:
            name = self.collection
            # 纯向量基线可在迁移前读取旧集合；混合检索没有这种回退。
            if not client.has_collection(name):
                name = settings.MILVUS_COLLECTION
            if not client.has_collection(name):
                return []
            self.last_collection = name
            hits = client.search(
                name, data=[vector], anns_field="vector",
                filter=self._filter(user_id, book_ids),
                limit=limit, output_fields=["user_id", "book_id", "title", "page", "text"],
                search_params={"metric_type": "COSINE", "params": {}},
                consistency_level="Strong",
            )[0]
            return [
                RetrievedChunk(
                    chunk=BookChunk(id=str(hit["id"]), book_id=hit["entity"]["book_id"],
                                    user_id=int(hit["entity"]["user_id"]),
                                    title=hit["entity"]["title"], page=int(hit["entity"]["page"]),
                                    text=hit["entity"]["text"]),
                    score=float(hit["distance"]), dense_score=float(hit["distance"]),
                )
                for hit in hits
            ]
        finally:
            client.close()

    async def search(self, user_id: int, vector: list[float], limit: int = 5,
                     book_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """异步入口，在线程中调用同步检索；默认最多取五个候选。"""
        return await asyncio.to_thread(self._search, user_id, vector, limit, book_ids or [])

    def _hybrid_search(self, user_id: int, vector: list[float], question: str,
                       book_ids: list[str]) -> list[RetrievedChunk]:
        """Milvus 原生两路召回与 RRF 融合；不将余弦阈值套到融合分数上。"""
        if not book_ids:
            return []
        client = self._client()
        try:
            if not client.has_collection(self.collection):
                raise ValueError("混合知识库尚未就绪，请停止后端并执行 python -m app.rag.migrate_index。")
            self._validate_hybrid_schema(client, self.collection)
            expression = self._filter(user_id, book_ids)
            requests = [
                AnnSearchRequest(data=[vector], anns_field="vector", param={"metric_type": "COSINE"},
                                 limit=settings.RAG_RECALL_K, expr=expression),
                AnnSearchRequest(data=[question], anns_field="sparse", param={"metric_type": "BM25"},
                                 limit=settings.RAG_RECALL_K, expr=expression),
            ]
            hits = client.hybrid_search(self.collection, reqs=requests, ranker=RRFRanker(k=settings.RAG_RRF_K),
                limit=settings.RAG_FUSION_K, output_fields=["user_id", "book_id", "title", "page", "text", "vector"],
                consistency_level="Strong")[0]
            result, seen = [], set()
            for hit in hits:
                chunk = self._chunk({**hit["entity"], "id": hit["id"]})
                if chunk.user_id != user_id or chunk.book_id not in book_ids:
                    raise ValueError("检索结果归属不匹配")
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                fusion = float(hit["distance"])
                result.append(RetrievedChunk(chunk, fusion, score_type="rrf", fusion_score=fusion,
                    dense_score=self.cosine(vector, hit["entity"]["vector"])))
            return result
        finally:
            client.close()

    async def hybrid_search(self, user_id: int, vector: list[float], question: str,
                            book_ids: list[str]) -> list[RetrievedChunk]:
        """将同步混合检索放入线程，返回已融合且去重的候选片段。"""
        return await asyncio.to_thread(self._hybrid_search, user_id, vector, question, book_ids)

    def _book_chunks(self, user_id: int, book_id: str, vector: list[float]) -> list[RetrievedChunk]:
        """补取短书原文；单次限制 25 条，业务层校验数量与书籍状态后决定是否采用。"""
        client = self._client()
        try:
            rows = client.query(self.collection, filter=self._filter(user_id, [book_id]), limit=25,
                output_fields=["id", "user_id", "book_id", "title", "page", "text", "vector"], consistency_level="Strong")
            return [RetrievedChunk(self._chunk(row), 0.0, score_type="expansion",
                                   dense_score=self.cosine(vector, row["vector"]), is_expansion=True) for row in rows]
        finally:
            client.close()

    async def book_chunks(self, user_id: int, book_id: str, vector: list[float]) -> list[RetrievedChunk]:
        """异步补取入口；原文仍限制在当前用户的一本书内。"""
        return await asyncio.to_thread(self._book_chunks, user_id, book_id, vector)

    def _delete_book(self, user_id: int, book_id: str) -> None:
        """同时按用户与书籍过滤删除，避免误删其他用户的数据。"""
        client = self._client()
        try:
            if client.has_collection(self.collection):
                client.delete(self.collection, filter=self._filter(user_id, [book_id]))
        finally:
            client.close()

    async def delete_book(self, user_id: int, book_id: str) -> None:
        """异步删除入口，将同步 Milvus 操作放入线程。"""
        await asyncio.to_thread(self._delete_book, user_id, book_id)
