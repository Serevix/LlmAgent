"""检索评测：对用户标注的书籍问题统计命中、排名、弃答和延迟。"""

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.infrastructure.database.database import engine
from app.models.knowledge_book import KnowledgeBook
from app.services.knowledge_service import KnowledgeService


def score_case(found_ids: list[str], relevant_ids: set[str]) -> tuple[float, float]:
    """根据首个正确来源的排名计算单题 Hit@K 和倒数排名。"""
    if not relevant_ids:
        raise ValueError("评测样本必须标注相关书籍或文本段 ID")
    rank = next((index for index, item in enumerate(found_ids, 1)
                 if item in relevant_ids), None)
    return (1.0 if rank else 0.0, 1.0 / rank if rank else 0.0)


async def evaluate(dataset: Path, user_id: int, limit: int, strategy: str | None = None) -> dict:
    """读取标注集并运行检索，汇总正例命中、负例弃答和耗时。"""
    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError("评测集不能为空")
    results = []
    service = KnowledgeService(retrieval_mode=strategy)
    async with AsyncSession(engine) as session:
        books = list((await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.user_id == user_id,
            KnowledgeBook.status == "ready",
        ))).all())
        by_hash = {book.sha256: book for book in books}
        for record in records:
            question = str(record["question"])
            should_abstain = bool(record.get("should_abstain", False))
            expected_chunks = set(record.get("relevant_chunk_ids", []))
            expected_books = set(record.get("relevant_book_ids", []))
            expected_pages = set()
            for label in record.get("relevant_pages", []):
                book_hash = label["book_sha256"]
                if book_hash not in by_hash:
                    raise ValueError(f"评测书籍未就绪或不属于用户：{book_hash}")
                expected_pages.update(f"{by_hash[book_hash].id}:{page}" for page in label["pages"])
            for book_hash in record.get("relevant_book_sha256", []):
                if book_hash not in by_hash:
                    raise ValueError(f"评测书籍未就绪或不属于用户：{book_hash}")
                expected_books.add(by_hash[book_hash].id)
            if not should_abstain and not expected_chunks and not expected_pages and not expected_books:
                raise ValueError(f"问题未标注相关来源：{question}")
            start = perf_counter()
            hits = await service.search(session, user_id, question, limit)
            elapsed_ms = (perf_counter() - start) * 1000
            # 列举问题可额外返回短书上下文，Hit@K/MRR 仍只统计前 K 项。
            returned_count = len(hits)
            hits = hits[:limit]
            if should_abstain:
                found_ids = []
                expected = set()
                granularity = "negative"
            elif expected_chunks:
                found_ids = [hit.chunk.id for hit in hits]
                expected = expected_chunks
                granularity = "chunk"
            elif expected_pages:
                found_ids = [f"{hit.chunk.book_id}:{hit.chunk.page}" for hit in hits]
                expected = expected_pages
                granularity = "page"
            else:
                found_ids = [hit.chunk.book_id for hit in hits]
                expected = expected_books
                granularity = "book"
            if should_abstain:
                hit, reciprocal_rank = None, None
            else:
                hit, reciprocal_rank = score_case(found_ids, expected)
            results.append({
                "question": question, "granularity": granularity,
                "hit_at_k": hit, "reciprocal_rank": reciprocal_rank,
                "abstain_correct": not hits if should_abstain else None,
                "latency_ms": round(elapsed_ms, 2),
                "returned_count": returned_count,
                "retrieved": [{"book_id": item.chunk.book_id, "chunk_id": item.chunk.id,
                               "page": item.chunk.page, "score": round(item.score, 6),
                               "score_type": item.score_type, "dense_score": item.dense_score,
                               "fusion_score": item.fusion_score, "rerank_score": item.rerank_score,
                               "is_expansion": item.is_expansion}
                              for item in hits],
            })
    latencies = [row["latency_ms"] for row in results]
    positive = [row for row in results if row["granularity"] != "negative"]
    negative = [row for row in results if row["granularity"] == "negative"]
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "strategy": service.retrieval_mode,
        "dense_min_score": settings.RAG_MIN_SCORE,
        "rerank_min_score": settings.RERANK_MIN_SCORE,
        "rerank_model": settings.RERANK_MODEL if service.retrieval_mode == "hybrid_rerank" else None,
        "recall_k": settings.RAG_RECALL_K, "fusion_k": settings.RAG_FUSION_K, "rrf_k": settings.RAG_RRF_K,
        "milvus_collection": service.vectors.last_collection,
        "corpus": [{"sha256": book.sha256, "chunks": book.chunk_count} for book in books],
        "count": len(results), "positive_count": len(positive),
        "negative_count": len(negative), "k": limit,
        "hit_at_k": round(statistics.mean(row["hit_at_k"] for row in positive), 4) if positive else None,
        "mrr": round(statistics.mean(row["reciprocal_rank"] for row in positive), 4) if positive else None,
        "abstain_accuracy": round(statistics.mean(row["abstain_correct"] for row in negative), 4) if negative else None,
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * .95))], 2),
        "cases": results,
    }


def main() -> None:
    """解析命令行参数，运行检索评测并输出 JSON 报告。"""
    parser = argparse.ArgumentParser(description="评估书籍向量检索的 Hit@K、MRR 和延迟")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", choices=["dense", "hybrid", "hybrid_rerank"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k 必须大于 0")
    result = asyncio.run(evaluate(args.dataset, args.user_id, args.k, args.strategy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
