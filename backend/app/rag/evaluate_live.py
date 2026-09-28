"""Run the real LangGraph answer path against labeled, locally available books."""

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.chat_graph import ChatContext, build_chat_graph
from app.core.config import settings
from app.infrastructure.database.database import engine
from app.llm.factory import create_chat_model
from app.models.knowledge_book import KnowledgeBook
from app.rag.evaluate_answers import evaluate as score_file


async def evaluate_live(dataset: Path, user_id: int, output: Path) -> dict:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not cases:
        raise ValueError("评测集不能为空")
    async with AsyncSession(engine) as session:
        books = list((await session.exec(select(KnowledgeBook).where(
            KnowledgeBook.user_id == user_id,
            KnowledgeBook.status == "ready",
        ))).all())
    by_hash = {book.sha256: book.id for book in books}
    for case in cases:
        for book_hash in case.get("relevant_book_sha256", []):
            if book_hash not in by_hash:
                raise ValueError(f"评测书籍未就绪或不属于用户：{book_hash}")

    model = create_chat_model(platform="siliconflow", model=settings.CHAT_MODEL)
    graph = build_chat_graph(model, InMemorySaver(), None)
    output.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    with output.open("w", encoding="utf-8") as file:
        for case in cases:
            started = perf_counter()
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=case["question"])]},
                config={"configurable": {"thread_id": uuid4().hex}},
                context=ChatContext(user_id=user_id),
            )
            answer = result["messages"][-1]
            row = {
                "question": case["question"],
                "answer": str(answer.content),
                "sources": [
                    {key: source[key] for key in ("book_id", "page", "chunk_id")}
                    for source in answer.additional_kwargs.get("sources", [])
                ],
                "route": result.get("route"),
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "required_facts": case.get("required_facts", []),
                "relevant_book_ids": [by_hash[item] for item in case.get("relevant_book_sha256", [])],
                "should_abstain": bool(case.get("should_abstain", False)),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            latencies.append(row["latency_ms"])
            print(f"{len(row['sources'])} sources, {row['latency_ms']} ms: {case['question']}", flush=True)
    metrics = score_file(output)
    metrics.update({
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": settings.CHAT_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "min_score": settings.RAG_MIN_SCORE,
        "corpus_sha256": sorted(by_hash),
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "latency_ms_max": round(max(latencies), 2),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="运行真实模型回答并评估引用和关键事实覆盖")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = asyncio.run(evaluate_live(args.dataset, args.user_id, args.output))
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
