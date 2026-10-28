"""实测问答的首字、最终答案、整轮耗时及各节点聊天模型调用次数。"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agents.chat_graph import ChatContext, build_chat_graph
from app.core.config import settings
from app.llm.factory import create_chat_model


class CallCounter(BaseCallbackHandler):
    """按 LangGraph 节点统计聊天请求；不记录提示词、凭据或用户回答正文。"""

    def __init__(self):
        """为本轮请求创建独立计数器，避免把上一轮的请求算进来。"""
        self.calls = Counter()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """每次聊天模型请求开始时计数；Embedding、rerank 不属于此回调。"""
        self.calls[(kwargs.get("metadata") or {}).get("langgraph_node", "unknown")] += 1


async def benchmark(questions: list[str], user_id: int) -> dict:
    """同一临时会话连续提问，使用真实书籍检索，但不写业务历史或长期记忆。"""
    model = create_chat_model("siliconflow", settings.CHAT_MODEL, temperature=0)
    graph = build_chat_graph(model, InMemorySaver(), InMemoryStore())
    thread_id = uuid4().hex
    rows = []
    for question in questions:
        counter = CallCounter()
        started = perf_counter()
        first_token = final_answer = None
        source_count = 0
        refused = None
        async for mode, payload in graph.astream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}, "callbacks": [counter]},
            context=ChatContext(user_id=user_id), stream_mode=["custom", "updates"],
        ):
            if mode == "custom" and payload.get("type") == "token" and payload.get("content"):
                if first_token is None:
                    first_token = round((perf_counter() - started) * 1000, 2)
            elif mode == "updates":
                for node in ("grounded_answer", "no_evidence"):
                    update = payload.get(node)
                    if update and update.get("messages"):
                        answer = update["messages"][-1]
                        final_answer = round((perf_counter() - started) * 1000, 2)
                        source_count = len(answer.additional_kwargs.get("sources", []))
                        refused = "没有足够依据" in str(answer.content)
        rows.append({"question": question, "first_token_ms": first_token,
                     "final_answer_ms": final_answer,
                     "done_ms": round((perf_counter() - started) * 1000, 2),
                     "chat_calls": dict(counter.calls), "chat_call_count": sum(counter.calls.values()),
                     "source_count": source_count, "refused": refused})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return {"measured_at": datetime.now(timezone.utc).isoformat(),
            "chat_model": settings.CHAT_MODEL, "temperature": 0,
            "query_rewrite_mode": settings.CHAT_QUERY_REWRITE_MODE,
            "memory_mode": settings.CHAT_MEMORY_MODE,
            "retrieval_mode": settings.RAG_RETRIEVAL_MODE,
            "note": "小样本连续请求，受网络与服务商负载影响；首字是待核对草稿。统计的是聊天模型调用，不包含 Embedding、rerank 或 SDK 内部 HTTP 重试次数。",
            "cases": rows}


def main():
    """从参数接收公开测试问题，并将实测结果写入指定 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(benchmark(args.question, args.user_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
