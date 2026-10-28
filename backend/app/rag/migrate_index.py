"""复制旧向量到新的 BM25 集合，校验后创建目标别名；旧集合始终保留。"""

import argparse
import hashlib
import json
import re
import socket
import struct
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.rag.milvus_store import MilvusBookStore


FIELDS = ["id", "vector", "user_id", "book_id", "title", "page", "text"]


def row_digest(row: dict) -> str:
    """对元数据及 float32 向量计算摘要，既检查数量也检查内容和原始 ID。"""
    metadata = {key: row[key] for key in FIELDS if key != "vector"}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode())
    digest.update(struct.pack(f"<{len(row['vector'])}f", *row["vector"]))
    return digest.hexdigest()


def read_batches(client, collection: str):
    """迭代全部原始行，不依赖 offset 的数量限制，始终关闭迭代器。"""
    iterator = client.query_iterator(collection, batch_size=100, filter="", output_fields=FIELDS,
                                     consistency_level="Strong")
    try:
        while batch := iterator.next():
            yield batch
    finally:
        iterator.close()


def snapshot(client, collection: str) -> dict[str, str]:
    """返回 ID 到内容摘要的映射；重复 ID 或无效行拒绝迁移。"""
    result = {}
    for batch in read_batches(client, collection):
        for row in batch:
            if row["id"] in result:
                raise ValueError("读取到重复片段 ID，停止迁移")
            result[row["id"]] = row_digest(row)
    return result


def ensure_backend_stopped() -> None:
    """默认要求本机 8000 后端停止，避免迁移窗口内继续导入或删除书籍。"""
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", 8000)) == 0:
            raise ValueError("请先停止 8000 端口的后端，再迁移索引；其他主机/端口的写入也须手工暂停。")


def migrate(source: str, target: str, *, client=None, verify_only: bool = False) -> dict:
    """复制、双向校验并发布别名，不删除或覆盖任何旧集合。

    source 的每个片段、向量及 ID 原样复制，BM25 稀疏向量由 Milvus 生成，
    不调用 Embedding API。失败时不发布别名，保留暂存集合以便排查。
    本机端口检查不能替代分布式写锁，执行前须暂停所有写入进程。
    """
    if source == target or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,190}", n) for n in (source, target)):
        raise ValueError("来源与目标须为不同的合法 Milvus 名称，长度不超过 191")
    if not verify_only:
        ensure_backend_stopped()
    owned = client is None
    client = client or MilvusBookStore()._client()
    stage = None
    try:
        if not client.has_collection(source):
            raise ValueError("来源集合不存在；新环境可直接导入书籍，无需迁移")
        if verify_only:
            MilvusBookStore(target)._validate_hybrid_schema(client, target)
            before, after = snapshot(client, source), snapshot(client, target)
            if before != after:
                raise ValueError("来源和目标内容不一致；若迁移后已有新导入，这是预期的版本差异，不能覆盖")
            return {"source": source, "target": target, "chunks": len(before), "verified": True}
        if client.has_collection(target):
            raise ValueError("目标集合或别名已存在，不会覆盖；可用 --verify-only 检查，或选择新的目标名称")
        description = client.describe_collection(source)
        dimension = next((int(f.get("params", {}).get("dim", 0)) for f in description["fields"] if f["name"] == "vector"), 0)
        if dimension != settings.EMBEDDING_DIMENSION:
            raise ValueError("来源向量维度与当前配置不一致，不能迁移")
        stage = target + "_build_" + uuid4().hex[:10]
        store = MilvusBookStore(stage)
        store._ensure_collection(client)
        before, sample = {}, None
        for batch in read_batches(client, source):
            for row in batch:
                if row["id"] in before:
                    raise ValueError("来源出现重复 ID")
                before[row["id"]] = row_digest(row)
                sample = sample or row
            client.insert(stage, data=batch)
        client.flush(stage)
        client.load_collection(stage)
        store._validate_hybrid_schema(client, stage)
        if snapshot(client, stage) != before or snapshot(client, source) != before:
            raise ValueError("迁移校验失败或来源在复制时发生变化；没有发布目标别名")
        if sample:
            # 本地烟测使用已保存向量与原文，不发付费 Embedding 或 rerank 请求。
            result = store._hybrid_search(sample["user_id"], sample["vector"], sample["text"][:200], [sample["book_id"]])
            if not result:
                raise ValueError("新集合混合检索未返回候选，停止发布")
        # 发布是最后一步。客户端从这时起才通过目标名称看到完整的新集合。
        client.create_alias(collection_name=stage, alias=target)
        manifest = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
        return {"source": source, "target": target, "physical_collection": stage,
                "chunks": len(before), "verified": True, "snapshot_sha256": manifest,
                "old_collection_preserved": True}
    except Exception:
        if stage:
            print(f"迁移未完成，暂存集合保留为：{stage}；未删除旧集合。")
        raise
    finally:
        if owned:
            client.close()


def main() -> None:
    """解析来源/目标与报告位置；默认使用配置中的旧集合与新集合名称。"""
    parser = argparse.ArgumentParser(description="将旧书籍向量迁移到 Milvus 原生 BM25 混合索引")
    parser.add_argument("--source", default=settings.MILVUS_COLLECTION)
    parser.add_argument("--target", default=settings.MILVUS_HYBRID_COLLECTION)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = migrate(args.source, args.target, verify_only=args.verify_only)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
