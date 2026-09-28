"""Score saved chat answers against a small, human-labeled question set.

Each JSONL row contains answer, sources, required_facts, and relevant_book_ids.
This measures coverage and citation use; human review is still needed for truth.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


CITATION = re.compile(r"\[(\d+)\]")


def score_answer(answer: str, sources: list[dict], required_facts: list[str],
                 relevant_book_ids: set[str], should_abstain: bool = False) -> dict:
    if should_abstain:
        return {
            "fact_coverage": None,
            "valid_citations": None,
            "relevant_source_hit": None,
            "citation_numbers": [],
            "abstain_correct": "当前知识库没有足够依据" in answer and not sources,
        }
    if not required_facts and not relevant_book_ids:
        raise ValueError("每个答案至少需要标注 required_facts 或 relevant_book_ids")
    normalized = answer.casefold()
    fact_coverage = (sum(fact.casefold() in normalized for fact in required_facts)
                     / len(required_facts) if required_facts else None)
    citation_numbers = {int(value) for value in CITATION.findall(answer)}
    valid_citations = bool(citation_numbers) and all(
        1 <= number <= len(sources) for number in citation_numbers
    )
    source_hit = (any(source.get("book_id") in relevant_book_ids for source in sources)
                  if relevant_book_ids else None)
    return {
        "fact_coverage": fact_coverage,
        "valid_citations": valid_citations,
        "relevant_source_hit": source_hit,
        "citation_numbers": sorted(citation_numbers),
        "abstain_correct": None,
    }


def evaluate(dataset: Path) -> dict:
    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError("评测集不能为空")
    cases = []
    for record in records:
        scores = score_answer(
            str(record["answer"]), record.get("sources", []),
            record.get("required_facts", []), set(record.get("relevant_book_ids", [])),
            bool(record.get("should_abstain", False)),
        )
        cases.append({"question": record["question"], **scores})
    facts = [case["fact_coverage"] for case in cases if case["fact_coverage"] is not None]
    sources = [case["relevant_source_hit"] for case in cases
               if case["relevant_source_hit"] is not None]
    citations = [case["valid_citations"] for case in cases if case["valid_citations"] is not None]
    abstentions = [case["abstain_correct"] for case in cases if case["abstain_correct"] is not None]
    return {
        "count": len(cases),
        "fact_coverage": round(statistics.mean(facts), 4) if facts else None,
        "valid_citation_rate": round(statistics.mean(citations), 4) if citations else None,
        "relevant_source_hit_rate": round(statistics.mean(sources), 4) if sources else None,
        "abstain_accuracy": round(statistics.mean(abstentions), 4) if abstentions else None,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估回答的事实覆盖和引用格式")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
