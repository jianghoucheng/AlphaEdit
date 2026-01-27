import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .news_dataset import (
    NewsRecord,
    apply_temporal_tag,
    load_news_records,
    shuffle_records,
    sort_temporal,
)


def _prompt_template(subject: str, relation: str) -> str:
    rel = relation.strip()
    if not rel:
        raise ValueError("Empty relation")
    # CounterFact prompt template format: must contain '{}' placeholder.
    return "{} " + rel + " "


def _to_alphaedit_record(case_id: int, rec: NewsRecord, temporal_tag: str) -> Dict[str, Any]:
    prompt = _prompt_template(rec.subject, rec.relation)
    prompt = apply_temporal_tag(prompt, rec.fact_date, temporal_tag)

    # Ripple questions are carried over as paraphrase prompts.
    # AlphaEdit's built-in evaluation assumes these should elicit target_new.
    paraphrase_prompts = [
        apply_temporal_tag(r.question, rec.fact_date, temporal_tag)
        for r in rec.ripples
        if r.question
    ]

    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": prompt,
            "relation_id": 0,
            "subject": rec.subject,
            # Keep CounterFact-style structure.
            "target_new": {"str": rec.obj},
            "target_true": {"str": rec.obj},
        },
        "paraphrase_prompts": paraphrase_prompts,
        "neighborhood_prompts": [],
        "generation_prompts": [],
        # Extra metadata (ignored by AlphaEdit, useful for debugging).
        "news": {
            "record_id": rec.record_id,
            "fact_date": rec.fact_date,
            "fact": rec.fact,
            "source": rec.item.get("source_name"),
            "url": rec.item.get("url"),
            "raw_text": rec.item.get("text"),
        },
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_meta_jsonl(path: Path, records: List[NewsRecord], temporal_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            obj = {
                "case_id": i,
                "record_id": rec.record_id,
                "fact_date": rec.fact_date,
                "subject": rec.subject,
                "relation": rec.relation,
                "obj": rec.obj,
                "fact": rec.fact,
                "temporal_tag": temporal_tag,
                "ripples": [
                    {
                        "category": r.category,
                        "question": r.question,
                        "expected_answer": r.expected_answer,
                        "answer_type": r.answer_type,
                    }
                    for r in rec.ripples
                ],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news_dataset_root", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_meta_jsonl", required=True)
    p.add_argument(
        "--order",
        choices=["temporal", "random", "input"],
        default="temporal",
        help="Record ordering before writing the AlphaEdit dataset",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--temporal_tag",
        choices=["none", "bracket_date"],
        default="none",
        help="Optional date prefix for prompts/questions",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of records (0 => no limit)",
    )
    args = p.parse_args()

    records = load_news_records(args.news_dataset_root)

    if args.order == "temporal":
        records = sort_temporal(records)
    elif args.order == "random":
        records = shuffle_records(records, seed=args.seed)

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    alpha_records: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        alpha_records.append(_to_alphaedit_record(i, rec, temporal_tag=args.temporal_tag))

    _write_json(Path(args.out_json), alpha_records)
    _write_meta_jsonl(Path(args.out_meta_jsonl), records, temporal_tag=args.temporal_tag)


if __name__ == "__main__":
    main()
