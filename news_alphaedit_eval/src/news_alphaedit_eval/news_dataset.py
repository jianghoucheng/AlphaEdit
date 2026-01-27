import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Ripple:
    category: str
    question: str
    expected_answer: str
    answer_type: str


@dataclass(frozen=True)
class NewsRecord:
    record_id: str
    fact_date: str  # ISO YYYY-MM-DD
    subject: str
    relation: str
    obj: str
    fact: str
    item: Dict[str, Any]
    ripples: List[Ripple]


def find_dataset_file(root: Path) -> Path:
    """Finds dataset.jsonl (preferred) or dataset.json under a root folder."""
    if root.is_file():
        return root

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # Fast path: common names at top-level
    for name in ("dataset.jsonl", "dataset.json"):
        p = root / name
        if p.exists() and p.is_file():
            return p

    # Recursive fallback
    candidates: List[Path] = []
    candidates.extend(root.rglob("dataset.jsonl"))
    candidates.extend(root.rglob("dataset.json"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset.jsonl or dataset.json under: {root}"
        )

    # Prefer jsonl if both exist
    candidates = sorted(candidates)
    for p in candidates:
        if p.name == "dataset.jsonl":
            return p
    return candidates[0]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json_list(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for obj in data:
        yield obj


def _coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def load_news_records(dataset_root_or_file: str) -> List[NewsRecord]:
    path = find_dataset_file(Path(dataset_root_or_file))

    if path.suffix == ".jsonl":
        it = _iter_jsonl(path)
    elif path.suffix == ".json":
        it = _iter_json_list(path)
    else:
        raise ValueError(f"Unsupported dataset file type: {path}")

    out: List[NewsRecord] = []

    for obj in it:
        rec_id = _coerce_str(obj.get("record_id", ""))
        extracted = obj.get("extracted") or {}
        item = obj.get("item") or {}

        fact_date = _coerce_str(extracted.get("fact_date") or item.get("date") or "")
        subject = _coerce_str(extracted.get("subject"))
        relation = _coerce_str(extracted.get("relation"))
        obj_str = _coerce_str(extracted.get("obj"))
        fact = _coerce_str(extracted.get("fact"))

        if not (rec_id and fact_date and subject and relation and obj_str):
            # Keep it strict; silent skipping makes later debugging painful.
            raise ValueError(
                "Missing required fields in record. "
                f"record_id={rec_id!r} fact_date={fact_date!r} "
                f"subject={subject!r} relation={relation!r} obj={obj_str!r}"
            )

        ripples_raw = extracted.get("ripples") or []
        ripples: List[Ripple] = []
        for r in ripples_raw:
            ripples.append(
                Ripple(
                    category=_coerce_str(r.get("category", "")),
                    question=_coerce_str(r.get("question", "")),
                    expected_answer=_coerce_str(r.get("expected_answer", "")),
                    answer_type=_coerce_str(r.get("answer_type", "")),
                )
            )

        out.append(
            NewsRecord(
                record_id=rec_id,
                fact_date=fact_date,
                subject=subject,
                relation=relation,
                obj=obj_str,
                fact=fact,
                item=item,
                ripples=ripples,
            )
        )

    return out


def sort_temporal(records: List[NewsRecord]) -> List[NewsRecord]:
    return sorted(records, key=lambda r: (r.fact_date, r.record_id))


def shuffle_records(records: List[NewsRecord], seed: int) -> List[NewsRecord]:
    import random

    rng = random.Random(seed)
    out = list(records)
    rng.shuffle(out)
    return out


def apply_temporal_tag(text: str, fact_date: str, mode: str) -> str:
    if mode in ("none", "", None):
        return text
    if mode == "bracket_date":
        return f"[{fact_date}] {text}"
    raise ValueError(f"Unknown temporal_tag mode: {mode}")
