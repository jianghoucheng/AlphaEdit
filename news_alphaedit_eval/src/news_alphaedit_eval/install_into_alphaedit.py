import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional


def _parse_data_dir(globals_yml: Path) -> Optional[str]:
    # Minimal YAML parsing for a line like: DATA_DIR: "data"
    pat = re.compile(r"^\s*DATA_DIR\s*:\s*['\"]?([^'\"]+)['\"]?\s*$")
    try:
        for line in globals_yml.open("r", encoding="utf-8"):
            m = pat.match(line.strip())
            if m:
                return m.group(1).strip()
    except FileNotFoundError:
        return None
    return None


def _load_json_list(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphaedit_repo", required=True)
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--ds_name", default="mcf")
    ap.add_argument(
        "--no_backup",
        action="store_true",
        help="Skip creating .bak copies of existing dataset files",
    )
    args = ap.parse_args()

    repo = Path(args.alphaedit_repo)
    if not repo.exists():
        raise FileNotFoundError(f"AlphaEdit repo not found: {repo}")

    dataset = _load_json_list(Path(args.dataset_json))

    globals_yml = repo / "globals.yml"
    data_dir_rel = _parse_data_dir(globals_yml) or "data"

    data_dir_candidates = [
        repo / data_dir_rel,
        repo / "data",
    ]

    # Dedup while preserving order.
    seen_base = set()
    data_dir_candidates = [
        p for p in data_dir_candidates if not (p in seen_base or seen_base.add(p))
    ]

    # Common dataset locations used across editing repos.
    targets: List[Path] = []

    for base in data_dir_candidates:
        targets.append(base / f"{args.ds_name}.json")
        targets.append(base / f"{args.ds_name}.jsonl")
        targets.append(base / "dsets" / f"{args.ds_name}.json")
        targets.append(base / "dsets" / f"{args.ds_name}.jsonl")

    targets.append(repo / "dsets" / f"{args.ds_name}.json")
    targets.append(repo / "dsets" / f"{args.ds_name}.jsonl")
    targets.append(repo / "data" / f"{args.ds_name}.json")
    targets.append(repo / "data" / f"{args.ds_name}.jsonl")

    # Dedup while preserving order.
    seen = set()
    uniq_targets: List[Path] = []
    for t in targets:
        if t in seen:
            continue
        seen.add(t)
        uniq_targets.append(t)

    written: List[Path] = []
    for t in uniq_targets:
        # Only write into reasonable paths.
        if repo not in t.parents and t != repo:
            continue

        if not args.no_backup:
            _backup_if_exists(t)

        if t.suffix == ".json":
            _write_json(t, dataset)
        elif t.suffix == ".jsonl":
            _write_jsonl(t, dataset)
        else:
            continue

        written.append(t)

    if not written:
        raise RuntimeError("No dataset files written; target paths resolution failed")

    print("[install_into_alphaedit] Wrote dataset to:")
    for p in written:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
