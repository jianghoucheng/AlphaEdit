import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def _env_with_pythonpath(alphaedit_repo: Path) -> Dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = f"{alphaedit_repo}:{existing}"
    else:
        env["PYTHONPATH"] = str(alphaedit_repo)

    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _list_run_dirs(results_root: Path) -> List[Path]:
    if not results_root.exists():
        return []
    out: List[Path] = []
    for p in results_root.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            out.append(p)
    return out


def _newest_run_dir(results_root: Path, started_ts: float) -> Optional[Path]:
    runs = _list_run_dirs(results_root)
    if not runs:
        return None

    # Prefer runs created/updated after the command started.
    post = [p for p in runs if p.stat().st_mtime >= started_ts - 2]
    cands = post if post else runs
    return max(cands, key=lambda p: p.stat().st_mtime)


def _stream_process(proc: subprocess.Popen, log_path: Optional[Path]) -> int:
    log_f = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w", encoding="utf-8")

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_f is not None:
                log_f.write(line)
        return proc.wait()
    finally:
        if log_f is not None:
            log_f.close()


def _copy_or_move(src: Path, dst: Path, move: bool) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphaedit_repo", required=True)
    ap.add_argument("--alg_name", default="AlphaEdit")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--hparams_fname", required=True)
    ap.add_argument("--ds_name", default="mcf")
    ap.add_argument("--dataset_size_limit", type=int, default=2000)
    ap.add_argument("--num_edits", type=int, default=100)
    ap.add_argument("--downstream_eval_steps", type=int, default=5)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--log_file", default="")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()

    alphaedit_repo = Path(args.alphaedit_repo)
    if not alphaedit_repo.exists():
        raise FileNotFoundError(f"AlphaEdit repo not found: {alphaedit_repo}")

    cmd = [
        "python3",
        "-m",
        "experiments.evaluate",
        f"--alg_name={args.alg_name}",
        f"--model_name={args.model_name}",
        f"--hparams_fname={args.hparams_fname}",
        f"--ds_name={args.ds_name}",
        f"--dataset_size_limit={args.dataset_size_limit}",
        f"--num_edits={args.num_edits}",
        f"--downstream_eval_steps={args.downstream_eval_steps}",
    ]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "alphaedit_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    env = _env_with_pythonpath(alphaedit_repo)

    started_ts = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(alphaedit_repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_path = Path(args.log_file) if args.log_file else None
    rc = _stream_process(proc, log_path)
    if rc != 0:
        raise RuntimeError(f"AlphaEdit evaluate failed with exit code {rc}")

    results_root = alphaedit_repo / "results" / args.alg_name
    run_dir = _newest_run_dir(results_root, started_ts)
    if run_dir is None:
        raise RuntimeError(f"Could not find AlphaEdit run directory under {results_root}")

    dst = out_root / run_dir.name
    _copy_or_move(run_dir, dst, move=args.move)

    (out_root / "alphaedit_run_dir.txt").write_text(str(dst) + "\n", encoding="utf-8")
    print(f"[run_alphaedit_eval] Results: {dst}")


if __name__ == "__main__":
    main()
