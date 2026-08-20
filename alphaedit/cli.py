from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import AlphaEditConfig
from .editor import AlphaEditor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit a dense Qwen model with AlphaEdit."
    )
    parser.add_argument("--model", required=True, help="Qwen model ID or local path")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stats-dir", type=Path, default=Path("data/stats"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--state",
        type=Path,
        help="Previously saved AlphaEdit state for continuing sequential edits",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repository Python code; enable only for trusted repositories",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("--device must be a CUDA device such as cuda:0")

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AlphaEditConfig.from_json(args.config)
    with args.requests.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    requests = payload["requests"] if isinstance(payload, dict) else payload
    if not isinstance(requests, list):
        raise ValueError(
            "Request file must be a JSON list or contain a 'requests' list"
        )

    editor = AlphaEditor(
        model,
        tokenizer,
        config,
        stats_dir=args.stats_dir,
        cache_dir=args.cache_dir,
        state_path=args.state,
    )
    editor.edit(requests)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    editor.save_state(args.output_dir / "alphaedit_state.pt")
    with (args.output_dir / "alphaedit_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(editor.config.to_dict(), handle, indent=2, ensure_ascii=False)
