from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .tokenized_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from .utils.hooks import Trace
from .utils.running_stats import CombinedStat, Mean, NormMean, SecondMoment, tally

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def layer_stats(
    model,
    tokenizer,
    layer_name: str,
    stats_dir: str | Path,
    dataset_name: str,
    to_collect: list[str],
    *,
    sample_size: int | None = None,
    precision: str = "float32",
    batch_tokens: int | None = None,
    force_recompute: bool = False,
):
    """Load cached layer statistics or compute them for a Qwen model."""

    if dataset_name not in {"wikipedia", "wikitext"}:
        raise ValueError("mom2_dataset must be 'wikipedia' or 'wikitext'")

    max_positions = min(
        int(getattr(model.config, "max_position_embeddings", 4096)),
        4096,
    )
    if batch_tokens is None:
        batch_tokens = max_positions * 3

    model_name = str(model.config._name_or_path).rstrip("/").split("/")[-1]
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    filename = (
        Path(stats_dir)
        / model_name
        / f"{dataset_name}_stats"
        / (f"{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz")
    )

    dataset = None
    if force_recompute or not filename.exists():
        if dataset_name == "wikipedia":
            raw = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        else:
            raw = load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split="train"
            )
        dataset = TokenizedDataset(raw, tokenizer, maxlen=max_positions)

    statistic = CombinedStat(**{name: STAT_TYPES[name]() for name in to_collect})
    loader = tally(
        statistic,
        dataset,
        cache=None if force_recompute else filename,
        sample_size=sample_size,
        batch_size=1,
        collate_fn=length_collation(batch_tokens),
        pin_memory=torch.cuda.is_available(),
        random_sample=1,
        num_workers=2,
    )
    total = None if sample_size is None else sample_size
    device = next(model.parameters()).device
    dtype = getattr(torch, precision)

    with torch.no_grad():
        for batch_group in tqdm(loader, total=total):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with Trace(
                    model,
                    layer_name,
                    retain_input=True,
                    retain_output=False,
                    stop=True,
                ) as trace:
                    model(**batch)
                features = flatten_masked_batch(
                    trace.input,
                    batch["attention_mask"],
                ).to(dtype=dtype)
                statistic.add(features)
    return statistic
