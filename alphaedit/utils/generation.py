from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def generate_contexts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    generations_per_prompt: int,
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        list(prompts),
        padding=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_k=5,
            max_new_tokens=max_new_tokens,
            num_return_sequences=generations_per_prompt,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
