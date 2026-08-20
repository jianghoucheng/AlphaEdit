from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .compute_target import get_module_input_output_at_words
from .config import AlphaEditConfig


def compute_keys(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    requests: Sequence[dict],
    config: AlphaEditConfig,
    layer: int,
    context_templates: Sequence[Sequence[str]],
) -> torch.Tensor:
    assert config.rewrite_module_template is not None
    representations = get_module_input_output_at_words(
        model,
        tokenizer,
        layer,
        context_templates=[
            context.format(request["prompt"])
            for request in requests
            for template_group in context_templates
            for context in template_group
        ],
        words=[
            request["subject"]
            for request in requests
            for template_group in context_templates
            for _ in template_group
        ],
        module_template=config.rewrite_module_template,
        fact_token_strategy=config.fact_token,
    )[0]

    group_lengths = [len(group) for group in context_templates]
    contexts_per_request = sum(group_lengths)
    averaged = []
    for request_start in range(0, representations.size(0), contexts_per_request):
        group_start = request_start
        group_means = []
        for group_length in group_lengths:
            group_end = group_start + group_length
            group_means.append(representations[group_start:group_end].mean(dim=0))
            group_start = group_end
        averaged.append(torch.stack(group_means).mean(dim=0))
    return torch.stack(averaged)
