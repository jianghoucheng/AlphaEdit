from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .utils import hooks


def get_reprs_at_word_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    context_templates: Sequence[str],
    words: Sequence[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    indices = get_words_idxs_in_templates(tokenizer, context_templates, words, subtoken)
    contexts = [
        template.format(word)
        for template, word in zip(context_templates, words, strict=True)
    ]
    return get_reprs_at_idxs(
        model,
        tokenizer,
        contexts,
        indices,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tokenizer: PreTrainedTokenizerBase,
    context_templates: Sequence[str],
    words: Sequence[str],
    subtoken: str,
) -> list[list[int]]:
    if len(context_templates) != len(words):
        raise ValueError("context_templates and words must have the same length")
    if any(template.count("{}") != 1 for template in context_templates):
        raise ValueError(
            "Each context template must contain exactly one '{}' placeholder"
        )

    indices = []
    for template, word in zip(context_templates, words, strict=True):
        prefix, suffix = template.split("{}")
        text = prefix + word + suffix
        start, end = len(prefix), len(prefix) + len(word)

        try:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            offsets = encoding["offset_mapping"]
            subject_tokens = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > start and token_start < end
            ]
        except (KeyError, NotImplementedError, TypeError):
            prefix_length = len(tokenizer.encode(prefix))
            subject_length = len(tokenizer.encode(prefix + word)) - prefix_length
            subject_tokens = list(range(prefix_length, prefix_length + subject_length))

        if not subject_tokens:
            raise ValueError(f"Tokenizer produced no tokens for subject {word!r}")
        if subtoken == "first":
            selected = subject_tokens[0]
        elif subtoken == "last":
            selected = subject_tokens[-1]
        elif subtoken == "first_after_last":
            input_length = len(tokenizer.encode(text))
            selected = min(subject_tokens[-1] + 1, input_length - 1)
        else:
            raise ValueError(f"Unknown subject subtoken strategy: {subtoken}")
        indices.append([selected])
    return indices


def get_reprs_at_idxs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    contexts: Sequence[str],
    indices: Sequence[Sequence[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if track not in {"in", "out", "both"}:
        raise ValueError("track must be 'in', 'out', or 'both'")

    retain_input = track in {"in", "both"}
    retain_output = track in {"out", "both"}
    module_name = module_template.format(layer)
    collected: dict[str, list[torch.Tensor]] = {"in": [], "out": []}

    for start in range(0, len(contexts), 32):
        batch_contexts = contexts[start : start + 32]
        batch_indices = indices[start : start + 32]
        inputs = tokenizer(list(batch_contexts), padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with (
            torch.no_grad(),
            hooks.Trace(
                model,
                module_name,
                retain_input=retain_input,
                retain_output=retain_output,
            ) as trace,
        ):
            model(**inputs)

        if retain_input:
            _collect(trace.input, batch_indices, collected["in"])
        if retain_output:
            _collect(trace.output, batch_indices, collected["out"])

    result = {key: torch.stack(value) for key, value in collected.items() if value}
    if track == "both":
        return result["in"], result["out"]
    return result[track]


def _collect(
    representation: torch.Tensor | tuple[torch.Tensor, ...],
    indices: Sequence[Sequence[int]],
    destination: list[torch.Tensor],
) -> None:
    tensor = representation[0] if isinstance(representation, tuple) else representation
    if tensor.shape[0] != len(indices):
        raise ValueError(
            "Expected batch-first Qwen activations, "
            f"got shape {tuple(tensor.shape)} for batch size {len(indices)}"
        )
    for row, row_indices in enumerate(indices):
        destination.append(tensor[row, list(row_indices)].mean(dim=0))
