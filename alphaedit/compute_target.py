from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from . import representations
from .config import AlphaEditConfig
from .utils import hooks


def compute_target(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    request: dict,
    config: AlphaEditConfig,
    layer: int,
    context_templates: Sequence[Sequence[str]],
) -> torch.Tensor:
    assert config.final_norm_module is not None
    assert config.layer_module_template is not None

    device = next(model.parameters()).device
    lm_head = hooks.get_module(model, config.lm_head_module)
    final_norm = hooks.get_module(model, config.final_norm_module)
    lm_weight = lm_head.weight.T
    lm_bias = getattr(lm_head, "bias", None)
    if lm_bias is None:
        lm_bias = torch.zeros(lm_head.weight.shape[0], device=device)

    target_ids = tokenizer(
        request["target_new"]["str"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].to(device)
    if target_ids.numel() == 0:
        raise ValueError("target_new.str must produce at least one token")

    rewriting_prompts = [
        context.format(request["prompt"]) + tokenizer.decode(target_ids[:-1])
        for template_group in context_templates
        for context in template_group
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    formatted_prompts = [prompt.format(request["subject"]) for prompt in all_prompts]
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    rewriting_targets = torch.full_like(inputs["input_ids"], -100)
    for row in range(len(rewriting_prompts)):
        sequence_length = int(inputs["attention_mask"][row].sum())
        rewriting_targets[row, sequence_length - len(target_ids) : sequence_length] = (
            target_ids
        )

    lookup_indices = [
        find_fact_lookup_index(
            prompt,
            request["subject"],
            tokenizer,
            config.fact_token,
        )
        for prompt in all_prompts
    ]
    loss_layer = max(config.v_loss_layer, layer)
    hidden_size = int(lm_head.weight.shape[1])
    delta = torch.zeros(
        hidden_size,
        device=device,
        dtype=next(model.parameters()).dtype,
        requires_grad=True,
    )
    target_initial = None
    kl_distribution_initial = None

    def edit_output(output, layer_name):
        nonlocal target_initial
        if layer_name != config.layer_module_template.format(layer):
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        if target_initial is None:
            target_initial = hidden[0, lookup_indices[0]].detach().clone()
        for row, index in enumerate(lookup_indices):
            hidden[row, index, :] += delta
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    optimizer = torch.optim.Adam([delta], lr=config.v_lr)
    hooks.set_requires_grad(False, model)

    for _ in range(config.v_num_grad_steps):
        optimizer.zero_grad()
        with hooks.TraceDict(
            model,
            layers=[
                config.layer_module_template.format(loss_layer),
                config.layer_module_template.format(layer),
            ],
            retain_output=True,
            edit_output=edit_output,
        ) as traces:
            logits = model(**inputs).logits
            kl_logits = torch.stack(
                [
                    logits[-len(kl_prompts) + row, index, :]
                    for row, index in enumerate(lookup_indices[-len(kl_prompts) :])
                ]
            )
            kl_log_probs = torch.log_softmax(kl_logits, dim=-1)
            if kl_distribution_initial is None:
                kl_distribution_initial = kl_log_probs.detach().clone()

        hidden = traces[config.layer_module_template.format(loss_layer)].output
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        rewritten_hidden = hidden[: len(rewriting_prompts)]
        output_logits = final_norm(rewritten_hidden) @ lm_weight.to(
            rewritten_hidden.device
        )
        output_logits = output_logits + lm_bias.to(rewritten_hidden.device)
        log_probs = torch.log_softmax(output_logits, dim=-1)
        gathered = torch.gather(
            log_probs,
            2,
            rewriting_targets.clamp_min(0).unsqueeze(2),
        ).squeeze(2)
        mask = rewriting_targets.ne(-100)
        nll_each = -(gathered * mask).sum(dim=1) / target_ids.numel()
        nll_loss = nll_each.mean()
        kl_loss = config.kl_factor * torch.nn.functional.kl_div(
            kl_distribution_initial,
            kl_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        assert target_initial is not None
        regularization = config.v_weight_decay * (
            delta.norm() / target_initial.norm().square()
        )
        loss = nll_loss + kl_loss + regularization
        if loss.item() < 0.05:
            break

        loss.backward()
        optimizer.step()
        maximum_norm = config.clamp_norm_factor * target_initial.norm()
        if delta.norm() > maximum_norm:
            with torch.no_grad():
                delta.mul_(maximum_norm / delta.norm())

    assert target_initial is not None
    return (target_initial + delta).detach()


def get_module_input_output_at_words(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer: int,
    context_templates: Sequence[str],
    words: Sequence[str],
    module_template: str,
    fact_token_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not fact_token_strategy.startswith("subject_"):
        raise ValueError("Qwen AlphaEdit requires a subject_* fact token strategy")
    subtoken = fact_token_strategy.removeprefix("subject_")
    inputs, outputs = representations.get_reprs_at_word_tokens(
        model=model,
        tokenizer=tokenizer,
        context_templates=context_templates,
        words=words,
        layer=layer,
        module_template=module_template,
        subtoken=subtoken,
        track="both",
    )
    return inputs.detach(), outputs.detach()


def find_fact_lookup_index(
    prompt: str,
    subject: str,
    tokenizer: PreTrainedTokenizerBase,
    fact_token_strategy: str,
) -> int:
    if not fact_token_strategy.startswith("subject_"):
        raise ValueError("Qwen AlphaEdit requires a subject_* fact token strategy")
    return representations.get_words_idxs_in_templates(
        tokenizer,
        [prompt],
        [subject],
        fact_token_strategy.removeprefix("subject_"),
    )[0][0]
