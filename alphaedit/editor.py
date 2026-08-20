from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .compute_keys import compute_keys
from .compute_target import compute_target, get_module_input_output_at_words
from .config import AlphaEditConfig
from .layer_stats import layer_stats
from .utils import hooks
from .utils.generation import generate_contexts


class AlphaEditor:
    """Stateful AlphaEdit editor for dense Qwen-family language models."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: AlphaEditConfig,
        *,
        stats_dir: str | Path,
        cache_dir: str | Path | None = None,
        state_path: str | Path | None = None,
        context_templates: list[list[str]] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config.resolve_layout(model)
        self.stats_dir = Path(stats_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.context_templates = context_templates

        self._validate_single_device()
        if state_path is None:
            self.projection, self.key_covariance = self._initialize_state()
        else:
            self.projection, self.key_covariance = self._load_state(state_path)

    def edit(self, requests: list[dict[str, Any]]) -> PreTrainedModel:
        """Apply one batch of edits in place and return the edited model."""

        normalized = self._normalize_requests(requests)
        contexts = self._get_context_templates()
        target_layer = self.config.layers[-1]
        targets = torch.stack(
            [
                self._load_or_compute_target(request, target_layer, contexts)
                for request in normalized
            ],
            dim=1,
        )
        assert self.config.rewrite_module_template is not None
        assert self.config.layer_module_template is not None

        for layer_index, layer in enumerate(self.config.layers):
            keys = compute_keys(
                self.model,
                self.tokenizer,
                normalized,
                self.config,
                layer,
                contexts,
            ).T.float()
            current_targets = get_module_input_output_at_words(
                self.model,
                self.tokenizer,
                target_layer,
                context_templates=[request["prompt"] for request in normalized],
                words=[request["subject"] for request in normalized],
                module_template=self.config.layer_module_template,
                fact_token_strategy=self.config.fact_token,
            )[1].T.float()
            residual = (targets.float() - current_targets) / (
                len(self.config.layers) - layer_index
            )

            device = keys.device
            projection = self.projection[layer_index].to(device)
            covariance = self.key_covariance[layer_index].to(device)
            identity = torch.eye(keys.shape[0], device=device)
            coefficient = projection @ (keys @ keys.T + covariance)
            coefficient = coefficient + self.config.l2_regularization * identity
            update = torch.linalg.solve(
                coefficient,
                projection @ keys @ residual.T,
            )

            weight_name = self.config.rewrite_module_template.format(layer) + ".weight"
            weight = hooks.get_parameter(self.model, weight_name)
            update = _match_shape(update, weight.shape).to(weight.dtype)
            with torch.no_grad():
                weight.add_(update)

            del keys, current_targets, residual, projection, covariance, update
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._update_key_covariance(normalized, contexts)
        return self.model

    def save_state(self, path: str | Path) -> None:
        """Save sequential-edit state needed to resume later."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layers": self.config.layers,
                "projection": self.projection,
                "key_covariance": self.key_covariance,
            },
            destination,
        )

    def _initialize_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        projections = [self._compute_projection(layer) for layer in self.config.layers]
        projection = torch.stack(projections).cpu()
        key_covariance = torch.zeros_like(projection)
        return projection, key_covariance

    def _compute_projection(self, layer: int) -> torch.Tensor:
        assert self.config.rewrite_module_template is not None
        statistic = layer_stats(
            self.model,
            self.tokenizer,
            self.config.rewrite_module_template.format(layer),
            self.stats_dir,
            self.config.mom2_dataset,
            ["mom2"],
            sample_size=self.config.mom2_n_samples,
            precision=self.config.mom2_dtype,
        )
        covariance = statistic.mom2.moment().float().cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        null_indices = eigenvalues < self.config.nullspace_threshold
        null_vectors = eigenvectors[:, null_indices]
        return null_vectors @ null_vectors.T

    def _update_key_covariance(
        self,
        requests: list[dict[str, Any]],
        context_templates: list[list[str]],
    ) -> None:
        for layer_index, layer in enumerate(self.config.layers):
            keys = compute_keys(
                self.model,
                self.tokenizer,
                requests,
                self.config,
                layer,
                context_templates,
            ).T.float()
            self.key_covariance[layer_index].add_((keys @ keys.T).cpu())

    def _load_or_compute_target(
        self,
        request: dict[str, Any],
        layer: int,
        context_templates: list[list[str]],
    ) -> torch.Tensor:
        cache_path = None
        if self.cache_dir is not None:
            cache_path = (
                self.cache_dir / f"layer_{layer}" / (f"case_{request['case_id']}.npz")
            )
            if cache_path.exists():
                data = np.load(cache_path)
                return torch.from_numpy(data["target"]).to(
                    next(self.model.parameters()).device
                )

        target = compute_target(
            self.model,
            self.tokenizer,
            request,
            self.config,
            layer,
            context_templates,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, target=target.float().cpu().numpy())
        return target

    def _get_context_templates(self) -> list[list[str]]:
        if self.context_templates is None:
            generated = generate_contexts(
                self.model,
                self.tokenizer,
                ["The", "Therefore", "Because", "I", "You"],
                generations_per_prompt=1,
                max_new_tokens=10,
            )
            self.context_templates = [
                ["{}"],
                [
                    text.replace("{", " ").replace("}", " ") + ". {}"
                    for text in generated
                ],
            ]
        return self.context_templates

    def _load_state(self, path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if state["layers"] != self.config.layers:
            raise ValueError("Saved state layers do not match the current config")
        return state["projection"].float(), state["key_covariance"].float()

    def _validate_single_device(self) -> None:
        devices = {parameter.device for parameter in self.model.parameters()}
        if len(devices) != 1:
            raise ValueError(
                "AlphaEdit currently requires the Qwen model on one device; "
                "device_map='auto' sharding is not supported"
            )
        device = next(iter(devices))
        if device.type != "cuda":
            raise ValueError("AlphaEdit requires a CUDA device")

    @staticmethod
    def _normalize_requests(
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            raise ValueError("At least one edit request is required")

        normalized = deepcopy(requests)
        for index, request in enumerate(normalized):
            missing = {"prompt", "subject", "target_new"} - request.keys()
            if missing:
                raise ValueError(f"Request {index} is missing: {sorted(missing)}")
            if request["prompt"].count("{}") != 1:
                raise ValueError(f"Request {index} prompt must contain one '{{}}'")
            target = request["target_new"]
            if isinstance(target, str):
                target = {"str": target}
                request["target_new"] = target
            if not isinstance(target, dict) or not target.get("str"):
                raise ValueError(f"Request {index} target_new must contain 'str'")
            if not target["str"].startswith(" "):
                target["str"] = " " + target["str"]
            request.setdefault("case_id", index)
        return normalized


def _match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Computed update shape {tuple(matrix.shape)} does not match "
        f"weight shape {tuple(shape)}"
    )
