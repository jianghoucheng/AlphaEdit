from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from torch import nn


@dataclass(frozen=True)
class AlphaEditConfig:
    """Hyperparameters for dense Qwen-family causal language models."""

    layers: list[int]
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 25
    v_lr: float = 0.1
    v_loss_layer: int = 0
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.75
    kl_factor: float = 0.0625
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100_000
    mom2_dtype: str = "float32"
    nullspace_threshold: float = 0.02
    l2_regularization: float = 10.0
    rewrite_module_template: str | None = None
    layer_module_template: str | None = None
    final_norm_module: str | None = None
    lm_head_module: str = "lm_head"

    @classmethod
    def from_json(cls, path: str | Path) -> AlphaEditConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)

        aliases = {
            "L2": "l2_regularization",
            "rewrite_module_tmp": "rewrite_module_template",
            "layer_module_tmp": "layer_module_template",
            "ln_f_module": "final_norm_module",
        }
        data = {aliases.get(key, key): value for key, value in raw.items()}
        supported = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in supported})

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("layers must contain at least one decoder layer")
        if self.layers != sorted(set(self.layers)):
            raise ValueError("layers must be unique and sorted")
        if self.fact_token not in {
            "subject_first",
            "subject_last",
            "subject_first_after_last",
        }:
            raise ValueError("fact_token must select a subject token")
        if self.v_num_grad_steps < 1:
            raise ValueError("v_num_grad_steps must be positive")
        if self.mom2_n_samples < 1:
            raise ValueError("mom2_n_samples must be positive")

    def resolve_layout(self, model: nn.Module) -> AlphaEditConfig:
        """Detect the module layout used by dense Qwen checkpoints."""

        model_type = str(getattr(model.config, "model_type", "")).lower()
        name = str(getattr(model.config, "_name_or_path", "")).lower()
        if "qwen" not in model_type and "qwen" not in name:
            raise ValueError(
                f"Only Qwen models are supported; got model_type={model_type!r}"
            )

        candidates = (
            ("model.layers.{}.mlp.down_proj", "model.layers.{}", "model.norm"),
            (
                "model.language_model.layers.{}.mlp.down_proj",
                "model.language_model.layers.{}",
                "model.language_model.norm",
            ),
        )
        requested_layer = self.layers[0]
        for rewrite, layer, norm in candidates:
            if _has_module(model, rewrite.format(requested_layer)) and _has_module(
                model, norm
            ):
                resolved = replace(
                    self,
                    rewrite_module_template=rewrite,
                    layer_module_template=layer,
                    final_norm_module=norm,
                )
                resolved._validate_model_modules(model)
                return resolved

        if (
            self.rewrite_module_template
            and self.layer_module_template
            and self.final_norm_module
        ):
            self._validate_model_modules(model)
            return self

        raise ValueError(
            "Unsupported Qwen architecture: expected a dense MLP with mlp.down_proj. "
            "Quantized and MoE checkpoints are not supported."
        )

    def _validate_model_modules(self, model: nn.Module) -> None:
        self.validate()
        assert self.rewrite_module_template is not None
        assert self.layer_module_template is not None
        assert self.final_norm_module is not None

        module_names = dict(model.named_modules())
        required = [
            *(self.rewrite_module_template.format(layer) for layer in self.layers),
            *(self.layer_module_template.format(layer) for layer in self.layers),
            self.layer_module_template.format(self.v_loss_layer),
            self.final_norm_module,
            self.lm_head_module,
        ]
        missing = [name for name in required if name not in module_names]
        if missing:
            raise ValueError(
                f"Model is missing configured modules: {', '.join(missing)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _has_module(model: nn.Module, name: str) -> bool:
    return any(module_name == name for module_name, _ in model.named_modules())
