from types import SimpleNamespace

import pytest
import torch
from torch import nn

from alphaedit.config import AlphaEditConfig
from alphaedit.editor import AlphaEditor, _match_shape


class DenseQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen2",
            _name_or_path="Qwen/Qwen2.5-test",
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer() for _ in range(3)])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(6, 4, bias=False)


def test_qwen_layout_is_detected():
    config = AlphaEditConfig(layers=[0, 1], v_loss_layer=2)
    resolved = config.resolve_layout(DenseQwen())

    assert resolved.rewrite_module_template == "model.layers.{}.mlp.down_proj"
    assert resolved.layer_module_template == "model.layers.{}"
    assert resolved.final_norm_module == "model.norm"


def test_non_qwen_model_is_rejected():
    model = DenseQwen()
    model.config.model_type = "llama"
    model.config._name_or_path = "meta-llama/test"

    with pytest.raises(ValueError, match="Only Qwen"):
        AlphaEditConfig(layers=[0], v_loss_layer=1).resolve_layout(model)


def test_request_normalization_accepts_string_target():
    requests = [
        {
            "prompt": "{} is in",
            "subject": "A",
            "target_new": "B",
        }
    ]

    normalized = AlphaEditor._normalize_requests(requests)

    assert normalized[0]["target_new"] == {"str": " B"}
    assert normalized[0]["case_id"] == 0
    assert requests[0]["target_new"] == "B"


def test_update_matrix_can_be_transposed_to_weight_shape():
    matrix = torch.zeros(3, 2)
    assert _match_shape(matrix, torch.Size([2, 3])).shape == (2, 3)
