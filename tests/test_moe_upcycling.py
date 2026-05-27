from __future__ import annotations

import pytest
import torch

from pyrrho.moe.upcycling import (
    compress_qwen_ffn,
    qwen_mlp_weight_keys,
    resolve_safetensor_files_for_keys,
    select_ffn_channels_by_norm,
)


def test_qwen_mlp_weight_keys_names_expected_tensors() -> None:
    assert qwen_mlp_weight_keys(2) == (
        "model.layers.2.mlp.gate_proj.weight",
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.mlp.down_proj.weight",
    )


def test_qwen_mlp_weight_keys_rejects_negative_layer() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        qwen_mlp_weight_keys(-1)


def test_resolve_safetensor_files_for_keys_uses_index_map() -> None:
    resolved = resolve_safetensor_files_for_keys(
        {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        },
        ["a", "b"],
    )

    assert resolved == {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",
    }


def test_resolve_safetensor_files_for_keys_falls_back_to_single_file() -> None:
    assert resolve_safetensor_files_for_keys(None, ["a"]) == {
        "a": "model.safetensors"
    }


def test_resolve_safetensor_files_for_keys_rejects_missing_index_key() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        resolve_safetensor_files_for_keys({"a": "model.safetensors"}, ["a", "b"])


def test_select_ffn_channels_by_norm_returns_strongest_in_original_order() -> None:
    gate = torch.zeros((5, 2))
    up = torch.zeros((5, 2))
    down = torch.zeros((2, 5))
    gate[3, 0] = 10.0
    up[1, 0] = 8.0
    down[0, 4] = 6.0

    selected = select_ffn_channels_by_norm(gate, up, down, target_dim=3)

    assert selected.tolist() == [1, 3, 4]


def test_compress_qwen_ffn_slices_gate_up_and_down_consistently() -> None:
    gate = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    up = gate + 100
    down = torch.arange(20, dtype=torch.float32).reshape(4, 5) + 200

    compressed = compress_qwen_ffn(gate, up, down, target_dim=2)

    idx = compressed.selected_indices
    assert compressed.gate_proj.shape == (2, 4)
    assert compressed.up_proj.shape == (2, 4)
    assert compressed.down_proj.shape == (4, 2)
    assert torch.equal(compressed.gate_proj, gate.index_select(0, idx))
    assert torch.equal(compressed.up_proj, up.index_select(0, idx))
    assert torch.equal(compressed.down_proj, down.index_select(1, idx))


def test_compress_qwen_ffn_rejects_invalid_target_dim() -> None:
    gate = torch.zeros((3, 2))
    up = torch.zeros((3, 2))
    down = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="exceeds"):
        compress_qwen_ffn(gate, up, down, target_dim=4)
