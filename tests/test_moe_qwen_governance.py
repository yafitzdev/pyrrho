from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from pyrrho.moe.qwen_governance import (
    QwenMoEForGovernance,
    QwenMoEGovernanceConfig,
    SemanticRouteAdapter,
    SparseExpertAdapterExperts,
    add_semantic_route_adapter,
    add_sparse_expert_adapters,
    last_token_pool,
    resolve_torch_dtype,
    set_final_dense_layer_trainability,
)


class DummyTrunk(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyLayeredTrunk(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, mlp_only_layers=[0, 2, 3])
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size, bias=False),
            ]
        )

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        hidden = torch.zeros((*input_ids.shape, self.config.hidden_size), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyExperts(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.proj(hidden_states) * top_k_weights[:, :1]


class DummySparseLayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.mlp = SimpleNamespace(experts=DummyExperts(hidden_size, num_experts))


class DummySparseTrunk(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False),
                DummySparseLayer(hidden_size, num_experts),
                DummySparseLayer(hidden_size, num_experts),
                DummySparseLayer(hidden_size, num_experts),
            ]
        )

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        hidden = torch.zeros((*input_ids.shape, self.config.hidden_size), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


def test_last_token_pool_uses_last_non_padding_token() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)

    pooled = last_token_pool(hidden, mask)

    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_qwen_governance_wrapper_matches_stage0_output_contract() -> None:
    cfg = QwenMoEGovernanceConfig(
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=6,
        dropout=0.0,
        freeze_trunk=True,
    )
    model = QwenMoEForGovernance(DummyTrunk(hidden_size=8), cfg)
    model.train()
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    route_ids = torch.tensor([1, 3], dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, route_ids=route_ids)

    assert outputs["governance_logits"].shape == (2, 3)
    assert outputs["route_logits"].shape == (2, 4)
    assert outputs["taxonomy_logits"].shape == (2, 5)
    assert outputs["scalar_preds"].shape == (2, 6)
    assert outputs["selected_routes"].tolist() == [1, 3]
    assert not any(param.requires_grad for param in model.trunk.parameters())


def test_qwen_governance_can_force_gold_routes_during_eval() -> None:
    cfg = QwenMoEGovernanceConfig(
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=6,
        dropout=0.0,
        freeze_trunk=True,
    )
    model = QwenMoEForGovernance(DummyTrunk(hidden_size=8), cfg)
    model.eval()
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    route_ids = torch.tensor([1, 3], dtype=torch.long)

    predicted = model(input_ids=input_ids, attention_mask=attention_mask, route_ids=route_ids)
    forced = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        route_ids=route_ids,
        force_route_ids=True,
    )

    assert forced["selected_routes"].tolist() == [1, 3]
    assert predicted["selected_routes"].shape == forced["selected_routes"].shape


def test_resolve_torch_dtype_aliases() -> None:
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("float32") is torch.float32
    assert resolve_torch_dtype("auto") == "auto"


def test_final_dense_layer_trainability_selects_only_last_dense_layers() -> None:
    model = QwenMoEForGovernance(
        DummyLayeredTrunk(hidden_size=4),
        QwenMoEGovernanceConfig(freeze_trunk=True),
    )

    count, selected = set_final_dense_layer_trainability(model, num_layers=2, trainable=True)

    assert selected == [2, 3]
    assert count == 32
    assert not any(param.requires_grad for param in model.trunk.layers[0].parameters())
    assert not any(param.requires_grad for param in model.trunk.layers[1].parameters())
    assert all(param.requires_grad for param in model.trunk.layers[2].parameters())
    assert all(param.requires_grad for param in model.trunk.layers[3].parameters())


def test_sparse_expert_adapter_starts_as_noop_and_is_trainable() -> None:
    base = DummyExperts(hidden_size=4, num_experts=3)
    adapter = SparseExpertAdapterExperts(base, rank=2, alpha=4.0, dropout=0.0)
    hidden = torch.randn(5, 4)
    top_k_index = torch.tensor([[0], [1], [1], [2], [0]], dtype=torch.long)
    top_k_weights = torch.ones(5, 1)

    base_output = base(hidden, top_k_index, top_k_weights)
    adapter_output = adapter(hidden, top_k_index, top_k_weights)

    assert torch.allclose(adapter_output, base_output)
    assert adapter.adapter_a.requires_grad
    assert adapter.adapter_b.requires_grad


def test_add_sparse_expert_adapters_selects_final_sparse_layers() -> None:
    model = QwenMoEForGovernance(
        DummySparseTrunk(hidden_size=4, num_experts=3),
        QwenMoEGovernanceConfig(freeze_trunk=True),
    )

    count, selected = add_sparse_expert_adapters(
        model,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        num_layers=2,
    )

    assert selected == [2, 3]
    assert count == 2 * 3 * ((2 * 4) + (4 * 2))
    assert not isinstance(model.trunk.layers[1].mlp.experts, SparseExpertAdapterExperts)
    assert isinstance(model.trunk.layers[2].mlp.experts, SparseExpertAdapterExperts)
    assert isinstance(model.trunk.layers[3].mlp.experts, SparseExpertAdapterExperts)


def test_semantic_route_adapter_starts_as_noop_and_is_trainable() -> None:
    adapter = SemanticRouteAdapter(
        num_routes=3,
        hidden_size=4,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    hidden = torch.randn(5, 4)
    route_ids = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    adapted = adapter(hidden, route_ids)

    assert torch.allclose(adapted, hidden)
    assert adapter.adapter_a.requires_grad
    assert adapter.adapter_b.requires_grad


def test_add_semantic_route_adapter_changes_governance_state_path() -> None:
    model = QwenMoEForGovernance(
        DummyTrunk(hidden_size=4),
        QwenMoEGovernanceConfig(num_routes=3, freeze_trunk=True),
    )

    count = add_semantic_route_adapter(model, rank=2, alpha=4.0, dropout=0.0)

    assert count == 3 * ((2 * 4) + (4 * 2))
    assert isinstance(model.semantic_route_adapter, SemanticRouteAdapter)
