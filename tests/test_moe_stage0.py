from __future__ import annotations

import json
from pathlib import Path

import torch

from pyrrho.moe.data import MoEJsonlDataset, MoEVocab, collate_moe_batch
from pyrrho.moe.losses import (
    GovernanceSampleWeightPolicy,
    MoELossWeights,
    TrustGuardTargetPolicy,
    build_governance_sample_weights,
    build_trust_guard_targets,
    multitask_loss,
)
from pyrrho.moe.metrics import moe_eval_metrics
from pyrrho.moe.modeling import (
    GuardedSupportAggregatingMoEConfig,
    GuardedSupportAggregatingMoEForGovernance,
    RouteCoupledMoEConfig,
    RouteCoupledMoEForGovernance,
    SupportAggregatingMoEConfig,
    SupportAggregatingMoEForGovernance,
    TinyMoEConfig,
    TinyMoEForGovernance,
    TokenRouteCoupledMoEConfig,
    TokenRouteCoupledMoEForGovernance,
    TrustGuardedSupportAggregatingMoEConfig,
    TrustGuardedSupportAggregatingMoEForGovernance,
)


def test_tiny_moe_forward_and_loss() -> None:
    cfg = TinyMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
    )
    model = TinyMoEForGovernance(cfg)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    route_ids = torch.tensor([1, 3], dtype=torch.long)
    outputs = model(input_ids, attention_mask, route_ids=route_ids)

    assert outputs["governance_logits"].shape == (2, 3)
    assert outputs["route_logits"].shape == (2, 4)
    assert outputs["taxonomy_logits"].shape == (2, 5)
    assert outputs["scalar_preds"].shape == (2, 3)
    assert outputs["selected_routes"].tolist() == [1, 3]

    loss, parts = multitask_loss(
        outputs,
        labels=torch.tensor([0, 2], dtype=torch.long),
        route_ids=route_ids,
        taxonomy_ids=torch.tensor([2, 4], dtype=torch.long),
        scalar_targets=torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]], dtype=torch.float32),
        scalar_mask=torch.ones((2, 3), dtype=torch.float32),
        weights=MoELossWeights(),
    )
    assert loss.requires_grad
    assert parts["loss"] > 0


def test_tiny_moe_can_force_gold_routes_at_eval() -> None:
    cfg = TinyMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
    )
    model = TinyMoEForGovernance(cfg).eval()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.zero_()

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    route_ids = torch.tensor([1, 3], dtype=torch.long)

    predicted = model(input_ids, attention_mask, route_ids=route_ids)
    forced = model(
        input_ids,
        attention_mask,
        route_ids=route_ids,
        force_route_ids=True,
    )

    assert predicted["selected_routes"].tolist() == [0, 0]
    assert forced["selected_routes"].tolist() == [1, 3]


def test_route_coupled_moe_uses_selected_route_path() -> None:
    cfg = RouteCoupledMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_expert_layers=2,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
    )
    model = RouteCoupledMoEForGovernance(cfg).eval()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.zero_()

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    route_ids = torch.tensor([1, 3], dtype=torch.long)

    predicted = model(input_ids, attention_mask, route_ids=route_ids)
    forced = model(
        input_ids,
        attention_mask,
        route_ids=route_ids,
        force_route_ids=True,
    )

    assert predicted["selected_routes"].tolist() == [0, 0]
    assert forced["selected_routes"].tolist() == [1, 3]
    assert not torch.allclose(predicted["governance_logits"], forced["governance_logits"])


def test_token_route_coupled_moe_uses_selected_route_path() -> None:
    cfg = TokenRouteCoupledMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_expert_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
    )
    model = TokenRouteCoupledMoEForGovernance(cfg).eval()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.zero_()

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    route_ids = torch.tensor([1, 3], dtype=torch.long)

    predicted = model(input_ids, attention_mask, route_ids=route_ids)
    forced = model(
        input_ids,
        attention_mask,
        route_ids=route_ids,
        force_route_ids=True,
    )

    assert predicted["governance_logits"].shape == (2, 3)
    assert predicted["route_logits"].shape == (2, 4)
    assert predicted["taxonomy_logits"].shape == (2, 5)
    assert predicted["scalar_preds"].shape == (2, 3)
    assert predicted["selected_routes"].tolist() == [0, 0]
    assert forced["selected_routes"].tolist() == [1, 3]
    assert not torch.allclose(predicted["governance_logits"], forced["governance_logits"])


def test_support_aggregating_moe_consumes_query_and_sources() -> None:
    cfg = SupportAggregatingMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_expert_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
        max_query_length=4,
        max_sources=3,
        max_source_length=5,
    )
    model = SupportAggregatingMoEForGovernance(cfg).eval()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.zero_()

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    query_input_ids = torch.tensor([[10, 11, 0], [12, 0, 0]], dtype=torch.long)
    query_attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    source_input_ids = torch.tensor(
        [
            [[20, 21, 0], [22, 23, 24], [0, 0, 0]],
            [[25, 26, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    source_attention_mask = torch.tensor(
        [
            [[1, 1, 0], [1, 1, 1], [0, 0, 0]],
            [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    source_valid_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    route_ids = torch.tensor([1, 3], dtype=torch.long)

    without_support = model(input_ids, attention_mask, route_ids=route_ids)
    with_support = model(
        input_ids,
        attention_mask,
        route_ids=route_ids,
        force_route_ids=True,
        query_input_ids=query_input_ids,
        query_attention_mask=query_attention_mask,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        source_valid_mask=source_valid_mask,
    )

    assert with_support["governance_logits"].shape == (2, 3)
    assert with_support["route_logits"].shape == (2, 4)
    assert with_support["taxonomy_logits"].shape == (2, 5)
    assert with_support["scalar_preds"].shape == (2, 3)
    assert with_support["selected_routes"].tolist() == [1, 3]
    assert not torch.allclose(
        without_support["governance_logits"],
        with_support["governance_logits"],
    )


def test_guarded_support_aggregating_moe_outputs_trust_penalty() -> None:
    cfg = GuardedSupportAggregatingMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_expert_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
        max_query_length=4,
        max_sources=2,
        max_source_length=5,
        trust_penalty_scale=1.5,
    )
    model = GuardedSupportAggregatingMoEForGovernance(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    query_input_ids = torch.tensor([[10, 11], [12, 0]], dtype=torch.long)
    query_attention_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    source_input_ids = torch.tensor(
        [[[20, 21, 0], [22, 0, 0]], [[23, 24, 0], [0, 0, 0]]],
        dtype=torch.long,
    )
    source_attention_mask = torch.tensor(
        [[[1, 1, 0], [1, 0, 0]], [[1, 1, 0], [0, 0, 0]]],
        dtype=torch.float32,
    )
    source_valid_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)

    outputs = model(
        input_ids,
        attention_mask,
        route_ids=torch.tensor([1, 3], dtype=torch.long),
        force_route_ids=True,
        query_input_ids=query_input_ids,
        query_attention_mask=query_attention_mask,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        source_valid_mask=source_valid_mask,
    )

    assert outputs["governance_logits"].shape == (2, 3)
    assert outputs["trust_penalty"].shape == (2,)
    assert torch.all(outputs["trust_penalty"] >= 0)


def test_trust_guarded_support_aggregating_moe_outputs_verifier() -> None:
    cfg = TrustGuardedSupportAggregatingMoEConfig(
        token_vocab_size=128,
        hidden_size=16,
        expert_hidden_size=32,
        num_expert_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routes=4,
        num_taxonomy_patterns=5,
        num_scalar_targets=3,
        max_query_length=4,
        max_sources=2,
        max_source_length=5,
        trust_guard_scale=0.5,
    )
    model = TrustGuardedSupportAggregatingMoEForGovernance(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    query_input_ids = torch.tensor([[10, 11], [12, 0]], dtype=torch.long)
    query_attention_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    source_input_ids = torch.tensor(
        [[[20, 21, 0], [22, 0, 0]], [[23, 24, 0], [0, 0, 0]]],
        dtype=torch.long,
    )
    source_attention_mask = torch.tensor(
        [[[1, 1, 0], [1, 0, 0]], [[1, 1, 0], [0, 0, 0]]],
        dtype=torch.float32,
    )
    source_valid_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)

    outputs = model(
        input_ids,
        attention_mask,
        route_ids=torch.tensor([1, 3], dtype=torch.long),
        force_route_ids=True,
        query_input_ids=query_input_ids,
        query_attention_mask=query_attention_mask,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        source_valid_mask=source_valid_mask,
    )

    assert outputs["candidate_governance_logits"].shape == (2, 3)
    assert outputs["governance_logits"].shape == (2, 3)
    assert outputs["trust_guard_logits"].shape == (2,)
    assert outputs["trust_guard_penalty"].shape == (2,)
    assert torch.all(outputs["trust_guard_penalty"] >= 0)
    assert torch.all(
        outputs["governance_logits"][:, 2]
        <= outputs["candidate_governance_logits"][:, 2]
    )


def test_moe_dataset_loads_teacher_logits_sidecar(tmp_path: Path) -> None:
    rows_path = tmp_path / "train.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    rows = [
        {
            "id": "row-1",
            "text": "Query plus context",
            "label_id": 2,
            "route_id": 1,
            "taxonomy_pattern_id": 0,
            "scalar_targets": {"trustworthy": 0.8},
        },
        {
            "id": "row-2",
            "text": "Different context",
            "label_id": 0,
            "route_id": 0,
            "taxonomy_pattern_id": 0,
            "scalar_targets": {},
        },
    ]
    rows_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    teacher_path.write_text(
        json.dumps({"id": "row-1", "logits": [0.1, 0.2, 3.0]}) + "\n",
        encoding="utf-8",
    )
    vocab = MoEVocab(
        route2id={"general": 0, "technology": 1},
        taxonomy_pattern2id={"supported": 0},
        scalar_fields=("trustworthy",),
    )

    ds = MoEJsonlDataset(
        rows_path,
        vocab=vocab,
        token_vocab_size=128,
        max_length=16,
        teacher_logits_path=teacher_path,
    )
    batch = collate_moe_batch([ds[0], ds[1]])

    assert batch["teacher_logits"].shape == (2, 3)
    assert batch["teacher_mask"].tolist() == [1.0, 0.0]
    assert torch.allclose(
        batch["teacher_logits"][0],
        torch.tensor([0.1, 0.2, 3.0], dtype=torch.float32),
    )


def test_moe_dataset_collates_query_and_sources(tmp_path: Path) -> None:
    rows_path = tmp_path / "train.jsonl"
    rows = [
        {
            "id": "row-1",
            "text": "Question: Q1\n\nSources:\n[1] Alpha\n[2] Beta",
            "query": "Q1",
            "contexts": ["Alpha source text", "Beta source text"],
            "label_id": 2,
            "route_id": 1,
            "taxonomy_pattern_id": 0,
            "scalar_targets": {},
        },
        {
            "id": "row-2",
            "text": "Question: Q2\n\nSources:\n[1] Gamma",
            "query": "Q2",
            "contexts": ["Gamma source text"],
            "label_id": 0,
            "route_id": 0,
            "taxonomy_pattern_id": 0,
            "scalar_targets": {},
        },
    ]
    rows_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    vocab = MoEVocab(
        route2id={"general": 0, "technology": 1},
        taxonomy_pattern2id={"supported": 0},
        scalar_fields=(),
    )

    ds = MoEJsonlDataset(
        rows_path,
        vocab=vocab,
        token_vocab_size=128,
        max_length=16,
        max_query_length=4,
        max_sources=3,
        max_source_length=5,
    )
    batch = collate_moe_batch([ds[0], ds[1]])

    assert batch["query_input_ids"].shape[0] == 2
    assert batch["source_input_ids"].shape[:2] == (2, 3)
    assert batch["source_attention_mask"].shape == batch["source_input_ids"].shape
    assert batch["source_valid_mask"].tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def test_moe_eval_metrics_include_trustworthy_calibration() -> None:
    metrics = moe_eval_metrics(
        governance_logits=torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [0.2, 3.0, 0.1],
                [0.1, 0.0, 3.0],
                [0.2, 0.1, 2.0],
            ]
        ).numpy(),
        labels=torch.tensor([0, 1, 2, 0]).numpy(),
        route_logits=torch.eye(4).numpy(),
        route_labels=torch.tensor([0, 1, 2, 3]).numpy(),
        taxonomy_logits=torch.eye(4).numpy(),
        taxonomy_labels=torch.tensor([0, 1, 2, 3]).numpy(),
        calibration_grid_size=4,
    )

    calibrated = metrics["governance_calibrated"]
    assert "threshold" in calibrated
    assert calibrated["false_trustworthy_rate"] <= 0.057
    assert metrics["route_accuracy"] == 1.0


def test_governance_sample_weight_policy_targets_known_slices() -> None:
    policy = GovernanceSampleWeightPolicy(
        support_taxonomy_ids=(10, 11),
        support_trustworthy_weight=1.5,
        support_taxonomy_weights=((11, 2.0),),
        ft_risk_route_ids=(2,),
        ft_risk_route_non_trustworthy_weight=1.25,
        ft_risk_taxonomy_ids=(12,),
        ft_risk_taxonomy_non_trustworthy_weight=1.4,
    )

    weights = build_governance_sample_weights(
        labels=torch.tensor([2, 0, 1, 2], dtype=torch.long),
        route_ids=torch.tensor([0, 2, 1, 2], dtype=torch.long),
        taxonomy_ids=torch.tensor([10, 12, 12, 11], dtype=torch.long),
        policy=policy,
    )

    assert weights is not None
    assert torch.allclose(weights, torch.tensor([1.5, 1.75, 1.4, 3.0]))


def test_trust_guard_target_policy_weights_known_slices() -> None:
    policy = TrustGuardTargetPolicy(
        positive_support_taxonomy_ids=(10,),
        positive_support_weight=1.5,
        negative_risk_route_ids=(2,),
        negative_risk_route_weight=1.25,
        negative_risk_taxonomy_ids=(12,),
        negative_risk_taxonomy_weight=1.4,
    )

    targets, weights = build_trust_guard_targets(
        labels=torch.tensor([2, 0, 1, 2], dtype=torch.long),
        route_ids=torch.tensor([0, 2, 1, 2], dtype=torch.long),
        taxonomy_ids=torch.tensor([10, 12, 12, 11], dtype=torch.long),
        policy=policy,
    )

    assert torch.allclose(targets, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(weights, torch.tensor([1.5, 1.75, 1.4, 1.0]))


def test_multitask_loss_can_include_teacher_distillation() -> None:
    outputs = {
        "governance_logits": torch.tensor([[0.1, 0.2, 1.0]], requires_grad=True),
        "route_logits": torch.tensor([[1.0, 0.0]], requires_grad=True),
        "taxonomy_logits": torch.tensor([[0.0, 1.0]], requires_grad=True),
        "scalar_preds": torch.tensor([[0.5]], requires_grad=True),
    }

    loss, parts = multitask_loss(
        outputs,
        labels=torch.tensor([2], dtype=torch.long),
        route_ids=torch.tensor([0], dtype=torch.long),
        taxonomy_ids=torch.tensor([1], dtype=torch.long),
        scalar_targets=torch.tensor([[1.0]], dtype=torch.float32),
        scalar_mask=torch.ones((1, 1), dtype=torch.float32),
        teacher_logits=torch.tensor([[3.0, 0.1, 0.1]], dtype=torch.float32),
        teacher_mask=torch.ones(1, dtype=torch.float32),
        weights=MoELossWeights(distillation=0.5),
    )

    assert loss.requires_grad
    assert parts["loss_distillation"] > 0


def test_multitask_loss_can_include_trust_guard_target() -> None:
    outputs = {
        "governance_logits": torch.tensor([[0.1, 0.2, 1.0]], requires_grad=True),
        "route_logits": torch.tensor([[1.0, 0.0]], requires_grad=True),
        "taxonomy_logits": torch.tensor([[0.0, 1.0]], requires_grad=True),
        "scalar_preds": torch.tensor([[0.5]], requires_grad=True),
        "trust_guard_logits": torch.tensor([0.2], requires_grad=True),
    }

    loss, parts = multitask_loss(
        outputs,
        labels=torch.tensor([2], dtype=torch.long),
        route_ids=torch.tensor([0], dtype=torch.long),
        taxonomy_ids=torch.tensor([1], dtype=torch.long),
        scalar_targets=torch.tensor([[1.0]], dtype=torch.float32),
        scalar_mask=torch.ones((1, 1), dtype=torch.float32),
        trust_guard_targets=torch.ones(1, dtype=torch.float32),
        trust_guard_weights=torch.ones(1, dtype=torch.float32),
        weights=MoELossWeights(trust_guard=0.3),
    )

    assert loss.requires_grad
    assert parts["loss_trust_guard"] > 0
