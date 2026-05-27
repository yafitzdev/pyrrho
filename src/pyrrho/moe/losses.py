"""Losses and metrics for pyrrho-MoE prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class MoELossWeights:
    governance: float = 1.0
    route: float = 0.7
    taxonomy: float = 0.3
    scalar: float = 0.2
    distillation: float = 0.0
    load_balance: float = 0.02
    false_trustworthy_weight: float = 2.3
    trust_guard: float = 0.0


@dataclass(frozen=True)
class GovernanceSampleWeightPolicy:
    support_taxonomy_ids: tuple[int, ...] = ()
    support_trustworthy_weight: float = 1.0
    support_taxonomy_weights: tuple[tuple[int, float], ...] = ()
    ft_risk_route_ids: tuple[int, ...] = ()
    ft_risk_route_non_trustworthy_weight: float = 1.0
    ft_risk_taxonomy_ids: tuple[int, ...] = ()
    ft_risk_taxonomy_non_trustworthy_weight: float = 1.0
    trustworthy_label_id: int = 2


@dataclass(frozen=True)
class TrustGuardTargetPolicy:
    positive_support_taxonomy_ids: tuple[int, ...] = ()
    positive_support_weight: float = 1.0
    negative_risk_route_ids: tuple[int, ...] = ()
    negative_risk_route_weight: float = 1.0
    negative_risk_taxonomy_ids: tuple[int, ...] = ()
    negative_risk_taxonomy_weight: float = 1.0
    trustworthy_label_id: int = 2


def load_balance_loss(route_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(route_logits, dim=-1).mean(dim=0)
    target = torch.full_like(probs, 1.0 / probs.numel())
    return functional.mse_loss(probs, target)


def scalar_mse(
    scalar_preds: torch.Tensor,
    scalar_targets: torch.Tensor,
    scalar_mask: torch.Tensor,
) -> torch.Tensor:
    denom = scalar_mask.sum().clamp_min(1.0)
    return (((scalar_preds - scalar_targets) ** 2) * scalar_mask).sum() / denom


def _isin(values: torch.Tensor, candidates: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(values, dtype=torch.bool)
    for candidate in candidates:
        mask |= values == int(candidate)
    return mask


def build_governance_sample_weights(
    *,
    labels: torch.Tensor,
    route_ids: torch.Tensor,
    taxonomy_ids: torch.Tensor,
    policy: GovernanceSampleWeightPolicy | None,
) -> torch.Tensor | None:
    if policy is None:
        return None
    weights = torch.ones(labels.shape, dtype=torch.float32, device=labels.device)
    trustworthy = labels == policy.trustworthy_label_id
    non_trustworthy = ~trustworthy

    if policy.support_taxonomy_ids and policy.support_trustworthy_weight != 1.0:
        support_mask = trustworthy & _isin(taxonomy_ids, policy.support_taxonomy_ids)
        weights = torch.where(
            support_mask,
            weights * float(policy.support_trustworthy_weight),
            weights,
        )
    for taxonomy_id, taxonomy_weight in policy.support_taxonomy_weights:
        if taxonomy_weight == 1.0:
            continue
        support_mask = trustworthy & (taxonomy_ids == int(taxonomy_id))
        weights = torch.where(
            support_mask,
            weights * float(taxonomy_weight),
            weights,
        )
    if policy.ft_risk_route_ids and policy.ft_risk_route_non_trustworthy_weight != 1.0:
        route_risk_mask = non_trustworthy & _isin(route_ids, policy.ft_risk_route_ids)
        weights = torch.where(
            route_risk_mask,
            weights * float(policy.ft_risk_route_non_trustworthy_weight),
            weights,
        )
    if (
        policy.ft_risk_taxonomy_ids
        and policy.ft_risk_taxonomy_non_trustworthy_weight != 1.0
    ):
        taxonomy_risk_mask = non_trustworthy & _isin(
            taxonomy_ids,
            policy.ft_risk_taxonomy_ids,
        )
        weights = torch.where(
            taxonomy_risk_mask,
            weights * float(policy.ft_risk_taxonomy_non_trustworthy_weight),
            weights,
        )
    return weights


def build_trust_guard_targets(
    *,
    labels: torch.Tensor,
    route_ids: torch.Tensor,
    taxonomy_ids: torch.Tensor,
    policy: TrustGuardTargetPolicy | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    trustworthy_label_id = (
        policy.trustworthy_label_id if policy is not None else 2
    )
    trustworthy = labels == int(trustworthy_label_id)
    targets = trustworthy.to(dtype=torch.float32)
    weights = torch.ones(labels.shape, dtype=torch.float32, device=labels.device)
    if policy is None:
        return targets, weights

    if policy.positive_support_taxonomy_ids and policy.positive_support_weight != 1.0:
        positive_support_mask = trustworthy & _isin(
            taxonomy_ids,
            policy.positive_support_taxonomy_ids,
        )
        weights = torch.where(
            positive_support_mask,
            weights * float(policy.positive_support_weight),
            weights,
        )
    negative = ~trustworthy
    if policy.negative_risk_route_ids and policy.negative_risk_route_weight != 1.0:
        route_risk_mask = negative & _isin(route_ids, policy.negative_risk_route_ids)
        weights = torch.where(
            route_risk_mask,
            weights * float(policy.negative_risk_route_weight),
            weights,
        )
    if (
        policy.negative_risk_taxonomy_ids
        and policy.negative_risk_taxonomy_weight != 1.0
    ):
        taxonomy_risk_mask = negative & _isin(
            taxonomy_ids,
            policy.negative_risk_taxonomy_ids,
        )
        weights = torch.where(
            taxonomy_risk_mask,
            weights * float(policy.negative_risk_taxonomy_weight),
            weights,
        )
    return targets, weights


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    *,
    labels: torch.Tensor,
    route_ids: torch.Tensor,
    taxonomy_ids: torch.Tensor,
    scalar_targets: torch.Tensor,
    scalar_mask: torch.Tensor,
    weights: MoELossWeights,
    teacher_logits: torch.Tensor | None = None,
    teacher_mask: torch.Tensor | None = None,
    governance_sample_weights: torch.Tensor | None = None,
    trust_guard_targets: torch.Tensor | None = None,
    trust_guard_weights: torch.Tensor | None = None,
    distillation_temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    class_weights = torch.tensor(
        [weights.false_trustworthy_weight, weights.false_trustworthy_weight, 1.0],
        dtype=outputs["governance_logits"].dtype,
        device=outputs["governance_logits"].device,
    )
    governance_per_row = functional.cross_entropy(
        outputs["governance_logits"],
        labels,
        weight=class_weights,
        reduction="none",
    )
    if governance_sample_weights is None:
        governance = governance_per_row.mean()
    else:
        row_weights = governance_sample_weights.to(
            device=governance_per_row.device,
            dtype=governance_per_row.dtype,
        )
        governance = (governance_per_row * row_weights).sum() / row_weights.sum().clamp_min(1.0)
    route = functional.cross_entropy(outputs["route_logits"], route_ids)
    taxonomy = functional.cross_entropy(outputs["taxonomy_logits"], taxonomy_ids)
    scalar = scalar_mse(outputs["scalar_preds"], scalar_targets, scalar_mask)
    balance = load_balance_loss(outputs["route_logits"])
    trust_guard = outputs["governance_logits"].new_tensor(0.0)
    if (
        weights.trust_guard > 0
        and "trust_guard_logits" in outputs
        and trust_guard_targets is not None
    ):
        target = trust_guard_targets.to(
            device=outputs["trust_guard_logits"].device,
            dtype=outputs["trust_guard_logits"].dtype,
        )
        guard_per_row = functional.binary_cross_entropy_with_logits(
            outputs["trust_guard_logits"],
            target,
            reduction="none",
        )
        if trust_guard_weights is None:
            trust_guard = guard_per_row.mean()
        else:
            row_weights = trust_guard_weights.to(
                device=guard_per_row.device,
                dtype=guard_per_row.dtype,
            )
            trust_guard = (guard_per_row * row_weights).sum() / row_weights.sum().clamp_min(
                1.0
            )
    distillation = outputs["governance_logits"].new_tensor(0.0)
    if teacher_logits is not None and weights.distillation > 0:
        if teacher_mask is None:
            active = torch.ones(
                teacher_logits.shape[0],
                dtype=torch.bool,
                device=teacher_logits.device,
            )
        else:
            active = teacher_mask.to(device=teacher_logits.device, dtype=torch.bool)
        if bool(active.any()):
            temperature = float(distillation_temperature)
            student_log_probs = functional.log_softmax(
                outputs["governance_logits"][active] / temperature,
                dim=-1,
            )
            teacher_probs = functional.softmax(
                teacher_logits[active].to(dtype=student_log_probs.dtype) / temperature,
                dim=-1,
            )
            distillation = (
                functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                * temperature
                * temperature
            )
    total = (
        weights.governance * governance
        + weights.route * route
        + weights.taxonomy * taxonomy
        + weights.scalar * scalar
        + weights.distillation * distillation
        + weights.load_balance * balance
        + weights.trust_guard * trust_guard
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "loss_governance": float(governance.detach().cpu()),
        "loss_route": float(route.detach().cpu()),
        "loss_taxonomy": float(taxonomy.detach().cpu()),
        "loss_scalar": float(scalar.detach().cpu()),
        "loss_distillation": float(distillation.detach().cpu()),
        "loss_load_balance": float(balance.detach().cpu()),
        "loss_trust_guard": float(trust_guard.detach().cpu()),
    }
    return total, parts
