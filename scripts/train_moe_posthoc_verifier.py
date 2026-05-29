"""Train a post-hoc verifier over frozen Stage 0 MoE candidate outputs.

This does not update the MoE trunk. It loads a frozen checkpoint, collects
candidate logits/features for train/eval/test, trains a binary verifier only on
rows the frozen model predicts as TRUSTWORTHY, selects a verifier threshold on
eval, and reports the held-out test tradeoff.

Run from project root:
    python scripts/train_moe_posthoc_verifier.py \
      --checkpoint outputs/moe/stage0_7_support_aggregation_g3_3seed/seed_42/model.pt \
      --config configs/moe/pyrrho_moe_stage0_7_support_aggregation.yaml \
      --output-dir outputs/moe/stage0_7_posthoc_verifier_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pyrrho.data import ID2LABEL, LABEL2ID
from pyrrho.metrics import compute_classification_metrics, gated_predictions
from pyrrho.moe.data import MoEJsonlDataset, MoEVocab, collate_moe_batch
from pyrrho.moe.metrics import route_accuracy, taxonomy_accuracy
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

TRUSTWORTHY_ID = LABEL2ID["TRUSTWORTHY"]

SUPPORT_PATTERNS = {
    "consistent_chain",
    "multi_source_corroboration",
    "quantitative_consensus",
    "expert_consensus",
}
FT_RISK_ROUTES = {"science_medicine", "general_commonsense", "technology_computing"}
FT_RISK_PATTERNS = {
    "factual_contradiction",
    "evidence_absent",
    "partial_overlap",
    "numerical_conflict",
    "temporal_conflict",
    "temporal_mismatch",
    "wrong_entity",
    "wrong_specificity",
    "scope_conflict",
}


@dataclass(frozen=True)
class FrozenSplit:
    ids: list[str]
    features: np.ndarray
    governance_logits: np.ndarray
    route_logits: np.ndarray
    taxonomy_logits: np.ndarray
    labels: np.ndarray
    route_labels: np.ndarray
    taxonomy_labels: np.ndarray
    route_names: list[str]
    taxonomy_names: list[str]
    base_preds: np.ndarray
    runner_up_non_t: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/moe/stage0_7_support_aggregation_g3_3seed/seed_42/model.pt"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_stage0_7_support_aggregation.yaml"),
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/moe/stage0_7_posthoc_verifier_seed42"),
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--base-threshold", type=float, default=None)
    p.add_argument("--target-ft", type=float, default=0.025)
    p.add_argument("--max-accuracy-drop", type=float, default=0.015)
    p.add_argument(
        "--max-support-accuracy-drop",
        type=float,
        default=None,
        help="Optional eval support-slice accuracy floor relative to verifier baseline",
    )
    p.add_argument(
        "--selection-objective",
        choices=["accuracy", "support"],
        default="accuracy",
        help="How to choose among thresholds satisfying the configured constraints",
    )
    p.add_argument("--threshold-grid-size", type=int, default=101)
    p.add_argument("--verifier-kind", choices=["hgb", "logistic"], default="hgb")
    p.add_argument("--positive-support-weight", type=float, default=1.6)
    p.add_argument("--negative-risk-route-weight", type=float, default=1.4)
    p.add_argument("--negative-risk-taxonomy-weight", type=float, default=1.8)
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def load_stage0_model(payload: dict[str, Any]) -> torch.nn.Module:
    model_kind = str(payload.get("model_kind", "tiny"))
    if model_kind == "tiny":
        cfg = TinyMoEConfig(**payload["config"])
        model = TinyMoEForGovernance(cfg)
    elif model_kind == "route_coupled":
        cfg = RouteCoupledMoEConfig(**payload["config"])
        model = RouteCoupledMoEForGovernance(cfg)
    elif model_kind == "route_coupled_token":
        cfg = TokenRouteCoupledMoEConfig(**payload["config"])
        model = TokenRouteCoupledMoEForGovernance(cfg)
    elif model_kind == "support_aggregating_token":
        cfg = SupportAggregatingMoEConfig(**payload["config"])
        model = SupportAggregatingMoEForGovernance(cfg)
    elif model_kind == "guarded_support_aggregating_token":
        cfg = GuardedSupportAggregatingMoEConfig(**payload["config"])
        model = GuardedSupportAggregatingMoEForGovernance(cfg)
    elif model_kind == "trust_guarded_support_aggregating_token":
        cfg = TrustGuardedSupportAggregatingMoEConfig(**payload["config"])
        model = TrustGuardedSupportAggregatingMoEForGovernance(cfg)
    else:
        raise ValueError(f"unknown checkpoint model_kind: {model_kind!r}")
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    kwargs = {"route_ids": batch["route_ids"]}
    if bool(getattr(model.config, "uses_support_aggregation", False)):
        kwargs.update(
            {
                "query_input_ids": batch["query_input_ids"],
                "query_attention_mask": batch["query_attention_mask"],
                "source_input_ids": batch["source_input_ids"],
                "source_attention_mask": batch["source_attention_mask"],
                "source_valid_mask": batch["source_valid_mask"],
            }
        )
    return model(batch["input_ids"], batch["attention_mask"], **kwargs)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1, keepdims=True)


def one_hot(ids: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((ids.shape[0], width), dtype=np.float32)
    out[np.arange(ids.shape[0]), ids.astype(int)] = 1.0
    return out


def build_features(
    *,
    governance_logits: np.ndarray,
    route_logits: np.ndarray,
    taxonomy_logits: np.ndarray,
    scalar_preds: np.ndarray,
) -> np.ndarray:
    governance_probs = softmax(governance_logits)
    route_probs = softmax(route_logits)
    taxonomy_probs = softmax(taxonomy_logits)
    route_pred = route_logits.argmax(axis=-1)
    taxonomy_pred = taxonomy_logits.argmax(axis=-1)
    p_t = governance_probs[:, [TRUSTWORTHY_ID]]
    non_t = governance_probs[:, [0, 1]]
    trust_margin = p_t - non_t.max(axis=1, keepdims=True)
    disputed_minus_abstain = governance_logits[:, [1]] - governance_logits[:, [0]]
    return np.concatenate(
        [
            governance_logits,
            governance_probs,
            route_logits,
            route_probs,
            taxonomy_logits,
            taxonomy_probs,
            scalar_preds,
            p_t,
            trust_margin,
            disputed_minus_abstain,
            entropy(governance_probs),
            entropy(route_probs),
            entropy(taxonomy_probs),
            one_hot(route_pred, route_logits.shape[1]),
            one_hot(taxonomy_pred, taxonomy_logits.shape[1]),
        ],
        axis=1,
    ).astype(np.float32)


def collect_split(
    *,
    split: str,
    model: torch.nn.Module,
    data_dir: Path,
    vocab: MoEVocab,
    route_names_by_id: dict[int, str],
    taxonomy_names_by_id: dict[int, str],
    token_vocab_size: int,
    max_length: int,
    max_query_length: int,
    max_sources: int,
    max_source_length: int,
    batch_size: int,
    limit: int | None,
    base_threshold: float,
    device: torch.device,
) -> FrozenSplit:
    ds = MoEJsonlDataset(
        data_dir / f"{split}.jsonl",
        vocab=vocab,
        token_vocab_size=token_vocab_size,
        max_length=max_length,
        max_query_length=max_query_length,
        max_sources=max_sources,
        max_source_length=max_source_length,
        limit=limit,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_moe_batch)
    ids: list[str] = []
    governance_logits: list[np.ndarray] = []
    route_logits: list[np.ndarray] = []
    taxonomy_logits: list[np.ndarray] = []
    scalar_preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    route_labels: list[np.ndarray] = []
    taxonomy_labels: list[np.ndarray] = []
    model = model.to(device)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model_forward(model, batch)
            ids.extend(str(row_id) for row_id in raw_batch["ids"])
            governance_logits.append(outputs["governance_logits"].detach().cpu().numpy())
            route_logits.append(outputs["route_logits"].detach().cpu().numpy())
            taxonomy_logits.append(outputs["taxonomy_logits"].detach().cpu().numpy())
            scalar_preds.append(outputs["scalar_preds"].detach().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
            route_labels.append(batch["route_ids"].detach().cpu().numpy())
            taxonomy_labels.append(batch["taxonomy_ids"].detach().cpu().numpy())

    gov = np.concatenate(governance_logits, axis=0)
    route = np.concatenate(route_logits, axis=0)
    taxonomy = np.concatenate(taxonomy_logits, axis=0)
    scalars = np.concatenate(scalar_preds, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    route_arr = np.concatenate(route_labels, axis=0)
    taxonomy_arr = np.concatenate(taxonomy_labels, axis=0)
    base_preds = gated_predictions(gov, base_threshold)
    runner_up_non_t = np.where(gov[:, 0] >= gov[:, 1], 0, 1)
    return FrozenSplit(
        ids=ids,
        features=build_features(
            governance_logits=gov,
            route_logits=route,
            taxonomy_logits=taxonomy,
            scalar_preds=scalars,
        ),
        governance_logits=gov,
        route_logits=route,
        taxonomy_logits=taxonomy,
        labels=labels_arr,
        route_labels=route_arr,
        taxonomy_labels=taxonomy_arr,
        route_names=[route_names_by_id.get(int(v), str(v)) for v in route_arr],
        taxonomy_names=[taxonomy_names_by_id.get(int(v), str(v)) for v in taxonomy_arr],
        base_preds=base_preds,
        runner_up_non_t=runner_up_non_t,
    )


def build_train_weights(
    split: FrozenSplit,
    *,
    positive_support_weight: float,
    negative_risk_route_weight: float,
    negative_risk_taxonomy_weight: float,
) -> np.ndarray:
    weights = np.ones(split.labels.shape[0], dtype=np.float32)
    trustworthy = split.labels == TRUSTWORTHY_ID
    support_positive = trustworthy & np.asarray(
        [name in SUPPORT_PATTERNS for name in split.taxonomy_names],
        dtype=bool,
    )
    risk_route_negative = ~trustworthy & np.asarray(
        [name in FT_RISK_ROUTES for name in split.route_names],
        dtype=bool,
    )
    risk_taxonomy_negative = ~trustworthy & np.asarray(
        [name in FT_RISK_PATTERNS for name in split.taxonomy_names],
        dtype=bool,
    )
    weights[support_positive] *= float(positive_support_weight)
    weights[risk_route_negative] *= float(negative_risk_route_weight)
    weights[risk_taxonomy_negative] *= float(negative_risk_taxonomy_weight)
    return weights


def make_verifier(kind: str):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs"),
        )
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=42,
    )


def fit_verifier(verifier, features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> None:
    if hasattr(verifier, "named_steps") and "logisticregression" in verifier.named_steps:
        verifier.fit(features, labels, logisticregression__sample_weight=weights)
        return
    verifier.fit(features, labels, sample_weight=weights)


def guarded_predictions(
    split: FrozenSplit,
    accept_scores: np.ndarray,
    verifier_threshold: float,
) -> np.ndarray:
    preds = split.base_preds.copy()
    reject_mask = (preds == TRUSTWORTHY_ID) & (accept_scores < verifier_threshold)
    preds[reject_mask] = split.runner_up_non_t[reject_mask]
    return preds


def group_metrics(
    *,
    labels: np.ndarray,
    preds: np.ndarray,
    groups: list[str],
) -> dict[str, dict[str, float | int]]:
    groups_arr = np.asarray(groups)
    out = {}
    for group in sorted(set(groups)):
        mask = groups_arr == group
        metrics = compute_classification_metrics(preds[mask], labels[mask])
        out[group] = {
            "n": int(mask.sum()),
            "accuracy": metrics["accuracy"],
            "false_trustworthy_rate": metrics["false_trustworthy_rate"],
            "trustworthy_recall": metrics["recall_trustworthy"],
        }
    return out


def evaluate_predictions(
    split: FrozenSplit,
    preds: np.ndarray,
) -> dict[str, Any]:
    return {
        "governance": compute_classification_metrics(preds, split.labels),
        "route_accuracy": route_accuracy(split.route_logits.argmax(axis=-1), split.route_labels),
        "taxonomy_accuracy": taxonomy_accuracy(
            split.taxonomy_logits.argmax(axis=-1),
            split.taxonomy_labels,
        ),
        "by_route": group_metrics(labels=split.labels, preds=preds, groups=split.route_names),
        "by_taxonomy": group_metrics(
            labels=split.labels,
            preds=preds,
            groups=split.taxonomy_names,
        ),
    }


def support_mask(split: FrozenSplit) -> np.ndarray:
    return np.asarray([name in SUPPORT_PATTERNS for name in split.taxonomy_names], dtype=bool)


def risk_mask(split: FrozenSplit) -> np.ndarray:
    routes = np.asarray([name in FT_RISK_ROUTES for name in split.route_names], dtype=bool)
    patterns = np.asarray([name in FT_RISK_PATTERNS for name in split.taxonomy_names], dtype=bool)
    return routes | patterns


def slice_accuracy(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((preds[mask] == labels[mask]).mean())


def slice_false_trustworthy(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    active_labels = labels[mask]
    active_preds = preds[mask]
    non_trustworthy = active_labels != TRUSTWORTHY_ID
    if not bool(non_trustworthy.any()):
        return 0.0
    return float((active_preds[non_trustworthy] == TRUSTWORTHY_ID).mean())


def select_threshold(
    *,
    split: FrozenSplit,
    accept_scores: np.ndarray,
    target_ft: float,
    max_accuracy_drop: float,
    max_support_accuracy_drop: float | None,
    selection_objective: str,
    grid_size: int,
) -> dict[str, Any]:
    baseline = compute_classification_metrics(split.base_preds, split.labels)
    support = support_mask(split)
    risk = risk_mask(split)
    baseline_support_accuracy = slice_accuracy(split.base_preds, split.labels, support)
    baseline_risk_ft = slice_false_trustworthy(split.base_preds, split.labels, risk)
    rows = []
    for threshold in np.linspace(0.0, 1.0, grid_size):
        preds = guarded_predictions(split, accept_scores, float(threshold))
        metrics = compute_classification_metrics(preds, split.labels)
        support_accuracy = slice_accuracy(preds, split.labels, support)
        risk_ft = slice_false_trustworthy(preds, split.labels, risk)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                "support_accuracy": support_accuracy,
                "support_accuracy_drop": baseline_support_accuracy - support_accuracy,
                "risk_false_trustworthy_rate": risk_ft,
                "risk_false_trustworthy_delta": risk_ft - baseline_risk_ft,
                "rejected_candidate_trustworthy": int(
                    ((split.base_preds == TRUSTWORTHY_ID) & (accept_scores < threshold)).sum()
                ),
            }
        )
    accuracy_floor = baseline["accuracy"] - max_accuracy_drop
    support_accuracy_floor = None
    if max_support_accuracy_drop is not None:
        support_accuracy_floor = baseline_support_accuracy - float(max_support_accuracy_drop)
    passing = [
        row
        for row in rows
        if row["false_trustworthy_rate"] <= target_ft
        and row["accuracy"] >= accuracy_floor
        and (
            support_accuracy_floor is None
            or row["support_accuracy"] >= support_accuracy_floor
        )
    ]
    if passing:
        if selection_objective == "support":
            selected = max(
                passing,
                key=lambda row: (
                    row["support_accuracy"],
                    row["accuracy"],
                    -row["false_trustworthy_rate"],
                ),
            )
        else:
            selected = max(
                passing,
                key=lambda row: (
                    row["accuracy"],
                    row["support_accuracy"],
                    -row["false_trustworthy_rate"],
                ),
            )
        reason = "target_ft_and_accuracy_floor"
    else:
        ft_passing = [row for row in rows if row["false_trustworthy_rate"] <= target_ft]
        if ft_passing:
            if selection_objective == "support":
                selected = max(
                    ft_passing,
                    key=lambda row: (row["support_accuracy"], row["accuracy"]),
                )
            else:
                selected = max(
                    ft_passing,
                    key=lambda row: (row["accuracy"], row["support_accuracy"]),
                )
            reason = "target_ft_only"
        else:
            selected = min(rows, key=lambda row: (row["false_trustworthy_rate"], -row["accuracy"]))
            reason = "min_false_trustworthy"
    return {
        "baseline": baseline,
        "baseline_support_accuracy": baseline_support_accuracy,
        "baseline_risk_false_trustworthy_rate": baseline_risk_ft,
        "target_ft": target_ft,
        "max_accuracy_drop": max_accuracy_drop,
        "accuracy_floor": accuracy_floor,
        "max_support_accuracy_drop": max_support_accuracy_drop,
        "support_accuracy_floor": support_accuracy_floor,
        "selection_objective": selection_objective,
        "selected": selected,
        "selection_reason": reason,
        "sweep": rows,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def markdown_report(report: dict[str, Any]) -> str:
    def metric_line(name: str, row: dict[str, Any]) -> str:
        gov = row["governance"]
        return (
            f"| {name} | {format_pct(gov['accuracy'])} | "
            f"{format_pct(gov['false_trustworthy_rate'])} | "
            f"{format_pct(gov['recall_trustworthy'])} |"
        )

    lines = [
        "# MoE Post-Hoc Verifier Report",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Verifier kind: `{report['verifier_kind']}`",
        f"- Base TRUSTWORTHY threshold: **{report['base_threshold']:.4f}**",
        f"- Selected verifier threshold: **{report['selected_threshold']:.4f}**",
        f"- Selection reason: `{report['selection_reason']}`",
        f"- Selection objective: `{report['selection_objective']}`",
        "",
        "| Split / mode | Accuracy | FT | T recall |",
        "|---|---:|---:|---:|",
        metric_line("eval baseline", report["eval"]["baseline"]),
        metric_line("eval guarded", report["eval"]["guarded"]),
        metric_line("test baseline", report["test"]["baseline"]),
        metric_line("test guarded", report["test"]["guarded"]),
        "",
        "## Key Test Slices",
        "",
        "| Slice | Baseline Acc | Guarded Acc | Baseline FT | Guarded FT |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline_tax = report["test"]["baseline"]["by_taxonomy"]
    guarded_tax = report["test"]["guarded"]["by_taxonomy"]
    for name in [
        "consistent_chain",
        "multi_source_corroboration",
        "quantitative_consensus",
        "expert_consensus",
        "factual_contradiction",
        "partial_overlap",
        "evidence_absent",
        "wrong_entity",
    ]:
        if name not in baseline_tax or name not in guarded_tax:
            continue
        b = baseline_tax[name]
        g = guarded_tax[name]
        lines.append(
            f"| {name} | {format_pct(float(b['accuracy']))} | "
            f"{format_pct(float(g['accuracy']))} | "
            f"{format_pct(float(b['false_trustworthy_rate']))} | "
            f"{format_pct(float(g['false_trustworthy_rate']))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = cfg.get("stage0", {})
    data_cfg = cfg.get("data", {})
    data_dir = (args.data_dir or Path(data_cfg.get("moe_output_dir", "data/moe_v8"))).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu")
    model = load_stage0_model(payload)
    model_cfg = model.config
    vocab = MoEVocab.from_metadata(data_dir / "metadata.json")
    route_names_by_id = invert_mapping(vocab.route2id)
    taxonomy_names_by_id = invert_mapping(vocab.taxonomy_pattern2id)
    max_length = int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768)))
    max_query_length = int(stage_cfg.get("max_query_length", 96))
    max_sources = int(stage_cfg.get("max_sources", 8))
    max_source_length = int(stage_cfg.get("max_source_length", 192))
    batch_size = int(args.batch_size or stage_cfg.get("per_device_eval_batch_size", 128))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_threshold = args.base_threshold
    metrics_path = args.checkpoint.parent / "final_metrics.json"
    if base_threshold is None and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        base_threshold = float(metrics["eval"]["governance_calibrated"]["threshold"])
    if base_threshold is None:
        base_threshold = 0.34

    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Data dir        : {data_dir}")
    print(f"Device          : {device}")
    print(f"Base threshold  : {base_threshold:.4f}")

    splits = {
        "train": collect_split(
            split="train",
            model=model,
            data_dir=data_dir,
            vocab=vocab,
            route_names_by_id=route_names_by_id,
            taxonomy_names_by_id=taxonomy_names_by_id,
            token_vocab_size=model_cfg.token_vocab_size,
            max_length=max_length,
            max_query_length=max_query_length,
            max_sources=max_sources,
            max_source_length=max_source_length,
            batch_size=batch_size,
            limit=args.max_train_samples,
            base_threshold=base_threshold,
            device=device,
        ),
        "eval": collect_split(
            split="eval",
            model=model,
            data_dir=data_dir,
            vocab=vocab,
            route_names_by_id=route_names_by_id,
            taxonomy_names_by_id=taxonomy_names_by_id,
            token_vocab_size=model_cfg.token_vocab_size,
            max_length=max_length,
            max_query_length=max_query_length,
            max_sources=max_sources,
            max_source_length=max_source_length,
            batch_size=batch_size,
            limit=args.max_eval_samples,
            base_threshold=base_threshold,
            device=device,
        ),
        "test": collect_split(
            split="test",
            model=model,
            data_dir=data_dir,
            vocab=vocab,
            route_names_by_id=route_names_by_id,
            taxonomy_names_by_id=taxonomy_names_by_id,
            token_vocab_size=model_cfg.token_vocab_size,
            max_length=max_length,
            max_query_length=max_query_length,
            max_sources=max_sources,
            max_source_length=max_source_length,
            batch_size=batch_size,
            limit=args.max_test_samples,
            base_threshold=base_threshold,
            device=device,
        ),
    }

    train = splits["train"]
    candidate_mask = train.base_preds == TRUSTWORTHY_ID
    if int(candidate_mask.sum()) < 2:
        raise ValueError("not enough candidate TRUSTWORTHY train rows for verifier")
    y_train = (train.labels[candidate_mask] == TRUSTWORTHY_ID).astype(int)
    train_weights = build_train_weights(
        train,
        positive_support_weight=args.positive_support_weight,
        negative_risk_route_weight=args.negative_risk_route_weight,
        negative_risk_taxonomy_weight=args.negative_risk_taxonomy_weight,
    )[candidate_mask]
    verifier = make_verifier(args.verifier_kind)
    fit_verifier(verifier, train.features[candidate_mask], y_train, train_weights)
    joblib.dump(verifier, args.output_dir / "verifier.joblib")

    accept_scores = {
        split: verifier.predict_proba(data.features)[:, 1]
        for split, data in splits.items()
    }
    selection = select_threshold(
        split=splits["eval"],
        accept_scores=accept_scores["eval"],
        target_ft=float(args.target_ft),
        max_accuracy_drop=float(args.max_accuracy_drop),
        max_support_accuracy_drop=args.max_support_accuracy_drop,
        selection_objective=args.selection_objective,
        grid_size=int(args.threshold_grid_size),
    )
    selected_threshold = float(selection["selected"]["threshold"])
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "data_dir": str(data_dir),
        "verifier_kind": args.verifier_kind,
        "base_threshold": float(base_threshold),
        "target_ft": float(args.target_ft),
        "max_accuracy_drop": float(args.max_accuracy_drop),
        "max_support_accuracy_drop": args.max_support_accuracy_drop,
        "selection_objective": args.selection_objective,
        "selected_threshold": selected_threshold,
        "selection_reason": selection["selection_reason"],
        "train": {
            "rows": len(train.ids),
            "candidate_trustworthy_rows": int(candidate_mask.sum()),
            "candidate_trustworthy_positive_rate": float(y_train.mean()),
        },
        "threshold_selection": selection,
        "eval": {},
        "test": {},
    }
    for split_name in ["eval", "test"]:
        split = splits[split_name]
        guarded = guarded_predictions(split, accept_scores[split_name], selected_threshold)
        report[split_name] = {
            "rows": len(split.ids),
            "baseline": evaluate_predictions(split, split.base_preds),
            "guarded": evaluate_predictions(split, guarded),
            "rejected_candidate_trustworthy": int(
                ((split.base_preds == TRUSTWORTHY_ID) & (accept_scores[split_name] < selected_threshold)).sum()
            ),
        }

    case_rows = []
    test = splits["test"]
    test_guarded = guarded_predictions(test, accept_scores["test"], selected_threshold)
    for idx, row_id in enumerate(test.ids):
        case_rows.append(
            {
                "id": row_id,
                "label": ID2LABEL[int(test.labels[idx])],
                "baseline_pred": ID2LABEL[int(test.base_preds[idx])],
                "guarded_pred": ID2LABEL[int(test_guarded[idx])],
                "accept_score": float(accept_scores["test"][idx]),
                "route": test.route_names[idx],
                "taxonomy": test.taxonomy_names[idx],
            }
        )
    write_jsonl(args.output_dir / "test_predictions.jsonl", case_rows)
    (args.output_dir / "verifier_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "verifier_report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(
        "Selected        : "
        f"threshold={selected_threshold:.4f} reason={selection['selection_reason']}"
    )
    for split_name in ["eval", "test"]:
        base = report[split_name]["baseline"]["governance"]
        guarded = report[split_name]["guarded"]["governance"]
        print(
            f"{split_name:5s} baseline acc={base['accuracy']:.4f} "
            f"ft={base['false_trustworthy_rate']:.4f}; "
            f"guarded acc={guarded['accuracy']:.4f} "
            f"ft={guarded['false_trustworthy_rate']:.4f}"
        )
    print(f"Wrote report    : {args.output_dir / 'verifier_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
