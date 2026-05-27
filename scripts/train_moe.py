"""Train/evaluate the pyrrho-MoE Stage 0 tiny route prototype.

This is not the terminal 4B model. It is a fast PyTorch prototype that proves:
data loading, supervised route loss, top-1 expert selection, multitask heads,
eval reports, and expert traffic accounting.

Run from project root:
    python scripts/train_moe.py --config configs/moe/pyrrho_moe_g3_alpha.yaml
    python scripts/train_moe.py --max-train-samples 512 --max-eval-samples 256 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pyrrho.data import LABEL2ID
from pyrrho.manifest import write_manifest
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
from pyrrho.training import set_all_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g3_alpha.yaml"),
        help="MoE YAML config",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Override data.moe_output_dir")
    p.add_argument("--output-dir", type=Path, default=None, help="Override stage0.output_dir")
    p.add_argument("--epochs", type=int, default=None, help="Override stage0.num_train_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Override eval batch size")
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit train rows for smoke runs")
    p.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval rows for smoke runs")
    p.add_argument("--seed", type=int, default=None, help="Override stage0.seed")
    p.add_argument(
        "--teacher-logits-dir",
        type=Path,
        default=None,
        help="Optional sidecar dir with train/eval/test teacher governance logits keyed by id",
    )
    p.add_argument("--loss-governance", type=float, default=None, help="Override governance loss weight")
    p.add_argument("--loss-route", type=float, default=None, help="Override route loss weight")
    p.add_argument("--loss-taxonomy", type=float, default=None, help="Override taxonomy loss weight")
    p.add_argument("--loss-scalar", type=float, default=None, help="Override scalar loss weight")
    p.add_argument("--loss-distillation", type=float, default=None, help="Override distillation loss weight")
    p.add_argument("--loss-load-balance", type=float, default=None, help="Override router load-balance loss weight")
    p.add_argument(
        "--false-trustworthy-weight",
        type=float,
        default=None,
        help="Override governance CE weight for ABSTAIN/DISPUTED vs TRUSTWORTHY",
    )
    p.add_argument("--distillation-temperature", type=float, default=2.0)
    p.add_argument("--calibration-grid-size", type=int, default=66)
    p.add_argument(
        "--eval-compare-gold-routes",
        action="store_true",
        help="Also report governance metrics when eval/test routing is forced to gold route IDs",
    )
    p.add_argument("--dry-run", action="store_true", help="Build datasets/model but skip training")
    return p.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def make_loss_weights(raw: dict) -> MoELossWeights:
    return MoELossWeights(**{k: float(v) for k, v in (raw or {}).items()})


def _ids_for_names(mapping: dict[str, int], names: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"unknown MoE metadata names in sample-weight config: {missing}")
    return tuple(int(mapping[name]) for name in names)


def make_governance_sample_policy(
    raw: dict,
    vocab: MoEVocab,
) -> GovernanceSampleWeightPolicy | None:
    raw = raw or {}
    if not raw:
        return None
    support_taxonomy_pattern_weights = raw.get("support_taxonomy_pattern_weights", {}) or {}
    missing_weight_names = [
        str(name)
        for name in support_taxonomy_pattern_weights
        if str(name) not in vocab.taxonomy_pattern2id
    ]
    if missing_weight_names:
        raise ValueError(
            "unknown MoE taxonomy names in support_taxonomy_pattern_weights: "
            f"{missing_weight_names}"
        )
    return GovernanceSampleWeightPolicy(
        support_taxonomy_ids=_ids_for_names(
            vocab.taxonomy_pattern2id,
            list(raw.get("support_taxonomy_patterns", [])),
        ),
        support_trustworthy_weight=float(raw.get("support_trustworthy_weight", 1.0)),
        support_taxonomy_weights=tuple(
            (
                int(vocab.taxonomy_pattern2id[str(name)]),
                float(weight),
            )
            for name, weight in support_taxonomy_pattern_weights.items()
        ),
        ft_risk_route_ids=_ids_for_names(
            vocab.route2id,
            list(raw.get("ft_risk_routes", [])),
        ),
        ft_risk_route_non_trustworthy_weight=float(
            raw.get("ft_risk_route_non_trustworthy_weight", 1.0)
        ),
        ft_risk_taxonomy_ids=_ids_for_names(
            vocab.taxonomy_pattern2id,
            list(raw.get("ft_risk_taxonomy_patterns", [])),
        ),
        ft_risk_taxonomy_non_trustworthy_weight=float(
            raw.get("ft_risk_taxonomy_non_trustworthy_weight", 1.0)
        ),
        trustworthy_label_id=LABEL2ID["TRUSTWORTHY"],
    )


def make_trust_guard_target_policy(
    raw: dict,
    vocab: MoEVocab,
) -> TrustGuardTargetPolicy | None:
    raw = raw or {}
    if not raw:
        return None
    return TrustGuardTargetPolicy(
        positive_support_taxonomy_ids=_ids_for_names(
            vocab.taxonomy_pattern2id,
            list(raw.get("positive_support_taxonomy_patterns", [])),
        ),
        positive_support_weight=float(raw.get("positive_support_weight", 1.0)),
        negative_risk_route_ids=_ids_for_names(
            vocab.route2id,
            list(raw.get("negative_risk_routes", [])),
        ),
        negative_risk_route_weight=float(raw.get("negative_risk_route_weight", 1.0)),
        negative_risk_taxonomy_ids=_ids_for_names(
            vocab.taxonomy_pattern2id,
            list(raw.get("negative_risk_taxonomy_patterns", [])),
        ),
        negative_risk_taxonomy_weight=float(
            raw.get("negative_risk_taxonomy_weight", 1.0)
        ),
        trustworthy_label_id=LABEL2ID["TRUSTWORTHY"],
    )


def apply_loss_overrides(weights: MoELossWeights, args: argparse.Namespace) -> MoELossWeights:
    return MoELossWeights(
        governance=weights.governance
        if args.loss_governance is None
        else float(args.loss_governance),
        route=weights.route if args.loss_route is None else float(args.loss_route),
        taxonomy=weights.taxonomy
        if args.loss_taxonomy is None
        else float(args.loss_taxonomy),
        scalar=weights.scalar if args.loss_scalar is None else float(args.loss_scalar),
        distillation=weights.distillation
        if args.loss_distillation is None
        else float(args.loss_distillation),
        load_balance=weights.load_balance
        if args.loss_load_balance is None
        else float(args.loss_load_balance),
        false_trustworthy_weight=weights.false_trustworthy_weight
        if args.false_trustworthy_weight is None
        else float(args.false_trustworthy_weight),
        trust_guard=weights.trust_guard,
    )


def build_model_config_and_model(
    stage_cfg: dict,
    *,
    num_routes: int,
    num_taxonomy_patterns: int,
    num_scalar_targets: int,
) -> tuple[
    str,
    TinyMoEConfig
    | RouteCoupledMoEConfig
    | TokenRouteCoupledMoEConfig
    | SupportAggregatingMoEConfig
    | GuardedSupportAggregatingMoEConfig
    | TrustGuardedSupportAggregatingMoEConfig,
    torch.nn.Module,
]:
    model_kind = str(stage_cfg.get("model_kind", "tiny"))
    common = {
        "token_vocab_size": int(stage_cfg.get("token_vocab_size", 32768)),
        "hidden_size": int(stage_cfg.get("hidden_size", 256)),
        "expert_hidden_size": int(stage_cfg.get("expert_hidden_size", 512)),
        "num_routes": num_routes,
        "num_taxonomy_patterns": num_taxonomy_patterns,
        "num_scalar_targets": num_scalar_targets,
        "dropout": float(stage_cfg.get("dropout", 0.1)),
    }
    if model_kind == "tiny":
        model_cfg = TinyMoEConfig(**common)
        return model_kind, model_cfg, TinyMoEForGovernance(model_cfg)
    if model_kind == "route_coupled":
        model_cfg = RouteCoupledMoEConfig(
            **common,
            num_expert_layers=int(stage_cfg.get("num_expert_layers", 4)),
        )
        return model_kind, model_cfg, RouteCoupledMoEForGovernance(model_cfg)
    if model_kind == "route_coupled_token":
        model_cfg = TokenRouteCoupledMoEConfig(
            **common,
            num_expert_layers=int(stage_cfg.get("num_expert_layers", 4)),
            num_attention_heads=int(stage_cfg.get("num_attention_heads", 6)),
            num_key_value_heads=int(stage_cfg.get("num_key_value_heads", 2)),
            rope_theta=float(stage_cfg.get("rope_theta", 10000.0)),
        )
        return model_kind, model_cfg, TokenRouteCoupledMoEForGovernance(model_cfg)
    if model_kind == "support_aggregating_token":
        model_cfg = SupportAggregatingMoEConfig(
            **common,
            num_expert_layers=int(stage_cfg.get("num_expert_layers", 4)),
            num_attention_heads=int(stage_cfg.get("num_attention_heads", 6)),
            num_key_value_heads=int(stage_cfg.get("num_key_value_heads", 2)),
            rope_theta=float(stage_cfg.get("rope_theta", 10000.0)),
            max_query_length=int(stage_cfg.get("max_query_length", 96)),
            max_sources=int(stage_cfg.get("max_sources", 8)),
            max_source_length=int(stage_cfg.get("max_source_length", 192)),
        )
        return model_kind, model_cfg, SupportAggregatingMoEForGovernance(model_cfg)
    if model_kind == "guarded_support_aggregating_token":
        model_cfg = GuardedSupportAggregatingMoEConfig(
            **common,
            num_expert_layers=int(stage_cfg.get("num_expert_layers", 4)),
            num_attention_heads=int(stage_cfg.get("num_attention_heads", 6)),
            num_key_value_heads=int(stage_cfg.get("num_key_value_heads", 2)),
            rope_theta=float(stage_cfg.get("rope_theta", 10000.0)),
            max_query_length=int(stage_cfg.get("max_query_length", 96)),
            max_sources=int(stage_cfg.get("max_sources", 8)),
            max_source_length=int(stage_cfg.get("max_source_length", 192)),
            trust_penalty_scale=float(stage_cfg.get("trust_penalty_scale", 1.0)),
            trust_penalty_init_bias=float(stage_cfg.get("trust_penalty_init_bias", -2.0)),
        )
        return model_kind, model_cfg, GuardedSupportAggregatingMoEForGovernance(model_cfg)
    if model_kind == "trust_guarded_support_aggregating_token":
        model_cfg = TrustGuardedSupportAggregatingMoEConfig(
            **common,
            num_expert_layers=int(stage_cfg.get("num_expert_layers", 4)),
            num_attention_heads=int(stage_cfg.get("num_attention_heads", 6)),
            num_key_value_heads=int(stage_cfg.get("num_key_value_heads", 2)),
            rope_theta=float(stage_cfg.get("rope_theta", 10000.0)),
            max_query_length=int(stage_cfg.get("max_query_length", 96)),
            max_sources=int(stage_cfg.get("max_sources", 8)),
            max_source_length=int(stage_cfg.get("max_source_length", 192)),
            trust_guard_scale=float(stage_cfg.get("trust_guard_scale", 0.75)),
            trust_guard_init_bias=float(stage_cfg.get("trust_guard_init_bias", 2.0)),
            trust_guard_detach_candidate_logits=bool(
                stage_cfg.get("trust_guard_detach_candidate_logits", True)
            ),
        )
        return model_kind, model_cfg, TrustGuardedSupportAggregatingMoEForGovernance(
            model_cfg
        )
    raise ValueError(f"unknown Stage 0 model_kind: {model_kind!r}")


def model_forward(
    model: torch.nn.Module,
    batch: dict,
    *,
    force_route_ids: bool = False,
) -> dict[str, torch.Tensor]:
    kwargs = {
        "route_ids": batch["route_ids"],
        "force_route_ids": force_route_ids,
    }
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


def trust_guard_batch_targets(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    weights: MoELossWeights,
    policy: TrustGuardTargetPolicy | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if weights.trust_guard <= 0 or "trust_guard_logits" not in outputs:
        return None, None
    return build_trust_guard_targets(
        labels=batch["labels"],
        route_ids=batch["route_ids"],
        taxonomy_ids=batch["taxonomy_ids"],
        policy=policy,
    )


def teacher_sidecar_path(root: Path | None, split: str, *, required: bool) -> Path | None:
    if root is None:
        return None
    path = root / f"{split}.jsonl"
    if path.exists():
        return path
    if required:
        raise FileNotFoundError(f"teacher logits sidecar not found for {split}: {path}")
    print(f"Teacher logits  : missing optional {split} sidecar at {path}")
    return None


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: MoELossWeights,
    governance_sample_policy: GovernanceSampleWeightPolicy | None,
    trust_guard_target_policy: TrustGuardTargetPolicy | None,
    calibration_grid_size: int,
    distillation_temperature: float,
    force_route_ids: bool = False,
) -> dict:
    model.eval()
    losses: list[float] = []
    gov_logits = []
    route_logits = []
    taxonomy_logits = []
    labels = []
    route_labels = []
    taxonomy_labels = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model_forward(model, batch, force_route_ids=force_route_ids)
            trust_guard_targets, trust_guard_weights = trust_guard_batch_targets(
                outputs,
                batch,
                weights,
                trust_guard_target_policy,
            )
            _, parts = multitask_loss(
                outputs,
                labels=batch["labels"],
                route_ids=batch["route_ids"],
                taxonomy_ids=batch["taxonomy_ids"],
                scalar_targets=batch["scalar_targets"],
                scalar_mask=batch["scalar_mask"],
                weights=weights,
                teacher_logits=batch.get("teacher_logits"),
                teacher_mask=batch.get("teacher_mask"),
                governance_sample_weights=build_governance_sample_weights(
                    labels=batch["labels"],
                    route_ids=batch["route_ids"],
                    taxonomy_ids=batch["taxonomy_ids"],
                    policy=governance_sample_policy,
                ),
                trust_guard_targets=trust_guard_targets,
                trust_guard_weights=trust_guard_weights,
                distillation_temperature=distillation_temperature,
            )
            losses.append(parts["loss"])
            gov_logits.append(outputs["governance_logits"].cpu().numpy())
            route_logits.append(outputs["route_logits"].cpu().numpy())
            taxonomy_logits.append(outputs["taxonomy_logits"].cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            route_labels.append(batch["route_ids"].cpu().numpy())
            taxonomy_labels.append(batch["taxonomy_ids"].cpu().numpy())

    metrics = moe_eval_metrics(
        governance_logits=np.concatenate(gov_logits, axis=0),
        labels=np.concatenate(labels, axis=0),
        route_logits=np.concatenate(route_logits, axis=0),
        route_labels=np.concatenate(route_labels, axis=0),
        taxonomy_logits=np.concatenate(taxonomy_logits, axis=0),
        taxonomy_labels=np.concatenate(taxonomy_labels, axis=0),
        calibration_grid_size=calibration_grid_size,
    )
    metrics["loss"] = float(mean(losses)) if losses else 0.0
    metrics["force_route_ids"] = bool(force_route_ids)
    return metrics


def main() -> int:
    args = parse_args()
    start = time.time()
    cfg = load_config(args.config)
    stage_cfg = cfg.get("stage0", {})
    data_cfg = cfg.get("data", {})

    seed = int(args.seed if args.seed is not None else stage_cfg.get("seed", 42))
    set_all_seeds(seed)

    data_dir = (args.data_dir or Path(data_cfg.get("moe_output_dir", "data/moe_v8"))).resolve()
    output_dir = (args.output_dir or Path(stage_cfg.get("output_dir", "outputs/moe/stage0_route_proto"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = MoEVocab.from_metadata(data_dir / "metadata.json")
    model_kind, model_cfg, model = build_model_config_and_model(
        stage_cfg,
        num_routes=len(vocab.route2id),
        num_taxonomy_patterns=len(vocab.taxonomy_pattern2id),
        num_scalar_targets=len(vocab.scalar_fields),
    )
    max_length = int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768)))
    max_query_length = int(stage_cfg.get("max_query_length", 96))
    max_sources = int(stage_cfg.get("max_sources", 8))
    max_source_length = int(stage_cfg.get("max_source_length", 192))
    train_batch_size = int(args.batch_size or stage_cfg.get("per_device_train_batch_size", 64))
    eval_batch_size = int(args.eval_batch_size or stage_cfg.get("per_device_eval_batch_size", 128))
    epochs = int(args.epochs if args.epochs is not None else stage_cfg.get("num_train_epochs", 3))
    weights = apply_loss_overrides(make_loss_weights(stage_cfg.get("loss_weights", {})), args)
    governance_sample_policy = make_governance_sample_policy(
        stage_cfg.get("governance_sample_weights", {}),
        vocab,
    )
    trust_guard_target_policy = make_trust_guard_target_policy(
        stage_cfg.get("trust_guard_targets", {}),
        vocab,
    )

    print(f"Config          : {args.config}")
    print(f"Data dir        : {data_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Seed            : {seed}")
    print(f"Routes          : {len(vocab.route2id)}")
    print(f"Taxonomy patterns: {len(vocab.taxonomy_pattern2id)}")
    print(f"Scalar targets  : {len(vocab.scalar_fields)}")
    print(f"Model kind      : {model_kind}")
    print(
        "Loss weights    : "
        f"gov={weights.governance} route={weights.route} tax={weights.taxonomy} "
        f"scalar={weights.scalar} distill={weights.distillation} "
        f"balance={weights.load_balance} ftw={weights.false_trustworthy_weight} "
        f"trust_guard={weights.trust_guard}"
    )
    if governance_sample_policy is not None:
        print(f"Sample weights  : {governance_sample_policy}")
    if trust_guard_target_policy is not None:
        print(f"Trust guard     : {trust_guard_target_policy}")
    if args.teacher_logits_dir is not None:
        print(f"Teacher logits  : {args.teacher_logits_dir}")
        print(f"Distillation    : weight={weights.distillation} temp={args.distillation_temperature}")

    train_ds = MoEJsonlDataset(
        data_dir / "train.jsonl",
        vocab=vocab,
        token_vocab_size=model_cfg.token_vocab_size,
        max_length=max_length,
        max_query_length=max_query_length,
        max_sources=max_sources,
        max_source_length=max_source_length,
        teacher_logits_path=teacher_sidecar_path(
            args.teacher_logits_dir,
            "train",
            required=weights.distillation > 0,
        ),
        limit=args.max_train_samples,
    )
    eval_ds = MoEJsonlDataset(
        data_dir / "eval.jsonl",
        vocab=vocab,
        token_vocab_size=model_cfg.token_vocab_size,
        max_length=max_length,
        max_query_length=max_query_length,
        max_sources=max_sources,
        max_source_length=max_source_length,
        teacher_logits_path=teacher_sidecar_path(
            args.teacher_logits_dir,
            "eval",
            required=False,
        ),
        limit=args.max_eval_samples,
    )
    test_ds = MoEJsonlDataset(
        data_dir / "test.jsonl",
        vocab=vocab,
        token_vocab_size=model_cfg.token_vocab_size,
        max_length=max_length,
        max_query_length=max_query_length,
        max_sources=max_sources,
        max_source_length=max_source_length,
        teacher_logits_path=teacher_sidecar_path(
            args.teacher_logits_dir,
            "test",
            required=False,
        ),
        limit=args.max_eval_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_moe_batch,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_moe_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_moe_batch,
    )
    print(f"Rows            : train={len(train_ds)} eval={len(eval_ds)} test={len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Device          : {device}")
    print(f"Model params    : {params:,}")

    if args.dry_run:
        write_manifest(
            output_dir,
            args.config,
            seed=seed,
            pyrrho_repo=Path.cwd(),
            fitz_gov_repo=Path.cwd().parent / "fitz-gov",
            extra={
                "script": "train_moe.py",
                "dry_run": True,
                "model_kind": model_kind,
                "model_params": params,
            },
            start_time=start,
        )
        print("Dry run complete.")
        return 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(stage_cfg.get("weight_decay", 0.01)),
    )
    clip_norm = float(stage_cfg.get("gradient_clip_norm", 1.0))
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        step_losses = []
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            outputs = model_forward(model, batch)
            trust_guard_targets, trust_guard_weights = trust_guard_batch_targets(
                outputs,
                batch,
                weights,
                trust_guard_target_policy,
            )
            loss, parts = multitask_loss(
                outputs,
                labels=batch["labels"],
                route_ids=batch["route_ids"],
                taxonomy_ids=batch["taxonomy_ids"],
                scalar_targets=batch["scalar_targets"],
                scalar_mask=batch["scalar_mask"],
                weights=weights,
                teacher_logits=batch.get("teacher_logits"),
                teacher_mask=batch.get("teacher_mask"),
                governance_sample_weights=build_governance_sample_weights(
                    labels=batch["labels"],
                    route_ids=batch["route_ids"],
                    taxonomy_ids=batch["taxonomy_ids"],
                    policy=governance_sample_policy,
                ),
                trust_guard_targets=trust_guard_targets,
                trust_guard_weights=trust_guard_weights,
                distillation_temperature=args.distillation_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            step_losses.append(parts["loss"])

        eval_metrics = evaluate(
            model,
            eval_loader,
            device,
            weights,
            governance_sample_policy,
            trust_guard_target_policy,
            args.calibration_grid_size,
            args.distillation_temperature,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(mean(step_losses)) if step_losses else 0.0,
                "eval": eval_metrics,
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} "
            f"eval_acc={eval_metrics['governance']['accuracy']:.4f} "
            f"eval_ft={eval_metrics['governance']['false_trustworthy_rate']:.4f} "
            f"route_acc={eval_metrics['route_accuracy']:.4f} "
            f"tax_acc={eval_metrics['taxonomy_accuracy']:.4f}"
        )

    eval_metrics = evaluate(
        model,
        eval_loader,
        device,
        weights,
        governance_sample_policy,
        trust_guard_target_policy,
        args.calibration_grid_size,
        args.distillation_temperature,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        weights,
        governance_sample_policy,
        trust_guard_target_policy,
        args.calibration_grid_size,
        args.distillation_temperature,
    )
    eval_metrics_gold_routes = None
    test_metrics_gold_routes = None
    if args.eval_compare_gold_routes:
        eval_metrics_gold_routes = evaluate(
            model,
            eval_loader,
            device,
            weights,
            governance_sample_policy,
            trust_guard_target_policy,
            args.calibration_grid_size,
            args.distillation_temperature,
            force_route_ids=True,
        )
        test_metrics_gold_routes = evaluate(
            model,
            test_loader,
            device,
            weights,
            governance_sample_policy,
            trust_guard_target_policy,
            args.calibration_grid_size,
            args.distillation_temperature,
            force_route_ids=True,
        )
    final = {
        "stage": str(stage_cfg.get("target", "stage0_tiny_route_prototype")),
        "model_kind": model_kind,
        "seed": seed,
        "stage0_params": params,
        "model_params": params,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "test_rows": len(test_ds),
        "history": history,
        "eval": eval_metrics,
        "test": test_metrics,
        "eval_gold_routes": eval_metrics_gold_routes,
        "test_gold_routes": test_metrics_gold_routes,
        "teacher_logits_dir": str(args.teacher_logits_dir)
        if args.teacher_logits_dir is not None
        else None,
        "loss_weights": {
            "governance": weights.governance,
            "route": weights.route,
            "taxonomy": weights.taxonomy,
            "scalar": weights.scalar,
            "distillation": weights.distillation,
            "distillation_temperature": args.distillation_temperature,
            "load_balance": weights.load_balance,
            "false_trustworthy_weight": weights.false_trustworthy_weight,
            "trust_guard": weights.trust_guard,
        },
        "governance_sample_weights": stage_cfg.get("governance_sample_weights", {}),
        "trust_guard_targets": stage_cfg.get("trust_guard_targets", {}),
        "route2id": vocab.route2id,
        "taxonomy_pattern2id": vocab.taxonomy_pattern2id,
        "scalar_fields": list(vocab.scalar_fields),
    }
    (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": model_kind,
            "config": model_cfg.__dict__,
        },
        output_dir / "model.pt",
    )
    write_manifest(
        output_dir,
        args.config,
        seed=seed,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=Path.cwd().parent / "fitz-gov",
        extra={"script": "train_moe.py", "model_kind": model_kind, "model_params": params},
        start_time=start,
    )
    if test_metrics_gold_routes is not None:
        gold = test_metrics_gold_routes["governance"]
        gold_cal = test_metrics_gold_routes["governance_calibrated"]
        print(
            "Gold-route test : "
            f"acc={gold['accuracy']:.4f} "
            f"ft={gold['false_trustworthy_rate']:.4f} "
            f"cal_acc={gold_cal['accuracy']:.4f} "
            f"cal_ft={gold_cal['false_trustworthy_rate']:.4f}"
        )
    print(f"Wrote metrics    : {output_dir / 'final_metrics.json'}")
    print(f"Wrote checkpoint : {output_dir / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
