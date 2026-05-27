"""Evaluate a pyrrho-MoE Stage 0 checkpoint without retraining.

Run from project root:
    python scripts/eval_moe.py --checkpoint outputs/moe/stage0_route_proto/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/moe/stage0_route_proto/model.pt"),
        help="Stage 0 checkpoint from train_moe.py",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g3_alpha.yaml"),
        help="MoE YAML config",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Override data.moe_output_dir")
    p.add_argument("--split", choices=["eval", "test", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--force-route-ids", action="store_true", help="Evaluate with gold routes forced")
    p.add_argument("--output", type=Path, default=None, help="Default: <checkpoint_parent>/eval_report.json")
    return p.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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


def load_stage0_model(payload: dict) -> torch.nn.Module:
    model_kind = str(payload.get("model_kind", "tiny"))
    if model_kind == "tiny":
        model_cfg = TinyMoEConfig(**payload["config"])
        model = TinyMoEForGovernance(model_cfg)
    elif model_kind == "route_coupled":
        model_cfg = RouteCoupledMoEConfig(**payload["config"])
        model = RouteCoupledMoEForGovernance(model_cfg)
    elif model_kind == "route_coupled_token":
        model_cfg = TokenRouteCoupledMoEConfig(**payload["config"])
        model = TokenRouteCoupledMoEForGovernance(model_cfg)
    elif model_kind == "support_aggregating_token":
        model_cfg = SupportAggregatingMoEConfig(**payload["config"])
        model = SupportAggregatingMoEForGovernance(model_cfg)
    elif model_kind == "guarded_support_aggregating_token":
        model_cfg = GuardedSupportAggregatingMoEConfig(**payload["config"])
        model = GuardedSupportAggregatingMoEForGovernance(model_cfg)
    elif model_kind == "trust_guarded_support_aggregating_token":
        model_cfg = TrustGuardedSupportAggregatingMoEConfig(**payload["config"])
        model = TrustGuardedSupportAggregatingMoEForGovernance(model_cfg)
    else:
        raise ValueError(f"unknown checkpoint model_kind: {model_kind!r}")
    model.load_state_dict(payload["model_state_dict"])
    return model


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


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: MoELossWeights,
    governance_sample_policy: GovernanceSampleWeightPolicy | None,
    trust_guard_target_policy: TrustGuardTargetPolicy | None,
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
                governance_sample_weights=build_governance_sample_weights(
                    labels=batch["labels"],
                    route_ids=batch["route_ids"],
                    taxonomy_ids=batch["taxonomy_ids"],
                    policy=governance_sample_policy,
                ),
                trust_guard_targets=trust_guard_targets,
                trust_guard_weights=trust_guard_weights,
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
    )
    metrics["loss"] = float(mean(losses)) if losses else 0.0
    return metrics


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    stage_cfg = cfg.get("stage0", {})
    data_cfg = cfg.get("data", {})
    data_dir = (args.data_dir or Path(data_cfg.get("moe_output_dir", "data/moe_v8"))).resolve()
    output_path = args.output or (args.checkpoint.parent / "eval_report.json")

    payload = torch.load(args.checkpoint, map_location="cpu")
    model = load_stage0_model(payload)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_cfg = model.config

    vocab = MoEVocab.from_metadata(data_dir / "metadata.json")
    max_length = int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768)))
    max_query_length = int(stage_cfg.get("max_query_length", 96))
    max_sources = int(stage_cfg.get("max_sources", 8))
    max_source_length = int(stage_cfg.get("max_source_length", 192))
    batch_size = int(args.batch_size or stage_cfg.get("per_device_eval_batch_size", 128))
    weights = make_loss_weights(stage_cfg.get("loss_weights", {}))
    governance_sample_policy = make_governance_sample_policy(
        stage_cfg.get("governance_sample_weights", {}),
        vocab,
    )
    trust_guard_target_policy = make_trust_guard_target_policy(
        stage_cfg.get("trust_guard_targets", {}),
        vocab,
    )

    split_names = ["eval", "test"] if args.split == "both" else [args.split]
    report = {
        "checkpoint": str(args.checkpoint),
        "model_kind": str(payload.get("model_kind", "tiny")),
        "data_dir": str(data_dir),
        "force_route_ids": bool(args.force_route_ids),
        "splits": {},
        "route2id": vocab.route2id,
        "taxonomy_pattern2id": vocab.taxonomy_pattern2id,
    }

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data dir   : {data_dir}")
    print(f"Device     : {device}")
    for split in split_names:
        ds = MoEJsonlDataset(
            data_dir / f"{split}.jsonl",
            vocab=vocab,
            token_vocab_size=model_cfg.token_vocab_size,
            max_length=max_length,
            max_query_length=max_query_length,
            max_sources=max_sources,
            max_source_length=max_source_length,
            limit=args.max_samples,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_moe_batch)
        metrics = evaluate(
            model,
            loader,
            device,
            weights,
            governance_sample_policy,
            trust_guard_target_policy,
            force_route_ids=args.force_route_ids,
        )
        report["splits"][split] = {"n": len(ds), **metrics}
        g = metrics["governance"]
        print(
            f"{split:5s} n={len(ds)} acc={g['accuracy']:.4f} "
            f"FT={g['false_trustworthy_rate']:.4f} "
            f"route={metrics['route_accuracy']:.4f} "
            f"taxonomy={metrics['taxonomy_accuracy']:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote      : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
