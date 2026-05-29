"""Analyze per-row failures for Stage 0/0.5 pyrrho-MoE checkpoints.

Run from project root:
    python scripts/analyze_moe_failures.py \
      --run-root outputs/moe/stage0_5_route_coupled_g3_3seed \
      --config configs/moe/pyrrho_moe_stage0_5_route_coupled.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pyrrho.data import ID2LABEL
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/moe/stage0_5_route_coupled_g3_3seed"),
        help="Directory containing seed_*/model.pt and seed_*/final_metrics.json",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_stage0_5_route_coupled.yaml"),
        help="MoE YAML config used for max length and batch size defaults",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Override data.moe_output_dir")
    p.add_argument("--split", choices=["eval", "test"], default="test")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--force-route-ids",
        action="store_true",
        help="Force gold route IDs during model forward pass",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-root>/failure_analysis_<split>",
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_model(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu")
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
        raise ValueError(f"unknown checkpoint model_kind in {checkpoint}: {model_kind!r}")
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def model_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    force_route_ids: bool,
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


def run_dirs(root: Path, seeds: list[int] | None) -> list[tuple[int, Path]]:
    if seeds:
        out = [(seed, root / f"seed_{seed}") for seed in seeds]
    else:
        out = []
        for path in sorted(root.glob("seed_*")):
            try:
                seed = int(path.name.removeprefix("seed_"))
            except ValueError:
                continue
            out.append((seed, path))
    missing = [str(path) for _, path in out if not (path / "model.pt").exists()]
    if missing:
        raise FileNotFoundError(f"missing model.pt under: {missing}")
    return out


def threshold_for_split(metrics_path: Path, split: str) -> float:
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(raw[split]["governance_calibrated"]["threshold"])


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def row_metadata(ds: MoEJsonlDataset, route_names: dict[int, str], taxonomy_names: dict[int, str]) -> dict[str, Any]:
    out = {}
    for row in ds.rows:
        row_id = str(row["id"])
        label_id = int(row["label_id"])
        route_id = int(row["route_id"])
        taxonomy_id = int(row["taxonomy_pattern_id"])
        out[row_id] = {
            "id": row_id,
            "label_id": label_id,
            "label": ID2LABEL[label_id],
            "route_id": route_id,
            "route": route_names.get(route_id, str(route_id)),
            "taxonomy_id": taxonomy_id,
            "taxonomy": taxonomy_names.get(taxonomy_id, str(taxonomy_id)),
            "difficulty": row.get("difficulty"),
            "dataset_version": row.get("dataset_version"),
            "query": row.get("query"),
        }
    return out


def collect_seed_predictions(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    force_route_ids: bool,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    model = model.to(device)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model_forward(model, batch, force_route_ids=force_route_ids)
            governance_logits = outputs["governance_logits"].detach().cpu().numpy()
            probs = torch.softmax(outputs["governance_logits"], dim=-1).detach().cpu().numpy()
            calibrated_preds = gated_predictions(governance_logits, threshold)
            route_preds = outputs["route_logits"].argmax(dim=-1).detach().cpu().numpy()
            taxonomy_preds = outputs["taxonomy_logits"].argmax(dim=-1).detach().cpu().numpy()
            selected_routes = outputs["selected_routes"].detach().cpu().numpy()
            trust_guard_probs = None
            trust_guard_penalties = None
            if "trust_guard_logits" in outputs:
                trust_guard_probs = (
                    torch.sigmoid(outputs["trust_guard_logits"]).detach().cpu().numpy()
                )
            if "trust_guard_penalty" in outputs:
                trust_guard_penalties = outputs["trust_guard_penalty"].detach().cpu().numpy()

            for i, row_id in enumerate(raw_batch["ids"]):
                row = {
                    "governance_logits": [float(v) for v in governance_logits[i].tolist()],
                    "governance_probs": [float(v) for v in probs[i].tolist()],
                    "pred_id": int(calibrated_preds[i]),
                    "raw_pred_id": int(governance_logits[i].argmax()),
                    "route_pred_id": int(route_preds[i]),
                    "taxonomy_pred_id": int(taxonomy_preds[i]),
                    "selected_route_id": int(selected_routes[i]),
                }
                if trust_guard_probs is not None:
                    row["trust_guard_accept_probability"] = float(trust_guard_probs[i])
                if trust_guard_penalties is not None:
                    row["trust_guard_penalty"] = float(trust_guard_penalties[i])
                out[str(row_id)] = row
    return out


def confusion_counts(golds: list[int], preds: list[int], names: dict[int, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for gold, pred in zip(golds, preds, strict=True):
        counts[f"{names.get(gold, str(gold))} -> {names.get(pred, str(pred))}"] += 1
    return dict(counts.most_common())


def group_metrics(
    *,
    row_ids: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    groups: list[str],
) -> dict[str, dict[str, float | int]]:
    out = {}
    groups_arr = np.asarray(groups)
    for group in sorted(set(groups)):
        mask = groups_arr == group
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        metrics = compute_classification_metrics(sub_preds, sub_labels)
        out[group] = {
            "n": int(mask.sum()),
            "errors": int((sub_preds != sub_labels).sum()),
            "accuracy": metrics["accuracy"],
            "false_trustworthy_rate": metrics["false_trustworthy_rate"],
        }
    return out


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_group_metrics(per_seed: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups = sorted({group for seed in per_seed for group in seed[key]})
    out = {}
    for group in groups:
        rows = [seed[key][group] for seed in per_seed if group in seed[key]]
        out[group] = {
            "n_mean": float(mean([float(row["n"]) for row in rows])),
            "errors_mean": float(mean([float(row["errors"]) for row in rows])),
            "accuracy": mean_std([float(row["accuracy"]) for row in rows]),
            "false_trustworthy_rate": mean_std(
                [float(row["false_trustworthy_rate"]) for row in rows]
            ),
        }
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def markdown_table_group(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | N | Errors | Accuracy | FT |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        groups.items(),
        key=lambda item: (item[1]["accuracy"]["mean"], -item[1]["errors_mean"]),
    )
    for group, row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    f"{row['n_mean']:.0f}",
                    f"{row['errors_mean']:.1f}",
                    f"{format_pct(row['accuracy']['mean'])} +/- {format_pct(row['accuracy']['std'])}",
                    f"{format_pct(row['false_trustworthy_rate']['mean'])} +/- {format_pct(row['false_trustworthy_rate']['std'])}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = cfg.get("stage0", {})
    data_cfg = cfg.get("data", {})
    data_dir = (args.data_dir or Path(data_cfg.get("moe_output_dir", "data/moe_v8"))).resolve()
    output_dir = args.output_dir or (args.run_root / f"failure_analysis_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = MoEVocab.from_metadata(data_dir / "metadata.json")
    route_names = invert_mapping(vocab.route2id)
    taxonomy_names = invert_mapping(vocab.taxonomy_pattern2id)
    route_names_for_confusion = {**route_names}
    taxonomy_names_for_confusion = {**taxonomy_names}
    label_names = dict(ID2LABEL)

    first_payload = torch.load(run_dirs(args.run_root, args.seeds)[0][1] / "model.pt", map_location="cpu")
    token_vocab_size = int(first_payload["config"]["token_vocab_size"])
    max_length = int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768)))
    max_query_length = int(stage_cfg.get("max_query_length", 96))
    max_sources = int(stage_cfg.get("max_sources", 8))
    max_source_length = int(stage_cfg.get("max_source_length", 192))
    batch_size = int(args.batch_size or stage_cfg.get("per_device_eval_batch_size", 256))

    ds = MoEJsonlDataset(
        data_dir / f"{args.split}.jsonl",
        vocab=vocab,
        token_vocab_size=token_vocab_size,
        max_length=max_length,
        max_query_length=max_query_length,
        max_sources=max_sources,
        max_source_length=max_source_length,
        limit=args.max_samples,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_moe_batch)
    metadata_by_id = row_metadata(ds, route_names, taxonomy_names)
    row_ids = [str(row["id"]) for row in ds.rows]
    labels = np.asarray([metadata_by_id[row_id]["label_id"] for row_id in row_ids])
    route_labels = np.asarray([metadata_by_id[row_id]["route_id"] for row_id in row_ids])
    taxonomy_labels = np.asarray([metadata_by_id[row_id]["taxonomy_id"] for row_id in row_ids])
    routes = [metadata_by_id[row_id]["route"] for row_id in row_ids]
    taxonomy_patterns = [metadata_by_id[row_id]["taxonomy"] for row_id in row_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed_reports = []
    predictions_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for seed, run_dir in run_dirs(args.run_root, args.seeds):
        threshold = threshold_for_split(run_dir / "final_metrics.json", args.split)
        model = load_model(run_dir / "model.pt")
        seed_predictions = collect_seed_predictions(
            model=model,
            loader=loader,
            device=device,
            threshold=threshold,
            force_route_ids=args.force_route_ids,
        )
        predictions_by_seed[seed] = seed_predictions

        preds = np.asarray([seed_predictions[row_id]["pred_id"] for row_id in row_ids])
        route_preds = np.asarray([seed_predictions[row_id]["route_pred_id"] for row_id in row_ids])
        taxonomy_preds = np.asarray(
            [seed_predictions[row_id]["taxonomy_pred_id"] for row_id in row_ids]
        )
        seed_report = {
            "seed": seed,
            "run_dir": str(run_dir),
            "threshold": threshold,
            "governance": compute_classification_metrics(preds, labels),
            "route_accuracy": route_accuracy(route_preds, route_labels),
            "taxonomy_accuracy": taxonomy_accuracy(taxonomy_preds, taxonomy_labels),
            "governance_confusion": confusion_counts(
                labels.tolist(),
                preds.tolist(),
                label_names,
            ),
            "route_confusion": confusion_counts(
                route_labels.tolist(),
                route_preds.tolist(),
                route_names_for_confusion,
            ),
            "taxonomy_confusion": confusion_counts(
                taxonomy_labels.tolist(),
                taxonomy_preds.tolist(),
                taxonomy_names_for_confusion,
            ),
            "by_route": group_metrics(
                row_ids=row_ids,
                labels=labels,
                preds=preds,
                groups=routes,
            ),
            "by_taxonomy": group_metrics(
                row_ids=row_ids,
                labels=labels,
                preds=preds,
                groups=taxonomy_patterns,
            ),
        }
        per_seed_reports.append(seed_report)

    case_rows = []
    hard_error_rows = []
    false_trustworthy_rows = []
    for row_id in row_ids:
        meta = metadata_by_id[row_id]
        seed_entries = {}
        miss_count = 0
        false_trustworthy_count = 0
        route_miss_count = 0
        taxonomy_miss_count = 0
        for seed, _ in run_dirs(args.run_root, args.seeds):
            pred = predictions_by_seed[seed][row_id]
            pred_id = int(pred["pred_id"])
            route_pred_id = int(pred["route_pred_id"])
            taxonomy_pred_id = int(pred["taxonomy_pred_id"])
            is_error = pred_id != meta["label_id"]
            is_false_trustworthy = pred_id == 2 and meta["label_id"] != 2
            route_error = route_pred_id != meta["route_id"]
            taxonomy_error = taxonomy_pred_id != meta["taxonomy_id"]
            miss_count += int(is_error)
            false_trustworthy_count += int(is_false_trustworthy)
            route_miss_count += int(route_error)
            taxonomy_miss_count += int(taxonomy_error)
            seed_entries[str(seed)] = {
                "pred_id": pred_id,
                "pred": ID2LABEL[pred_id],
                "raw_pred_id": int(pred["raw_pred_id"]),
                "raw_pred": ID2LABEL[int(pred["raw_pred_id"])],
                "trustworthy_probability": float(pred["governance_probs"][2]),
                "route_pred_id": route_pred_id,
                "route_pred": route_names.get(route_pred_id, str(route_pred_id)),
                "taxonomy_pred_id": taxonomy_pred_id,
                "taxonomy_pred": taxonomy_names.get(taxonomy_pred_id, str(taxonomy_pred_id)),
                "selected_route_id": int(pred["selected_route_id"]),
                "selected_route": route_names.get(
                    int(pred["selected_route_id"]),
                    str(pred["selected_route_id"]),
                ),
                "trust_guard_accept_probability": pred.get(
                    "trust_guard_accept_probability"
                ),
                "trust_guard_penalty": pred.get("trust_guard_penalty"),
                "is_error": is_error,
                "is_false_trustworthy": is_false_trustworthy,
                "is_route_error": route_error,
                "is_taxonomy_error": taxonomy_error,
            }
        case_row = {
            **meta,
            "miss_count": miss_count,
            "false_trustworthy_count": false_trustworthy_count,
            "route_miss_count": route_miss_count,
            "taxonomy_miss_count": taxonomy_miss_count,
            "seeds": seed_entries,
        }
        case_rows.append(case_row)
        if miss_count == len(seed_entries):
            hard_error_rows.append(case_row)
        if false_trustworthy_count:
            false_trustworthy_rows.append(case_row)

    seed_count = len(per_seed_reports)
    summary = {
        "run_root": str(args.run_root),
        "config": str(args.config),
        "data_dir": str(data_dir),
        "split": args.split,
        "n": len(row_ids),
        "device": str(device),
        "force_route_ids": bool(args.force_route_ids),
        "seeds": [item["seed"] for item in per_seed_reports],
        "overall": {
            "accuracy": mean_std([item["governance"]["accuracy"] for item in per_seed_reports]),
            "false_trustworthy_rate": mean_std(
                [item["governance"]["false_trustworthy_rate"] for item in per_seed_reports]
            ),
            "route_accuracy": mean_std([item["route_accuracy"] for item in per_seed_reports]),
            "taxonomy_accuracy": mean_std([item["taxonomy_accuracy"] for item in per_seed_reports]),
        },
        "overlap": {
            "hard_error_count": len(hard_error_rows),
            "hard_error_rate": len(hard_error_rows) / len(row_ids),
            "any_error_count": sum(1 for row in case_rows if row["miss_count"] > 0),
            "any_error_rate": sum(1 for row in case_rows if row["miss_count"] > 0) / len(row_ids),
            "all_seed_false_trustworthy_count": sum(
                1 for row in case_rows if row["false_trustworthy_count"] == seed_count
            ),
            "any_seed_false_trustworthy_count": len(false_trustworthy_rows),
            "miss_count_histogram": dict(Counter(row["miss_count"] for row in case_rows)),
            "false_trustworthy_count_histogram": dict(
                Counter(row["false_trustworthy_count"] for row in case_rows)
            ),
            "route_miss_count_histogram": dict(Counter(row["route_miss_count"] for row in case_rows)),
            "taxonomy_miss_count_histogram": dict(
                Counter(row["taxonomy_miss_count"] for row in case_rows)
            ),
        },
        "by_route": aggregate_group_metrics(per_seed_reports, "by_route"),
        "by_taxonomy": aggregate_group_metrics(per_seed_reports, "by_taxonomy"),
        "per_seed": per_seed_reports,
        "top_hard_errors": sorted(
            hard_error_rows,
            key=lambda row: (
                -row["false_trustworthy_count"],
                -row["route_miss_count"],
                row["route"],
                row["taxonomy"],
                row["id"],
            ),
        )[:100],
        "top_false_trustworthy": sorted(
            false_trustworthy_rows,
            key=lambda row: (
                -row["false_trustworthy_count"],
                row["route"],
                row["taxonomy"],
                row["id"],
            ),
        )[:100],
    }

    (output_dir / "failure_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "case_predictions.jsonl", case_rows)

    md_lines = [
        "# MoE Failure Analysis",
        "",
        f"- Run root: `{args.run_root}`",
        f"- Split: `{args.split}`",
        f"- Rows: **{len(row_ids)}**",
        f"- Seeds: **{', '.join(str(s) for s in summary['seeds'])}**",
        f"- Accuracy: **{format_pct(summary['overall']['accuracy']['mean'])} +/- {format_pct(summary['overall']['accuracy']['std'])}**",
        f"- False-trustworthy: **{format_pct(summary['overall']['false_trustworthy_rate']['mean'])} +/- {format_pct(summary['overall']['false_trustworthy_rate']['std'])}**",
        f"- Route accuracy: **{format_pct(summary['overall']['route_accuracy']['mean'])} +/- {format_pct(summary['overall']['route_accuracy']['std'])}**",
        f"- Taxonomy accuracy: **{format_pct(summary['overall']['taxonomy_accuracy']['mean'])} +/- {format_pct(summary['overall']['taxonomy_accuracy']['std'])}**",
        "",
        "## Error Overlap",
        "",
        f"- Missed by all seeds: **{summary['overlap']['hard_error_count']}** ({format_pct(summary['overlap']['hard_error_rate'])})",
        f"- Missed by at least one seed: **{summary['overlap']['any_error_count']}** ({format_pct(summary['overlap']['any_error_rate'])})",
        f"- False-TRUSTWORTHY in all seeds: **{summary['overlap']['all_seed_false_trustworthy_count']}**",
        f"- False-TRUSTWORTHY in at least one seed: **{summary['overlap']['any_seed_false_trustworthy_count']}**",
        "",
    ]
    md_lines.extend(markdown_table_group("Route Breakdown", summary["by_route"]))
    md_lines.extend(markdown_table_group("Taxonomy Breakdown", summary["by_taxonomy"]))
    md_lines.extend(
        [
            "## Top Hard Errors",
            "",
            "| ID | Gold | Route | Taxonomy | FT seeds | Route misses | Query |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in summary["top_hard_errors"][:30]:
        query = str(row.get("query") or "").replace("|", "\\|")
        if len(query) > 100:
            query = query[:97] + "..."
        md_lines.append(
            f"| `{row['id']}` | {row['label']} | {row['route']} | {row['taxonomy']} | "
            f"{row['false_trustworthy_count']} | {row['route_miss_count']} | {query} |"
        )
    md_lines.append("")
    (output_dir / "failure_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote JSON report : {output_dir / 'failure_report.json'}")
    print(f"Wrote predictions : {output_dir / 'case_predictions.jsonl'}")
    print(f"Wrote markdown    : {output_dir / 'failure_report.md'}")
    print(
        "Summary          : "
        f"acc={summary['overall']['accuracy']['mean']:.4f} "
        f"ft={summary['overall']['false_trustworthy_rate']['mean']:.4f} "
        f"route={summary['overall']['route_accuracy']['mean']:.4f} "
        f"taxonomy={summary['overall']['taxonomy_accuracy']['mean']:.4f} "
        f"hard_errors={summary['overlap']['hard_error_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
