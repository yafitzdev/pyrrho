"""Compare packaged Stage 0.7 post-hoc verifier seed and ensemble policies.

This uses existing package artifacts and frozen checkpoints only. It does not
train, call APIs, or touch fitz-gov generation state.

Run from project root:
    python scripts/compare_moe_posthoc_policies.py \
      --package-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package \
      --split both \
      --output-dir outputs/moe/stage0_7_posthoc_policy_compare_ft028
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pyrrho.data import ID2LABEL
from pyrrho.metrics import compute_classification_metrics
from pyrrho.moe.data import MoEVocab
from pyrrho.moe.posthoc_policies import build_default_policy_outputs
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage, softmax

DEFAULT_PACKAGE_DIR = Path("outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package")
DEFAULT_OUTPUT_DIR = Path("outputs/moe/stage0_7_posthoc_policy_compare_ft028")


def _load_training_helpers():
    try:
        import train_moe_posthoc_verifier

        return train_moe_posthoc_verifier
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parent / "train_moe_posthoc_verifier.py"
        spec = importlib.util.spec_from_file_location("train_moe_posthoc_verifier", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {module_path}") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("train_moe_posthoc_verifier", module)
        spec.loader.exec_module(module)
        return module


_HELPERS = _load_training_helpers()
collect_split = _HELPERS.collect_split
invert_mapping = _HELPERS.invert_mapping
load_config = _HELPERS.load_config
load_stage0_model = _HELPERS.load_stage0_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["eval", "test", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def split_names(raw: str) -> list[str]:
    return ["eval", "test"] if raw == "both" else [raw]


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def metric_row(preds: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    metrics = compute_classification_metrics(preds, labels)
    counts = {
        ID2LABEL[label_id]: int((preds == label_id).sum())
        for label_id in sorted(ID2LABEL)
    }
    return {
        "accuracy": float(metrics["accuracy"]),
        "false_trustworthy_rate": float(metrics["false_trustworthy_rate"]),
        "trustworthy_recall": float(metrics["recall_trustworthy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "pred_counts": counts,
    }


def aggregate_seed_metrics(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accuracy": mean_std([float(row["metrics"]["accuracy"]) for row in seed_rows]),
        "false_trustworthy_rate": mean_std(
            [float(row["metrics"]["false_trustworthy_rate"]) for row in seed_rows]
        ),
        "trustworthy_recall": mean_std(
            [float(row["metrics"]["trustworthy_recall"]) for row in seed_rows]
        ),
        "rejected_candidate_trustworthy": mean_std(
            [float(row["rejected_candidate_trustworthy"]) for row in seed_rows]
        ),
    }


def policy_recommendations(split_report: dict[str, Any]) -> dict[str, Any]:
    """Select safety-first and support-retaining policy recommendations."""

    policy_rows = split_report["policies"]
    seed_mean = split_report["per_seed_mean_std"]
    safety_first = min(
        policy_rows,
        key=lambda row: (
            float(row["metrics"]["false_trustworthy_rate"]),
            -float(row["metrics"]["accuracy"]),
            -float(row["metrics"]["trustworthy_recall"]),
        ),
    )
    seed_accuracy = float(seed_mean["accuracy"]["mean"])
    seed_ft = float(seed_mean["false_trustworthy_rate"]["mean"])
    dominating = [
        row
        for row in policy_rows
        if float(row["metrics"]["accuracy"]) >= seed_accuracy
        and float(row["metrics"]["false_trustworthy_rate"]) <= seed_ft
    ]
    candidates = dominating or policy_rows
    support_retaining = max(
        candidates,
        key=lambda row: (
            float(row["metrics"]["accuracy"]),
            float(row["metrics"]["trustworthy_recall"]),
            -float(row["metrics"]["false_trustworthy_rate"]),
        ),
    )
    return {
        "safety_first": safety_first["name"],
        "support_retaining": support_retaining["name"],
        "support_retaining_criteria": (
            "max accuracy among policies with accuracy >= per-seed mean and "
            "FT <= per-seed mean"
            if dominating
            else "max accuracy fallback; no policy dominated the per-seed mean on both accuracy and FT"
        ),
    }


def collect_seed_outputs(
    *,
    package: PosthocVerifierPackage,
    root: Path,
    data_dir: Path,
    split: str,
    batch_size_override: int | None,
    max_samples: int | None,
) -> dict[str, Any]:
    vocab = MoEVocab.from_metadata(data_dir / "metadata.json")
    route_names_by_id = invert_mapping(vocab.route2id)
    taxonomy_names_by_id = invert_mapping(vocab.taxonomy_pattern2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_rows = []
    guarded_preds = []
    base_preds = []
    probabilities = []
    features = []
    accept_scores = []
    rejected_masks = []
    reference_ids: list[str] | None = None
    reference_labels: np.ndarray | None = None

    for seed_entry in package.manifest["seeds"]:
        seed = int(seed_entry["seed"])
        checkpoint = resolve_path(seed_entry["checkpoint"], root)
        config_path = resolve_path(seed_entry["config"], root)
        cfg = load_config(config_path)
        stage_cfg = cfg.get("stage0", {})
        data_cfg = cfg.get("data", {})
        payload = torch.load(checkpoint, map_location="cpu")
        model = load_stage0_model(payload)
        model_cfg = model.config
        batch_size = int(batch_size_override or stage_cfg.get("per_device_eval_batch_size", 128))
        frozen = collect_split(
            split=split,
            model=model,
            data_dir=data_dir,
            vocab=vocab,
            route_names_by_id=route_names_by_id,
            taxonomy_names_by_id=taxonomy_names_by_id,
            token_vocab_size=model_cfg.token_vocab_size,
            max_length=int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768))),
            max_query_length=int(stage_cfg.get("max_query_length", 96)),
            max_sources=int(stage_cfg.get("max_sources", 8)),
            max_source_length=int(stage_cfg.get("max_source_length", 192)),
            batch_size=batch_size,
            limit=max_samples,
            base_threshold=float(seed_entry["base_threshold"]),
            device=device,
        )
        if reference_ids is None:
            reference_ids = list(frozen.ids)
            reference_labels = frozen.labels.copy()
        elif reference_ids != list(frozen.ids) or not np.array_equal(reference_labels, frozen.labels):
            raise ValueError(f"split {split} rows are not aligned for seed {seed}")

        guarded = package.apply_features(
            seed=seed,
            features=frozen.features,
            governance_logits=frozen.governance_logits,
        )
        guarded_preds.append(guarded.guarded_predictions)
        base_preds.append(guarded.base_predictions)
        probabilities.append(softmax(frozen.governance_logits))
        features.append(frozen.features.astype(np.float32, copy=False))
        accept_scores.append(guarded.accept_scores)
        rejected_masks.append(guarded.rejected_mask)
        seed_rows.append(
            {
                "seed": seed,
                "base_threshold": float(seed_entry["base_threshold"]),
                "verifier_threshold": float(seed_entry["selected_threshold"]),
                "metrics": metric_row(guarded.guarded_predictions, frozen.labels),
                "base_metrics": metric_row(guarded.base_predictions, frozen.labels),
                "rejected_candidate_trustworthy": guarded.rejected_count,
            }
        )

    assert reference_ids is not None and reference_labels is not None
    return {
        "ids": reference_ids,
        "labels": reference_labels,
        "seed_rows": seed_rows,
        "guarded_preds": np.stack(guarded_preds, axis=0),
        "base_preds": np.stack(base_preds, axis=0),
        "probabilities": np.stack(probabilities, axis=0),
        "features": np.stack(features, axis=0),
        "accept_scores": np.stack(accept_scores, axis=0),
        "rejected_masks": np.stack(rejected_masks, axis=0),
    }


def compare_split(
    *,
    package: PosthocVerifierPackage,
    root: Path,
    data_dir: Path,
    split: str,
    batch_size_override: int | None,
    max_samples: int | None,
    write_predictions: bool,
    output_dir: Path,
) -> dict[str, Any]:
    collected = collect_seed_outputs(
        package=package,
        root=root,
        data_dir=data_dir,
        split=split,
        batch_size_override=batch_size_override,
        max_samples=max_samples,
    )
    labels = collected["labels"]
    policies = build_default_policy_outputs(
        seed_predictions=collected["guarded_preds"],
        seed_probabilities=collected["probabilities"],
    )
    policy_rows = [
        {
            "name": policy.name,
            "metrics": metric_row(policy.predictions, labels),
        }
        for policy in policies
    ]
    seed_rows = collected["seed_rows"]
    report = {
        "split": split,
        "rows": int(labels.shape[0]),
        "per_seed": seed_rows,
        "per_seed_mean_std": aggregate_seed_metrics(seed_rows),
        "policies": policy_rows,
    }
    report["recommendations"] = policy_recommendations(report)
    if write_predictions:
        prediction_rows = []
        policy_by_name = {policy.name: policy.predictions for policy in policies}
        for idx, row_id in enumerate(collected["ids"]):
            prediction_rows.append(
                {
                    "id": row_id,
                    "label": ID2LABEL[int(labels[idx])],
                    "seed_guarded": {
                        str(seed_rows[seed_idx]["seed"]): ID2LABEL[
                            int(collected["guarded_preds"][seed_idx, idx])
                        ]
                        for seed_idx in range(len(seed_rows))
                    },
                    "seed_rejected": {
                        str(seed_rows[seed_idx]["seed"]): bool(
                            collected["rejected_masks"][seed_idx, idx]
                        )
                        for seed_idx in range(len(seed_rows))
                    },
                    "policies": {
                        name: ID2LABEL[int(preds[idx])]
                        for name, preds in policy_by_name.items()
                    },
                }
            )
        write_jsonl(output_dir / f"{split}_policy_predictions.jsonl", prediction_rows)
    return report


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MoE Post-Hoc Verifier Policy Compare",
        "",
        f"- Package: `{report['package_dir']}`",
        f"- Data dir: `{report['data_dir']}`",
        f"- Max samples: `{report['max_samples']}`",
        "",
    ]
    for split, split_report in report["splits"].items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Rows: **{split_report['rows']}**",
                "",
                "| Policy | Accuracy | FT | T Recall | Pred T |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        seed_mean = split_report["per_seed_mean_std"]
        lines.append(
            "| per-seed guarded mean | "
            f"{seed_mean['accuracy']['mean'] * 100:.2f} +/- {seed_mean['accuracy']['std'] * 100:.2f}% | "
            f"{seed_mean['false_trustworthy_rate']['mean'] * 100:.2f} +/- {seed_mean['false_trustworthy_rate']['std'] * 100:.2f}% | "
            f"{seed_mean['trustworthy_recall']['mean'] * 100:.2f} +/- {seed_mean['trustworthy_recall']['std'] * 100:.2f}% | n/a |"
        )
        for row in split_report["policies"]:
            metrics = row["metrics"]
            lines.append(
                f"| {row['name']} | "
                f"{metrics['accuracy'] * 100:.2f}% | "
                f"{metrics['false_trustworthy_rate'] * 100:.2f}% | "
                f"{metrics['trustworthy_recall'] * 100:.2f}% | "
                f"{metrics['pred_counts']['TRUSTWORTHY']} |"
            )
        recs = split_report["recommendations"]
        lines.extend(
            [
                "",
                f"- Support-retaining recommendation: `{recs['support_retaining']}`.",
                f"- Safety-first policy: `{recs['safety_first']}`.",
                f"- Recommendation rule: {recs['support_retaining_criteria']}.",
            ]
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    package = PosthocVerifierPackage.load(args.package_dir, verify_hashes=True)
    root = (args.root or Path(package.manifest["root"])).resolve()
    data_dir = (args.data_dir or Path(package.manifest["data_dir"])).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "pyrrho_moe_posthoc_policy_compare_v1",
        "package_dir": str(args.package_dir.resolve()),
        "data_dir": str(data_dir),
        "split": args.split,
        "max_samples": args.max_samples,
        "seeds": list(package.seed_ids),
        "splits": {},
    }
    for split in split_names(args.split):
        split_report = compare_split(
            package=package,
            root=root,
            data_dir=data_dir,
            split=split,
            batch_size_override=args.batch_size,
            max_samples=args.max_samples,
            write_predictions=args.write_predictions,
            output_dir=args.output_dir,
        )
        report["splits"][split] = split_report
        policy_by_name = {row["name"]: row for row in split_report["policies"]}
        recs = split_report["recommendations"]
        safety_first = policy_by_name[recs["safety_first"]]
        support_retaining = policy_by_name[recs["support_retaining"]]
        print(
            f"{split:5s} recommended={support_retaining['name']} "
            f"acc={support_retaining['metrics']['accuracy']:.4f} "
            f"ft={support_retaining['metrics']['false_trustworthy_rate']:.4f} "
            f"safety_first={safety_first['name']} "
            f"safety_ft={safety_first['metrics']['false_trustworthy_rate']:.4f}"
        )

    write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Wrote summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
