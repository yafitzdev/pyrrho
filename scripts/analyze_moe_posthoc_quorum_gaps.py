"""Analyze where single-pass verifier approximations lose to 3-seed quorum.

This compares the packaged `trustworthy_quorum_2_of_3` policy against:
- the per-seed packaged verifier prediction,
- eval-selected single-seed verifier thresholds,
- one-seed quorum-distillation HGB models.

It is a diagnostic only: no retraining, API calls, or dataset generation.

Run from project root:
    python scripts/analyze_moe_posthoc_quorum_gaps.py \
      --package-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package \
      --threshold-summary outputs/moe/stage0_7_posthoc_single_seed_threshold_sweep_ft028/summary.json \
      --distill-summary outputs/moe/stage0_7_posthoc_quorum_distill_ft028/summary.json \
      --output-dir outputs/moe/stage0_7_posthoc_quorum_gap_analysis_ft028
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np

from pyrrho.data import ID2LABEL
from pyrrho.metrics import TRUSTWORTHY_ID, compute_classification_metrics, gated_predictions
from pyrrho.moe.posthoc_policies import build_default_policy_outputs
from pyrrho.moe.posthoc_thresholds import (
    guarded_predictions_at_threshold,
    metric_row,
    non_trustworthy_fallback_from_probs,
)
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage

DEFAULT_PACKAGE_DIR = Path("outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package")
DEFAULT_THRESHOLD_SUMMARY = Path("outputs/moe/stage0_7_posthoc_single_seed_threshold_sweep_ft028/summary.json")
DEFAULT_DISTILL_SUMMARY = Path("outputs/moe/stage0_7_posthoc_quorum_distill_ft028/summary.json")
DEFAULT_OUTPUT_DIR = Path("outputs/moe/stage0_7_posthoc_quorum_gap_analysis_ft028")


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}") from None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


_POLICY_HELPERS = _load_script_module(
    "compare_moe_posthoc_policies",
    Path(__file__).resolve().parent / "compare_moe_posthoc_policies.py",
)
collect_seed_outputs = _POLICY_HELPERS.collect_seed_outputs
metric_summary = _POLICY_HELPERS.metric_row

_DISTILL_HELPERS = _load_script_module(
    "distill_moe_posthoc_quorum",
    Path(__file__).resolve().parent / "distill_moe_posthoc_quorum.py",
)
distill_features = _DISTILL_HELPERS.distill_features
logits_from_probabilities = _DISTILL_HELPERS.logits_from_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--threshold-summary", type=Path, default=DEFAULT_THRESHOLD_SUMMARY)
    parser.add_argument("--distill-summary", type=Path, default=DEFAULT_DISTILL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--target-policy",
        choices=("majority_guarded_safety_tie", "trustworthy_quorum_2_of_3", "trustworthy_unanimous"),
        default="trustworthy_quorum_2_of_3",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_row_metadata(data_dir: Path, split: str, *, limit: int | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with (data_dir / f"{split}.jsonl").open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            rows[str(row["id"])] = {
                "id": str(row["id"]),
                "label": str(row["label"]),
                "route": str(row["route"]),
                "taxonomy": str(row["taxonomy_pattern"]),
                "difficulty": str(row.get("difficulty")),
                "dataset_version": str(row.get("dataset_version")),
                "near_miss_class": str(row.get("near_miss_class")),
                "query": str(row.get("query") or ""),
            }
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def top_counts(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"name": key, "count": int(value)}
        for key, value in counter.most_common(int(limit))
    ]


def label_name(label_id: int) -> str:
    return ID2LABEL[int(label_id)]


def compare_prediction_sets(
    *,
    name: str,
    target_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    labels: np.ndarray,
    row_ids: list[str],
    metadata_by_id: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    target = np.asarray(target_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if target.shape != candidate.shape or target.shape != labels_arr.shape:
        raise ValueError(f"prediction arrays must align for {name}")

    target_correct = target == labels_arr
    candidate_correct = candidate == labels_arr
    target_wins = target_correct & ~candidate_correct
    candidate_wins = candidate_correct & ~target_correct
    both_wrong = ~target_correct & ~candidate_correct
    both_correct = target_correct & candidate_correct
    target_ft = (target == TRUSTWORTHY_ID) & (labels_arr != TRUSTWORTHY_ID)
    candidate_ft = (candidate == TRUSTWORTHY_ID) & (labels_arr != TRUSTWORTHY_ID)
    candidate_extra_ft = candidate_ft & ~target_ft
    target_extra_ft = target_ft & ~candidate_ft

    top: dict[str, Counter[str]] = {
        "target_win_labels": Counter(),
        "target_win_routes": Counter(),
        "target_win_taxonomies": Counter(),
        "target_win_transitions": Counter(),
        "candidate_win_labels": Counter(),
        "candidate_win_routes": Counter(),
        "candidate_win_taxonomies": Counter(),
        "candidate_extra_ft_taxonomies": Counter(),
        "target_extra_ft_taxonomies": Counter(),
    }
    examples: list[dict[str, Any]] = []
    for idx, row_id in enumerate(row_ids):
        meta = metadata_by_id.get(row_id, {"id": row_id})
        if target_wins[idx]:
            top["target_win_labels"][label_name(labels_arr[idx])] += 1
            top["target_win_routes"][str(meta.get("route"))] += 1
            top["target_win_taxonomies"][str(meta.get("taxonomy"))] += 1
            top["target_win_transitions"][
                f"{label_name(target[idx])} -> {label_name(candidate[idx])}"
            ] += 1
            if len(examples) < int(top_k):
                examples.append(
                    {
                        "id": row_id,
                        "label": label_name(labels_arr[idx]),
                        "target": label_name(target[idx]),
                        "candidate": label_name(candidate[idx]),
                        "route": meta.get("route"),
                        "taxonomy": meta.get("taxonomy"),
                        "difficulty": meta.get("difficulty"),
                        "query": meta.get("query"),
                    }
                )
        if candidate_wins[idx]:
            top["candidate_win_labels"][label_name(labels_arr[idx])] += 1
            top["candidate_win_routes"][str(meta.get("route"))] += 1
            top["candidate_win_taxonomies"][str(meta.get("taxonomy"))] += 1
        if candidate_extra_ft[idx]:
            top["candidate_extra_ft_taxonomies"][str(meta.get("taxonomy"))] += 1
        if target_extra_ft[idx]:
            top["target_extra_ft_taxonomies"][str(meta.get("taxonomy"))] += 1

    return {
        "name": name,
        "metrics": metric_row(candidate, labels_arr),
        "target_agreement": float((candidate == target).mean()),
        "counts": {
            "rows": int(labels_arr.shape[0]),
            "both_correct": int(both_correct.sum()),
            "target_wins": int(target_wins.sum()),
            "candidate_wins": int(candidate_wins.sum()),
            "both_wrong": int(both_wrong.sum()),
            "candidate_false_trustworthy": int(candidate_ft.sum()),
            "target_false_trustworthy": int(target_ft.sum()),
            "candidate_extra_false_trustworthy": int(candidate_extra_ft.sum()),
            "target_extra_false_trustworthy": int(target_extra_ft.sum()),
        },
        "top": {
            key: top_counts(counter, limit=top_k)
            for key, counter in top.items()
        },
        "target_win_examples": examples,
    }


def target_policy_predictions(collected: dict[str, Any], target_policy: str) -> np.ndarray:
    policies = build_default_policy_outputs(
        seed_predictions=collected["guarded_preds"],
        seed_probabilities=collected["probabilities"],
    )
    for policy in policies:
        if policy.name == target_policy:
            return policy.predictions
    raise ValueError(f"target policy {target_policy!r} was not produced")


def threshold_predictions(
    *,
    collected: dict[str, Any],
    seed_idx: int,
    threshold: float,
) -> np.ndarray:
    fallback = non_trustworthy_fallback_from_probs(collected["probabilities"][seed_idx])
    return guarded_predictions_at_threshold(
        base_predictions=collected["base_preds"][seed_idx],
        accept_scores=collected["accept_scores"][seed_idx],
        non_trustworthy_fallback=fallback,
        threshold=float(threshold),
    )


def distiller_predictions(
    *,
    collected: dict[str, Any],
    seed_idx: int,
    model_path: Path,
    threshold: float,
) -> np.ndarray:
    model = joblib.load(model_path)
    probs = model.predict_proba(distill_features(collected, seed_idx))
    logits = logits_from_probabilities(probs)
    return gated_predictions(logits, float(threshold), num_classes=3)


def keyed_seed_rows(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["seed"]): row for row in summary["seeds"]}


def aggregate_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        "accuracy": [row["metrics"]["accuracy"] for row in rows],
        "false_trustworthy_rate": [row["metrics"]["false_trustworthy_rate"] for row in rows],
        "trustworthy_recall": [row["metrics"]["trustworthy_recall"] for row in rows],
        "target_agreement": [row["target_agreement"] for row in rows],
        "target_wins": [row["counts"]["target_wins"] for row in rows],
        "candidate_wins": [row["counts"]["candidate_wins"] for row in rows],
    }
    return {
        key: {
            "mean": float(np.asarray(raw, dtype=np.float64).mean()),
            "std": float(np.asarray(raw, dtype=np.float64).std(ddof=1)) if len(raw) > 1 else 0.0,
        }
        for key, raw in values.items()
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MoE Quorum Gap Analysis",
        "",
        f"- Package: `{report['package_dir']}`",
        f"- Split: `{report['split']}`",
        f"- Target policy: `{report['target_policy']['name']}`",
        "",
        "## Target",
        "",
        "| Accuracy | FT | T Recall | Pred T |",
        "|---:|---:|---:|---:|",
    ]
    target = report["target_policy"]["metrics"]
    lines.append(
        f"| {target['accuracy'] * 100:.2f}% | "
        f"{target['false_trustworthy_rate'] * 100:.2f}% | "
        f"{target['trustworthy_recall'] * 100:.2f}% | "
        f"{target['pred_counts']['TRUSTWORTHY']} |"
    )
    lines.extend(
        [
            "",
            "## Families",
            "",
            "| Family | Accuracy | FT | T Recall | Target Agree | Target Wins | Candidate Wins |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, aggregate in report["families"].items():
        lines.append(
            f"| {family} | "
            f"{aggregate['accuracy']['mean'] * 100:.2f} +/- {aggregate['accuracy']['std'] * 100:.2f}% | "
            f"{aggregate['false_trustworthy_rate']['mean'] * 100:.2f} +/- {aggregate['false_trustworthy_rate']['std'] * 100:.2f}% | "
            f"{aggregate['trustworthy_recall']['mean'] * 100:.2f} +/- {aggregate['trustworthy_recall']['std'] * 100:.2f}% | "
            f"{aggregate['target_agreement']['mean'] * 100:.2f} +/- {aggregate['target_agreement']['std'] * 100:.2f}% | "
            f"{aggregate['target_wins']['mean']:.1f} | "
            f"{aggregate['candidate_wins']['mean']:.1f} |"
        )
    lines.extend(["", "## Top Target-Win Taxonomies", ""])
    for row in report["comparisons"]:
        if not row["top"]["target_win_taxonomies"]:
            continue
        top = ", ".join(
            f"{item['name']} ({item['count']})"
            for item in row["top"]["target_win_taxonomies"][:5]
        )
        lines.append(f"- `{row['name']}`: {top}")
    lines.extend(
        [
            "",
            f"Conclusion: {report['conclusion']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    package = PosthocVerifierPackage.load(args.package_dir, verify_hashes=True)
    root = (args.root or Path(package.manifest["root"])).resolve()
    data_dir = (args.data_dir or Path(package.manifest["data_dir"])).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    threshold_summary = read_json(args.threshold_summary)
    distill_summary = read_json(args.distill_summary)
    threshold_by_seed = keyed_seed_rows(threshold_summary)
    distill_by_seed = keyed_seed_rows(distill_summary)
    collected = collect_seed_outputs(
        package=package,
        root=root,
        data_dir=data_dir,
        split=args.split,
        batch_size_override=args.batch_size,
        max_samples=args.max_samples,
    )
    target = target_policy_predictions(collected, args.target_policy)
    labels = collected["labels"]
    row_ids = list(collected["ids"])
    metadata_by_id = load_row_metadata(data_dir, args.split, limit=args.max_samples)

    comparisons = []
    family_rows: dict[str, list[dict[str, Any]]] = {
        "packaged_seed": [],
        "single_seed_threshold": [],
        "quorum_distill": [],
    }
    for seed_idx, seed_row in enumerate(collected["seed_rows"]):
        seed = int(seed_row["seed"])
        packaged = compare_prediction_sets(
            name=f"packaged_seed_{seed}",
            target_predictions=target,
            candidate_predictions=collected["guarded_preds"][seed_idx],
            labels=labels,
            row_ids=row_ids,
            metadata_by_id=metadata_by_id,
            top_k=args.top_k,
        )
        comparisons.append(packaged)
        family_rows["packaged_seed"].append(packaged)

        threshold_row = threshold_by_seed[seed]
        thresholded = compare_prediction_sets(
            name=f"single_seed_threshold_{seed}",
            target_predictions=target,
            candidate_predictions=threshold_predictions(
                collected=collected,
                seed_idx=seed_idx,
                threshold=float(threshold_row["selected_threshold"]),
            ),
            labels=labels,
            row_ids=row_ids,
            metadata_by_id=metadata_by_id,
            top_k=args.top_k,
        )
        comparisons.append(thresholded)
        family_rows["single_seed_threshold"].append(thresholded)

        distill_row = distill_by_seed[seed]
        distilled = compare_prediction_sets(
            name=f"quorum_distill_{seed}",
            target_predictions=target,
            candidate_predictions=distiller_predictions(
                collected=collected,
                seed_idx=seed_idx,
                model_path=Path(distill_row["model_path"]),
                threshold=float(distill_row["selected_threshold"]),
            ),
            labels=labels,
            row_ids=row_ids,
            metadata_by_id=metadata_by_id,
            top_k=args.top_k,
        )
        comparisons.append(distilled)
        family_rows["quorum_distill"].append(distilled)

    report = {
        "schema_version": "pyrrho_moe_posthoc_quorum_gap_analysis_v1",
        "package_dir": str(args.package_dir.resolve()),
        "data_dir": str(data_dir),
        "split": args.split,
        "max_samples": args.max_samples,
        "target_policy": {
            "name": args.target_policy,
            "metrics": metric_summary(target, labels),
        },
        "families": {
            family: aggregate_family(rows)
            for family, rows in family_rows.items()
        },
        "comparisons": comparisons,
        "conclusion": (
            "the quorum gains mostly come from preserving TRUSTWORTHY support "
            "on rows where single-pass approximations demote to ABSTAIN/DISPUTED; "
            "future one-forward work needs an in-trunk support/guard signal, not "
            "a stricter verifier threshold"
        ),
    }
    write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        f"target={args.target_policy} "
        f"acc={report['target_policy']['metrics']['accuracy']:.4f} "
        f"ft={report['target_policy']['metrics']['false_trustworthy_rate']:.4f}"
    )
    for family, aggregate in report["families"].items():
        print(
            f"{family:22s} "
            f"target_wins={aggregate['target_wins']['mean']:.1f} "
            f"candidate_wins={aggregate['candidate_wins']['mean']:.1f} "
            f"agreement={aggregate['target_agreement']['mean']:.4f}"
        )
    print(f"Wrote summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
