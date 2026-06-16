"""Test whether one packaged verifier seed can approximate quorum by thresholding.

This uses existing Stage 0.7 package artifacts and frozen checkpoints only. It
selects a verifier accept-score threshold on eval, then applies that threshold
to held-out test. No retraining, API calls, or dataset generation are involved.

Run from project root:
    python scripts/sweep_moe_posthoc_single_seed_thresholds.py \
      --package-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package \
      --output-dir outputs/moe/stage0_7_posthoc_single_seed_threshold_sweep_ft028
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

from pyrrho.moe.posthoc_policies import build_default_policy_outputs
from pyrrho.moe.posthoc_thresholds import (
    guarded_predictions_at_threshold,
    metric_row,
    non_trustworthy_fallback_from_probs,
    select_threshold_row,
    sweep_verifier_thresholds,
)
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage

DEFAULT_PACKAGE_DIR = Path("outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package")
DEFAULT_OUTPUT_DIR = Path("outputs/moe/stage0_7_posthoc_single_seed_threshold_sweep_ft028")


def _load_policy_compare_helpers():
    module_path = Path(__file__).resolve().parent / "compare_moe_posthoc_policies.py"
    spec = importlib.util.spec_from_file_location("compare_moe_posthoc_policies", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_path}") from None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_moe_posthoc_policies", module)
    spec.loader.exec_module(module)
    return module


_POLICY_HELPERS = _load_policy_compare_helpers()
collect_seed_outputs = _POLICY_HELPERS.collect_seed_outputs
metric_summary = _POLICY_HELPERS.metric_row
resolve_path = _POLICY_HELPERS.resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument(
        "--target-policy",
        choices=("majority_guarded_safety_tie", "trustworthy_quorum_2_of_3", "trustworthy_unanimous"),
        default="trustworthy_quorum_2_of_3",
    )
    parser.add_argument(
        "--min-accuracy-delta",
        type=float,
        default=None,
        help="Optional eval accuracy floor relative to target policy, e.g. -0.005 allows a 0.5 pp drop.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def target_policy_metrics(collected: dict[str, Any], target_policy: str) -> dict[str, Any]:
    policies = build_default_policy_outputs(
        seed_predictions=collected["guarded_preds"],
        seed_probabilities=collected["probabilities"],
    )
    for policy in policies:
        if policy.name == target_policy:
            return {
                "name": policy.name,
                "metrics": metric_summary(policy.predictions, collected["labels"]),
            }
    raise ValueError(f"target policy {target_policy!r} was not produced")


def seed_threshold_sweep(
    *,
    seed_idx: int,
    seed: int,
    eval_outputs: dict[str, Any],
    test_outputs: dict[str, Any],
    thresholds: np.ndarray,
    target_eval_ft: float,
    min_eval_accuracy: float | None,
) -> dict[str, Any]:
    eval_fallback = non_trustworthy_fallback_from_probs(eval_outputs["probabilities"][seed_idx])
    test_fallback = non_trustworthy_fallback_from_probs(test_outputs["probabilities"][seed_idx])
    eval_sweep = sweep_verifier_thresholds(
        labels=eval_outputs["labels"],
        base_predictions=eval_outputs["base_preds"][seed_idx],
        accept_scores=eval_outputs["accept_scores"][seed_idx],
        non_trustworthy_fallback=eval_fallback,
        thresholds=thresholds,
    )
    selected = select_threshold_row(
        eval_sweep,
        target_ft=target_eval_ft,
        min_accuracy=min_eval_accuracy,
    )
    selected_threshold = float(selected["threshold"])
    eval_preds = guarded_predictions_at_threshold(
        base_predictions=eval_outputs["base_preds"][seed_idx],
        accept_scores=eval_outputs["accept_scores"][seed_idx],
        non_trustworthy_fallback=eval_fallback,
        threshold=selected_threshold,
    )
    test_preds = guarded_predictions_at_threshold(
        base_predictions=test_outputs["base_preds"][seed_idx],
        accept_scores=test_outputs["accept_scores"][seed_idx],
        non_trustworthy_fallback=test_fallback,
        threshold=selected_threshold,
    )
    package_eval = eval_outputs["seed_rows"][seed_idx]
    package_test = test_outputs["seed_rows"][seed_idx]
    return {
        "seed": int(seed),
        "selected_threshold": selected_threshold,
        "selection_reason": selected["selection_reason"],
        "target_eval_ft": float(target_eval_ft),
        "min_eval_accuracy": None if min_eval_accuracy is None else float(min_eval_accuracy),
        "eval": metric_row(eval_preds, eval_outputs["labels"]),
        "test": metric_row(test_preds, test_outputs["labels"]),
        "package_threshold": {
            "threshold": float(package_eval["verifier_threshold"]),
            "eval": package_eval["metrics"],
            "test": package_test["metrics"],
        },
        "eval_sweep_selected": selected,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MoE Single-Seed Verifier Threshold Sweep",
        "",
        f"- Package: `{report['package_dir']}`",
        f"- Data dir: `{report['data_dir']}`",
        f"- Target policy: `{report['target_policy']['name']}`",
        f"- Target eval FT: **{report['target_policy']['eval']['false_trustworthy_rate'] * 100:.2f}%**",
        f"- Grid size: `{report['grid_size']}`",
        "",
        "## Target Policy",
        "",
        "| Split | Accuracy | FT | T Recall |",
        "|---|---:|---:|---:|",
    ]
    for split in ("eval", "test"):
        metrics = report["target_policy"][split]
        lines.append(
            f"| {split} | {metrics['accuracy'] * 100:.2f}% | "
            f"{metrics['false_trustworthy_rate'] * 100:.2f}% | "
            f"{metrics['trustworthy_recall'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Eval-Selected Single-Seed Thresholds",
            "",
            "| Seed | Selected tau | Eval Acc | Eval FT | Eval T Recall | Test Acc | Test FT | Test T Recall |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['selected_threshold']:.3f} | "
            f"{row['eval']['accuracy'] * 100:.2f}% | "
            f"{row['eval']['false_trustworthy_rate'] * 100:.2f}% | "
            f"{row['eval']['trustworthy_recall'] * 100:.2f}% | "
            f"{row['test']['accuracy'] * 100:.2f}% | "
            f"{row['test']['false_trustworthy_rate'] * 100:.2f}% | "
            f"{row['test']['trustworthy_recall'] * 100:.2f}% |"
        )
    agg = report["selected_test_mean_std"]
    lines.extend(
        [
            "",
            "## Selected Test Mean",
            "",
            f"- Accuracy: **{agg['accuracy']['mean'] * 100:.2f} +/- {agg['accuracy']['std'] * 100:.2f}%**",
            f"- False-TRUSTWORTHY: **{agg['false_trustworthy_rate']['mean'] * 100:.2f} +/- {agg['false_trustworthy_rate']['std'] * 100:.2f}%**",
            f"- TRUSTWORTHY recall: **{agg['trustworthy_recall']['mean'] * 100:.2f} +/- {agg['trustworthy_recall']['std'] * 100:.2f}%**",
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
    thresholds = np.linspace(0.0, 1.0, int(args.grid_size))

    eval_outputs = collect_seed_outputs(
        package=package,
        root=root,
        data_dir=data_dir,
        split="eval",
        batch_size_override=args.batch_size,
        max_samples=args.max_samples,
    )
    test_outputs = collect_seed_outputs(
        package=package,
        root=root,
        data_dir=data_dir,
        split="test",
        batch_size_override=args.batch_size,
        max_samples=args.max_samples,
    )
    eval_target = target_policy_metrics(eval_outputs, args.target_policy)
    test_target = target_policy_metrics(test_outputs, args.target_policy)
    target_eval_accuracy = float(eval_target["metrics"]["accuracy"])
    min_eval_accuracy = None
    if args.min_accuracy_delta is not None:
        min_eval_accuracy = target_eval_accuracy + float(args.min_accuracy_delta)

    seed_rows = []
    for seed_idx, seed_row in enumerate(eval_outputs["seed_rows"]):
        seed = int(seed_row["seed"])
        seed_rows.append(
            seed_threshold_sweep(
                seed_idx=seed_idx,
                seed=seed,
                eval_outputs=eval_outputs,
                test_outputs=test_outputs,
                thresholds=thresholds,
                target_eval_ft=float(eval_target["metrics"]["false_trustworthy_rate"]),
                min_eval_accuracy=min_eval_accuracy,
            )
        )
    agg = {
        "accuracy": mean_std([row["test"]["accuracy"] for row in seed_rows]),
        "false_trustworthy_rate": mean_std([row["test"]["false_trustworthy_rate"] for row in seed_rows]),
        "trustworthy_recall": mean_std([row["test"]["trustworthy_recall"] for row in seed_rows]),
    }
    best_single = max(
        seed_rows,
        key=lambda row: (
            float(row["test"]["accuracy"]),
            -float(row["test"]["false_trustworthy_rate"]),
            float(row["test"]["trustworthy_recall"]),
        ),
    )
    target_test_acc = float(test_target["metrics"]["accuracy"])
    target_test_ft = float(test_target["metrics"]["false_trustworthy_rate"])
    conclusion = (
        "single-seed eval-selected thresholds do not match the quorum's held-out "
        "accuracy/safety tradeoff"
    )
    if (
        float(best_single["test"]["accuracy"]) >= target_test_acc
        and float(best_single["test"]["false_trustworthy_rate"]) <= target_test_ft
    ):
        conclusion = "at least one single-seed threshold matched or beat the quorum on held-out accuracy and FT"

    report = {
        "schema_version": "pyrrho_moe_posthoc_single_seed_threshold_sweep_v1",
        "package_dir": str(args.package_dir.resolve()),
        "data_dir": str(data_dir),
        "max_samples": args.max_samples,
        "grid_size": int(args.grid_size),
        "min_accuracy_delta": args.min_accuracy_delta,
        "target_policy": {
            "name": args.target_policy,
            "eval": eval_target["metrics"],
            "test": test_target["metrics"],
        },
        "seeds": seed_rows,
        "selected_test_mean_std": agg,
        "best_single_seed_by_test_accuracy": int(best_single["seed"]),
        "conclusion": conclusion,
    }
    write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        f"target={args.target_policy} "
        f"test_acc={target_test_acc:.4f} test_ft={target_test_ft:.4f}"
    )
    print(
        "single_mean "
        f"test_acc={agg['accuracy']['mean']:.4f} "
        f"test_ft={agg['false_trustworthy_rate']['mean']:.4f}"
    )
    print(f"best_single_seed={best_single['seed']} {conclusion}")
    print(f"Wrote summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
