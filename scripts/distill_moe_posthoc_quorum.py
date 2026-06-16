"""Distill a packaged 3-seed verifier policy into one seed's frozen features.

This is a local diagnostic for the Stage 0.7 post-hoc verifier branch. It
trains a small 3-class classifier from one seed's frozen outputs to the packaged
ensemble policy target, then selects a TRUSTWORTHY threshold on eval and applies
it to held-out test. It does not retrain the MoE trunk, call APIs, or generate
dataset rows.

Run from project root:
    python scripts/distill_moe_posthoc_quorum.py \
      --package-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package \
      --output-dir outputs/moe/stage0_7_posthoc_quorum_distill_ft028
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

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pyrrho.metrics import compute_classification_metrics, find_optimal_threshold, gated_predictions
from pyrrho.moe.posthoc_policies import build_default_policy_outputs
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage

DEFAULT_PACKAGE_DIR = Path("outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package")
DEFAULT_OUTPUT_DIR = Path("outputs/moe/stage0_7_posthoc_quorum_distill_ft028")


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
metric_row = _POLICY_HELPERS.metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument(
        "--target-policy",
        choices=("majority_guarded_safety_tie", "trustworthy_quorum_2_of_3", "trustworthy_unanimous"),
        default="trustworthy_quorum_2_of_3",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def one_hot(ids: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((ids.shape[0], width), dtype=np.float32)
    out[np.arange(ids.shape[0]), ids.astype(int)] = 1.0
    return out


def distill_features(collected: dict[str, Any], seed_idx: int) -> np.ndarray:
    return np.concatenate(
        [
            collected["features"][seed_idx],
            collected["probabilities"][seed_idx],
            collected["accept_scores"][seed_idx, :, None],
            one_hot(collected["base_preds"][seed_idx], 3),
            one_hot(collected["guarded_preds"][seed_idx], 3),
        ],
        axis=1,
    ).astype(np.float32)


def policy_predictions(collected: dict[str, Any], target_policy: str) -> np.ndarray:
    policies = build_default_policy_outputs(
        seed_predictions=collected["guarded_preds"],
        seed_probabilities=collected["probabilities"],
    )
    for policy in policies:
        if policy.name == target_policy:
            return policy.predictions
    raise ValueError(f"target policy {target_policy!r} was not produced")


def logits_from_probabilities(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0))


def agreement(preds: np.ndarray, target: np.ndarray) -> float:
    return float((np.asarray(preds) == np.asarray(target)).mean())


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def train_seed_distiller(
    *,
    seed_idx: int,
    seed: int,
    splits: dict[str, dict[str, Any]],
    targets: dict[str, np.ndarray],
    target_eval_ft: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
        l2_regularization=float(args.l2_regularization),
        random_state=int(seed),
    )
    train_x = distill_features(splits["train"], seed_idx)
    model.fit(train_x, targets["train"])

    split_payloads: dict[str, Any] = {}
    eval_logits = None
    for split_name, split in splits.items():
        x = distill_features(split, seed_idx)
        probs = model.predict_proba(x)
        logits = logits_from_probabilities(probs)
        raw_preds = probs.argmax(axis=1)
        if split_name == "eval":
            eval_logits = logits
        split_payloads[split_name] = {
            "raw": metric_row(raw_preds, split["labels"]),
            "raw_target_agreement": agreement(raw_preds, targets[split_name]),
            "logits": logits,
        }
    if eval_logits is None:
        raise ValueError("eval split is required")
    threshold = find_optimal_threshold(
        eval_logits,
        splits["eval"]["labels"],
        target_ft=float(target_eval_ft),
        grid_size=int(args.threshold_grid_size),
        num_classes=3,
    )
    selected_threshold = float(threshold["threshold"])
    final: dict[str, Any] = {}
    for split_name, split in splits.items():
        preds = gated_predictions(
            split_payloads[split_name]["logits"],
            selected_threshold,
            num_classes=3,
        )
        final[split_name] = {
            "metrics": metric_row(preds, split["labels"]),
            "target_agreement": agreement(preds, targets[split_name]),
        }
    model_path = args.output_dir / "seeds" / f"seed_{seed}" / "quorum_distiller.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {
        "seed": int(seed),
        "model_path": str(model_path),
        "selected_threshold": selected_threshold,
        "threshold_selection": threshold,
        "raw": {
            split_name: {
                "metrics": payload["raw"],
                "target_agreement": payload["raw_target_agreement"],
            }
            for split_name, payload in split_payloads.items()
        },
        "calibrated": final,
    }


def aggregate_seed_rows(seed_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return {
        "accuracy": mean_std([row["calibrated"][split]["metrics"]["accuracy"] for row in seed_rows]),
        "false_trustworthy_rate": mean_std(
            [row["calibrated"][split]["metrics"]["false_trustworthy_rate"] for row in seed_rows]
        ),
        "trustworthy_recall": mean_std(
            [row["calibrated"][split]["metrics"]["trustworthy_recall"] for row in seed_rows]
        ),
        "target_agreement": mean_std([row["calibrated"][split]["target_agreement"] for row in seed_rows]),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MoE Post-Hoc Quorum Distillation",
        "",
        f"- Package: `{report['package_dir']}`",
        f"- Data dir: `{report['data_dir']}`",
        f"- Target policy: `{report['target_policy']['name']}`",
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
            "## Distilled Single-Seed Calibrated Test",
            "",
            "| Seed | Tau | Accuracy | FT | T Recall | Target Agreement |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["seeds"]:
        metrics = row["calibrated"]["test"]["metrics"]
        lines.append(
            f"| {row['seed']} | {row['selected_threshold']:.3f} | "
            f"{metrics['accuracy'] * 100:.2f}% | "
            f"{metrics['false_trustworthy_rate'] * 100:.2f}% | "
            f"{metrics['trustworthy_recall'] * 100:.2f}% | "
            f"{row['calibrated']['test']['target_agreement'] * 100:.2f}% |"
        )
    agg = report["calibrated_test_mean_std"]
    lines.extend(
        [
            "",
            "## Calibrated Test Mean",
            "",
            f"- Accuracy: **{agg['accuracy']['mean'] * 100:.2f} +/- {agg['accuracy']['std'] * 100:.2f}%**",
            f"- False-TRUSTWORTHY: **{agg['false_trustworthy_rate']['mean'] * 100:.2f} +/- {agg['false_trustworthy_rate']['std'] * 100:.2f}%**",
            f"- TRUSTWORTHY recall: **{agg['trustworthy_recall']['mean'] * 100:.2f} +/- {agg['trustworthy_recall']['std'] * 100:.2f}%**",
            f"- Target agreement: **{agg['target_agreement']['mean'] * 100:.2f} +/- {agg['target_agreement']['std'] * 100:.2f}%**",
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

    splits = {
        split: collect_seed_outputs(
            package=package,
            root=root,
            data_dir=data_dir,
            split=split,
            batch_size_override=args.batch_size,
            max_samples=args.max_samples,
        )
        for split in ("train", "eval", "test")
    }
    targets = {
        split: policy_predictions(payload, args.target_policy)
        for split, payload in splits.items()
    }
    target_metrics = {
        split: metric_row(targets[split], splits[split]["labels"])
        for split in ("eval", "test")
    }
    seed_rows = []
    for seed_idx, seed_row in enumerate(splits["train"]["seed_rows"]):
        seed = int(seed_row["seed"])
        seed_rows.append(
            train_seed_distiller(
                seed_idx=seed_idx,
                seed=seed,
                splits=splits,
                targets=targets,
                target_eval_ft=float(target_metrics["eval"]["false_trustworthy_rate"]),
                args=args,
            )
        )

    test_agg = aggregate_seed_rows(seed_rows, "test")
    best_single = max(
        seed_rows,
        key=lambda row: (
            row["calibrated"]["test"]["metrics"]["accuracy"],
            -row["calibrated"]["test"]["metrics"]["false_trustworthy_rate"],
            row["calibrated"]["test"]["metrics"]["trustworthy_recall"],
        ),
    )
    target_test = target_metrics["test"]
    conclusion = (
        "quorum distillation did not match the 3-forward quorum on held-out "
        "accuracy and false-TRUSTWORTHY"
    )
    if (
        best_single["calibrated"]["test"]["metrics"]["accuracy"] >= target_test["accuracy"]
        and best_single["calibrated"]["test"]["metrics"]["false_trustworthy_rate"]
        <= target_test["false_trustworthy_rate"]
    ):
        conclusion = "at least one distilled single-seed model matched or beat the quorum on held-out accuracy and FT"

    report = {
        "schema_version": "pyrrho_moe_posthoc_quorum_distill_v1",
        "package_dir": str(args.package_dir.resolve()),
        "data_dir": str(data_dir),
        "max_samples": args.max_samples,
        "distiller": {
            "kind": "HistGradientBoostingClassifier",
            "max_iter": int(args.max_iter),
            "learning_rate": float(args.learning_rate),
            "l2_regularization": float(args.l2_regularization),
            "threshold_grid_size": int(args.threshold_grid_size),
        },
        "target_policy": {
            "name": args.target_policy,
            "eval": target_metrics["eval"],
            "test": target_metrics["test"],
        },
        "seeds": seed_rows,
        "calibrated_test_mean_std": test_agg,
        "best_single_seed_by_test_accuracy": int(best_single["seed"]),
        "conclusion": conclusion,
    }
    write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        f"target={args.target_policy} "
        f"test_acc={target_test['accuracy']:.4f} test_ft={target_test['false_trustworthy_rate']:.4f}"
    )
    print(
        "distilled_mean "
        f"test_acc={test_agg['accuracy']['mean']:.4f} "
        f"test_ft={test_agg['false_trustworthy_rate']['mean']:.4f}"
    )
    print(f"best_single_seed={best_single['seed']} {conclusion}")
    print(f"Wrote summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
