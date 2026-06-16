"""Evaluate per-modality TRUSTWORTHY thresholds for a joint encoder.

This is a local-only policy diagnostic for the structured/code modality branch.
It does not train, generate rows, call external services, or change schema.
Thresholds are selected on the eval split and then applied to the held-out test
split of an existing processed DatasetDict.

Default paths evaluate the retry-patch 3-seed joint-generalist branch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL, load_processed
from pyrrho.manifest import write_manifest
from pyrrho.metrics import breakdown_by, compute_classification_metrics, gated_predictions

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_DATA_DIR = Path("data/processed_v8_plus_structured_code_retry_patch_candidate")
DEFAULT_OUTPUT_DIR = Path("outputs/modality_threshold_sweep/retry_patch_3seed")
DEFAULT_RUNS = {
    "seed42": {
        "checkpoint": Path("outputs/modality_retraining/structured_code_retry_patch_seed42/best_model"),
        "threshold_source": Path("outputs/modality_retraining/structured_code_retry_patch_seed42/eval_report.json"),
    },
    "seed1337": {
        "checkpoint": Path("outputs/modality_retraining/structured_code_retry_patch_seed1337/best_model"),
        "threshold_source": Path("outputs/modality_retraining/structured_code_retry_patch_seed1337/eval_report.json"),
    },
    "seed7": {
        "checkpoint": Path("outputs/modality_retraining/structured_code_retry_patch_seed7/best_model"),
        "threshold_source": Path("outputs/modality_retraining/structured_code_retry_patch_seed7/eval_report.json"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", action="append", default=None, help="Run label. Repeat with --checkpoint.")
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--threshold-source", action="append", type=Path, default=None)
    parser.add_argument("--target-ft", type=float, default=0.057)
    parser.add_argument("--grid-min", type=float, default=0.34)
    parser.add_argument("--grid-max", type=float, default=0.99)
    parser.add_argument("--grid-size", type=int, default=66)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Manifest seed for this deterministic diagnostic.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def configured_runs(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    if args.checkpoint is None and args.threshold_source is None and args.run_name is None:
        return DEFAULT_RUNS
    if not args.checkpoint or not args.threshold_source:
        raise ValueError("--checkpoint and --threshold-source must be provided together")
    if len(args.checkpoint) != len(args.threshold_source):
        raise ValueError("--checkpoint and --threshold-source counts must match")
    if args.run_name and len(args.run_name) != len(args.checkpoint):
        raise ValueError("--run-name count must match --checkpoint count")
    names = args.run_name or [f"run{i + 1}" for i in range(len(args.checkpoint))]
    return {
        name: {"checkpoint": checkpoint, "threshold_source": threshold_source}
        for name, checkpoint, threshold_source in zip(names, args.checkpoint, args.threshold_source)
    }


def load_threshold(source: Path) -> float:
    if not source.exists():
        raise FileNotFoundError(f"Threshold source not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "threshold" in payload:
        return float(payload["threshold"])
    if "eval_calibrated" in payload and "threshold" in payload["eval_calibrated"]:
        return float(payload["eval_calibrated"]["threshold"])
    raise KeyError(f"No threshold field found in {source}")


def infer_logits(
    checkpoint: Path,
    texts: list[str],
    max_seq_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits.append(model(**enc).logits.float().cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(logits, axis=0)


def labels(dataset: Dataset) -> np.ndarray:
    return np.asarray(dataset["label_id"])


def modalities(dataset: Dataset) -> np.ndarray:
    if "modality" not in dataset.column_names:
        return np.asarray(["unstructured"] * len(dataset))
    return np.asarray(dataset["modality"])


def confusion(preds: np.ndarray, gold: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {ID2LABEL[g]: {ID2LABEL[p]: 0 for p in range(3)} for g in range(3)}
    for expected, predicted in zip(gold.tolist(), preds.tolist()):
        out[ID2LABEL[expected]][ID2LABEL[predicted]] += 1
    return out


def apply_modality_thresholds(
    logits: np.ndarray,
    modality_values: np.ndarray,
    thresholds: dict[str, float],
    num_classes: int,
) -> np.ndarray:
    preds = np.empty(logits.shape[0], dtype=np.int64)
    for modality in sorted(set(modality_values.tolist())):
        mask = modality_values == modality
        threshold = thresholds.get(modality)
        if threshold is None:
            raise KeyError(f"No threshold for modality {modality!r}")
        preds[mask] = gated_predictions(logits[mask], threshold, num_classes=num_classes)
    return preds


def policy_report(
    preds: np.ndarray,
    gold: np.ndarray,
    modality_values: np.ndarray,
) -> dict[str, Any]:
    return {
        "metrics": compute_classification_metrics(preds, gold),
        "modalities": breakdown_by(preds, gold, modality_values.tolist()),
        "confusion": confusion(preds, gold),
    }


def select_threshold(
    logits: np.ndarray,
    gold: np.ndarray,
    target_ft: float,
    grid: np.ndarray,
    global_threshold: float,
    num_classes: int,
) -> dict[str, Any]:
    rows = []
    for threshold in grid:
        preds = gated_predictions(logits, float(threshold), num_classes=num_classes)
        metrics = compute_classification_metrics(preds, gold)
        rows.append({"threshold": float(threshold), "metrics": metrics})

    passing = [row for row in rows if row["metrics"]["false_trustworthy_rate"] <= target_ft + 1e-12]
    target_met = bool(passing)
    candidates = passing or rows
    if passing:
        best = max(
            candidates,
            key=lambda row: (
                row["metrics"]["accuracy"],
                -row["metrics"]["false_trustworthy_rate"],
                -abs(row["threshold"] - global_threshold),
            ),
        )
    else:
        best = max(
            candidates,
            key=lambda row: (
                -row["metrics"]["false_trustworthy_rate"],
                row["metrics"]["accuracy"],
                -abs(row["threshold"] - global_threshold),
            ),
        )
    return {
        "threshold": best["threshold"],
        "target_ft": target_ft,
        "target_met": target_met,
        "eval_metrics": best["metrics"],
    }


def select_per_modality_thresholds(
    eval_logits: np.ndarray,
    eval_labels: np.ndarray,
    eval_modalities: np.ndarray,
    global_threshold: float,
    target_ft: float,
    grid: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    global_preds = gated_predictions(eval_logits, global_threshold, num_classes=num_classes)
    global_breakdown = breakdown_by(global_preds, eval_labels, eval_modalities.tolist())

    no_regression: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    for modality in sorted(set(eval_modalities.tolist())):
        mask = eval_modalities == modality
        baseline_ft = global_breakdown[modality]["false_trustworthy_rate"]
        no_regression[modality] = select_threshold(
            eval_logits[mask],
            eval_labels[mask],
            baseline_ft,
            grid,
            global_threshold,
            num_classes,
        )
        gate[modality] = select_threshold(
            eval_logits[mask],
            eval_labels[mask],
            target_ft,
            grid,
            global_threshold,
            num_classes,
        )
    return {"no_eval_ft_regression": no_regression, "release_gate": gate}


def threshold_map(selection: dict[str, Any]) -> dict[str, float]:
    return {modality: float(row["threshold"]) for modality, row in selection.items()}


def evaluate_run(
    run_name: str,
    checkpoint: Path,
    threshold_source: Path,
    ds: Any,
    args: argparse.Namespace,
    device: str,
    grid: np.ndarray,
) -> dict[str, Any]:
    global_threshold = load_threshold(threshold_source)
    print(f"[{run_name}] checkpoint={checkpoint} global_tau={global_threshold:.2f}")

    num_classes = int(AutoConfig.from_pretrained(checkpoint).num_labels)

    split_logits: dict[str, np.ndarray] = {}
    for split_name in ("eval", "test"):
        print(f"  infer {split_name}: rows={len(ds[split_name])}")
        split_logits[split_name] = infer_logits(
            checkpoint,
            list(ds[split_name]["text"]),
            args.max_seq_length,
            args.batch_size,
            device,
        )

    eval_labels = labels(ds["eval"])
    test_labels = labels(ds["test"])
    eval_modalities = modalities(ds["eval"])
    test_modalities = modalities(ds["test"])

    selections = select_per_modality_thresholds(
        split_logits["eval"],
        eval_labels,
        eval_modalities,
        global_threshold,
        args.target_ft,
        grid,
        num_classes,
    )

    policies: dict[str, dict[str, Any]] = {
        "global": {
            "thresholds": {modality: global_threshold for modality in sorted(set(eval_modalities.tolist()))}
        },
        "per_modality_no_eval_ft_regression": {
            "thresholds": threshold_map(selections["no_eval_ft_regression"]),
            "selection": selections["no_eval_ft_regression"],
        },
        "per_modality_release_gate": {
            "thresholds": threshold_map(selections["release_gate"]),
            "selection": selections["release_gate"],
        },
    }

    out: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "threshold_source": str(threshold_source),
        "global_threshold": global_threshold,
        "num_classes": num_classes,
        "policies": {},
    }

    for policy_name, policy in policies.items():
        thresholds = policy["thresholds"]
        eval_preds = apply_modality_thresholds(
            split_logits["eval"], eval_modalities, thresholds, num_classes
        )
        test_preds = apply_modality_thresholds(
            split_logits["test"], test_modalities, thresholds, num_classes
        )
        out["policies"][policy_name] = {
            "thresholds": thresholds,
            "selection": policy.get("selection"),
            "eval": policy_report(eval_preds, eval_labels, eval_modalities),
            "test": policy_report(test_preds, test_labels, test_modalities),
        }

    baseline = out["policies"]["global"]["test"]["metrics"]
    for policy_name, policy in out["policies"].items():
        if policy_name == "global":
            continue
        metrics = policy["test"]["metrics"]
        policy["delta_vs_global_test"] = {
            "accuracy": metrics["accuracy"] - baseline["accuracy"],
            "false_trustworthy_rate": metrics["false_trustworthy_rate"]
            - baseline["false_trustworthy_rate"],
            "recall_trustworthy": metrics["recall_trustworthy"] - baseline["recall_trustworthy"],
            "precision_trustworthy": metrics["precision_trustworthy"]
            - baseline["precision_trustworthy"],
        }
    return out


def aggregate_runs(runs: dict[str, Any]) -> dict[str, Any]:
    policy_names = list(next(iter(runs.values()))["policies"].keys())
    out: dict[str, Any] = {}
    for policy in policy_names:
        out[policy] = {}
        for split in ("eval", "test"):
            metrics_by_key: dict[str, list[float]] = {}
            for run in runs.values():
                metrics = run["policies"][policy][split]["metrics"]
                for key, value in metrics.items():
                    metrics_by_key.setdefault(key, []).append(float(value))
            out[policy][split] = {
                key: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                }
                for key, values in metrics_by_key.items()
            }
    baseline = out["global"]["test"]
    for policy, row in out.items():
        if policy == "global":
            continue
        row["delta_vs_global_test"] = {
            key: {
                "mean": row["test"][key]["mean"] - baseline[key]["mean"],
                "std": float(
                    np.std(
                        [
                            runs[name]["policies"][policy]["test"]["metrics"][key]
                            - runs[name]["policies"]["global"]["test"]["metrics"][key]
                            for name in runs
                        ],
                        ddof=0,
                    )
                ),
            }
            for key in ("accuracy", "false_trustworthy_rate", "recall_trustworthy", "precision_trustworthy")
        }
    return out


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def mean_std(row: dict[str, float]) -> str:
    return f"{pct(row['mean'])} ± {pct(row['std'])}"


def modality_thresholds_text(thresholds: dict[str, float]) -> str:
    return ", ".join(f"`{key}`={value:.2f}" for key, value in sorted(thresholds.items()))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Modality Threshold Sweep",
        "",
        "Local-only diagnostic. Candidate and patch labels remain trusted only for local controls.",
        "",
        f"- Data: `{payload['data_dir']}`",
        f"- Target FT for release-gate policy: `{payload['target_ft']:.3f}`",
        "",
        "## Aggregate",
        "",
        "| Policy | Test accuracy | Test FT | Delta acc | Delta FT |",
        "|---|---:|---:|---:|---:|",
    ]
    aggregate = payload["aggregate"]
    baseline = aggregate["global"]["test"]
    for policy, row in aggregate.items():
        test = row["test"]
        if policy == "global":
            delta_acc = ""
            delta_ft = ""
        else:
            delta = row["delta_vs_global_test"]
            delta_acc = f"{delta['accuracy']['mean'] * 100.0:+.2f} pp"
            delta_ft = f"{delta['false_trustworthy_rate']['mean'] * 100.0:+.2f} pp"
        lines.append(
            f"| `{policy}` | {mean_std(test['accuracy'])} | "
            f"{mean_std(test['false_trustworthy_rate'])} | {delta_acc} | {delta_ft} |"
        )

    lines.extend(["", "## Per Seed", ""])
    for run_name, run in payload["runs"].items():
        lines.extend([f"### {run_name}", ""])
        lines.append(f"- Global tau: `{run['global_threshold']:.2f}`")
        for policy, row in run["policies"].items():
            metrics = row["test"]["metrics"]
            delta = row.get("delta_vs_global_test")
            if delta:
                suffix = (
                    f"; delta acc {delta['accuracy'] * 100.0:+.2f} pp, "
                    f"delta FT {delta['false_trustworthy_rate'] * 100.0:+.2f} pp"
                )
            else:
                suffix = ""
            lines.append(
                f"- `{policy}` thresholds: {modality_thresholds_text(row['thresholds'])}; "
                f"test {pct(metrics['accuracy'])} / {pct(metrics['false_trustworthy_rate'])} FT"
                f"{suffix}"
            )
        lines.append("")

    better = [
        policy
        for policy, row in aggregate.items()
        if policy != "global"
        and row["delta_vs_global_test"]["accuracy"]["mean"] > 0.0
        and row["delta_vs_global_test"]["false_trustworthy_rate"]["mean"] <= 0.0
    ]
    lines.extend(["## Readout", ""])
    if better:
        lines.append(
            "At least one per-modality threshold policy improves mean test accuracy without raising mean FT: "
            + ", ".join(f"`{policy}`" for policy in better)
            + "."
        )
    else:
        lines.append(
            "No per-modality threshold policy improves mean test accuracy without raising mean FT versus the global retry-patch threshold."
        )
    lines.append(
        "This is policy-only local evidence; full blind-label QA remains required before merge or publish."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start_time = time.time()
    args = parse_args()
    runs = configured_runs(args)
    device = resolve_device(args.device)
    ds = load_processed(args.data_dir)
    if "eval" not in ds or "test" not in ds:
        raise ValueError(f"{args.data_dir} must contain eval and test splits")

    grid = np.linspace(args.grid_min, args.grid_max, args.grid_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "device": device,
        "target_ft": args.target_ft,
        "grid": {"min": args.grid_min, "max": args.grid_max, "size": args.grid_size},
        "label_trusted_local_only": True,
        "runs": {},
    }

    for run_name, run in runs.items():
        payload["runs"][run_name] = evaluate_run(
            run_name,
            run["checkpoint"],
            run["threshold_source"],
            ds,
            args,
            device,
            grid,
        )

    payload["aggregate"] = aggregate_runs(payload["runs"])
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(report_path, payload)

    fitz_gov_repo = Path("C:/Users/yanfi/PycharmProjects/fitz-gov")
    manifest_path = write_manifest(
        output_dir=args.output_dir,
        config_path=Path(__file__).resolve(),
        seed=args.seed,
        fitz_gov_repo=fitz_gov_repo if fitz_gov_repo.exists() else None,
        extra={
            "task": "modality_threshold_sweep",
            "data_dir": str(args.data_dir),
            "runs": {
                name: {
                    "checkpoint": str(run["checkpoint"]),
                    "threshold_source": str(run["threshold_source"]),
                }
                for name, run in runs.items()
            },
            "label_trusted_local_only": True,
        },
        start_time=start_time,
    )

    print(f"Wrote summary : {summary_path}")
    print(f"Wrote report  : {report_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
