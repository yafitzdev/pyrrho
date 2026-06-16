"""Compare simple modality-specialist routing against a joint generalist.

This is a local diagnostic harness for the structured/code modality branch. It
does not train, generate rows, or change schema. It evaluates fixed checkpoints
on a processed dataset and routes predictions by the existing ``modality``
column:

- unstructured -> joint generalist
- code -> optional code specialist
- structured -> optional structured specialist

Default paths compare the retry-patch seed-42 generalist against the existing
seed-42 code-only and structured-only local controls.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL, load_processed
from pyrrho.manifest import write_manifest
from pyrrho.metrics import breakdown_by, compute_classification_metrics, gated_predictions

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_DATA_DIR = Path("data/processed_v8_plus_structured_code_retry_patch_candidate")
DEFAULT_OUTPUT_DIR = Path("outputs/modality_specialist_compare/retry_patch_seed42_router")

DEFAULT_GENERALIST = Path("outputs/modality_retraining/structured_code_retry_patch_seed42/best_model")
DEFAULT_GENERALIST_THRESHOLD = Path(
    "outputs/modality_retraining/structured_code_retry_patch_seed42/eval_report.json"
)
DEFAULT_CODE_SPECIALIST = Path("outputs/modality_retraining/code_seed42/best_model")
DEFAULT_CODE_THRESHOLD = Path("outputs/modality_retraining/code_seed42/final_metrics.json")
DEFAULT_STRUCTURED_SPECIALIST = Path("outputs/modality_retraining/structured_seed42/best_model")
DEFAULT_STRUCTURED_THRESHOLD = Path("outputs/modality_retraining/structured_seed42/final_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generalist-checkpoint", type=Path, default=DEFAULT_GENERALIST)
    parser.add_argument("--generalist-threshold-source", type=Path, default=DEFAULT_GENERALIST_THRESHOLD)
    parser.add_argument("--generalist-threshold", type=float, default=None)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_SPECIALIST)
    parser.add_argument("--code-threshold-source", type=Path, default=DEFAULT_CODE_THRESHOLD)
    parser.add_argument("--code-threshold", type=float, default=None)
    parser.add_argument("--structured-checkpoint", type=Path, default=DEFAULT_STRUCTURED_SPECIALIST)
    parser.add_argument("--structured-threshold-source", type=Path, default=DEFAULT_STRUCTURED_THRESHOLD)
    parser.add_argument("--structured-threshold", type=float, default=None)
    parser.add_argument("--splits", nargs="+", default=["eval", "test"])
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Manifest seed for this fixed-checkpoint diagnostic.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def load_threshold(explicit: float | None, source: Path) -> float:
    if explicit is not None:
        return float(explicit)
    if not source.exists():
        raise FileNotFoundError(f"Threshold source not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "threshold" in payload:
        return float(payload["threshold"])
    if "eval_calibrated" in payload and "threshold" in payload["eval_calibrated"]:
        return float(payload["eval_calibrated"]["threshold"])
    raise KeyError(f"No threshold field found in {source}")


def infer_preds(
    checkpoint: Path,
    texts: list[str],
    threshold: float,
    max_seq_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    preds: list[np.ndarray] = []
    num_classes = int(model.config.num_labels)
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
            logits = model(**enc).logits.float().cpu().numpy()
            preds.append(gated_predictions(logits, threshold, num_classes=num_classes))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0)


def count_values(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def summarize_policy(
    preds: np.ndarray,
    labels: np.ndarray,
    modalities: list[str],
    baseline: dict[str, float] | None,
) -> dict[str, Any]:
    metrics = compute_classification_metrics(preds, labels)
    row: dict[str, Any] = {
        "metrics": metrics,
        "modalities": breakdown_by(preds, labels, modalities),
    }
    if baseline is not None:
        row["delta_vs_generalist"] = {
            "accuracy": metrics["accuracy"] - baseline["accuracy"],
            "false_trustworthy_rate": metrics["false_trustworthy_rate"]
            - baseline["false_trustworthy_rate"],
            "recall_trustworthy": metrics["recall_trustworthy"] - baseline["recall_trustworthy"],
            "precision_trustworthy": metrics["precision_trustworthy"]
            - baseline["precision_trustworthy"],
        }
    return row


def route_predictions(
    generalist: np.ndarray,
    code: np.ndarray,
    structured: np.ndarray,
    modalities: np.ndarray,
    *,
    use_code: bool,
    use_structured: bool,
) -> np.ndarray:
    routed = generalist.copy()
    if use_code:
        mask = modalities == "code"
        routed[mask] = code[mask]
    if use_structured:
        mask = modalities == "structured"
        routed[mask] = structured[mask]
    return routed


def confusion(preds: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {ID2LABEL[g]: {ID2LABEL[p]: 0 for p in range(3)} for g in range(3)}
    for gold, pred in zip(labels.tolist(), preds.tolist()):
        out[ID2LABEL[gold]][ID2LABEL[pred]] += 1
    return out


def evaluate_split(
    split_name: str,
    dataset: Dataset,
    model_preds: dict[str, np.ndarray],
) -> dict[str, Any]:
    labels = np.asarray(dataset["label_id"])
    modalities = list(dataset["modality"]) if "modality" in dataset.column_names else ["unstructured"] * len(dataset)
    modality_arr = np.asarray(modalities)

    policies = {
        "retry_patch_generalist": model_preds["generalist"],
        "route_code_specialist": route_predictions(
            model_preds["generalist"],
            model_preds["code"],
            model_preds["structured"],
            modality_arr,
            use_code=True,
            use_structured=False,
        ),
        "route_structured_specialist": route_predictions(
            model_preds["generalist"],
            model_preds["code"],
            model_preds["structured"],
            modality_arr,
            use_code=False,
            use_structured=True,
        ),
        "route_code_structured_specialists": route_predictions(
            model_preds["generalist"],
            model_preds["code"],
            model_preds["structured"],
            modality_arr,
            use_code=True,
            use_structured=True,
        ),
    }

    baseline_metrics = compute_classification_metrics(policies["retry_patch_generalist"], labels)
    out: dict[str, Any] = {
        "n": len(dataset),
        "modality_counts": count_values(modalities),
        "policies": {},
    }
    for name, preds in policies.items():
        baseline = None if name == "retry_patch_generalist" else baseline_metrics
        out["policies"][name] = summarize_policy(preds, labels, modalities, baseline)
        out["policies"][name]["confusion"] = confusion(preds, labels)
    return out


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def modality_cell(row: dict[str, Any], modality: str) -> str:
    stats = row["modalities"].get(modality)
    if not stats:
        return "-"
    return f"{pct(stats['accuracy'])} / {pct(stats['false_trustworthy_rate'])}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Modality Specialist Comparison",
        "",
        "Local-only diagnostic. Candidate and patch labels remain trusted only for local controls.",
        "",
        "## Inputs",
        "",
        f"- Data: `{payload['data_dir']}`",
        f"- Generalist: `{payload['checkpoints']['generalist']['path']}` "
        f"(tau `{payload['checkpoints']['generalist']['threshold']:.2f}`)",
        f"- Code specialist: `{payload['checkpoints']['code']['path']}` "
        f"(tau `{payload['checkpoints']['code']['threshold']:.2f}`)",
        f"- Structured specialist: `{payload['checkpoints']['structured']['path']}` "
        f"(tau `{payload['checkpoints']['structured']['threshold']:.2f}`)",
        "",
    ]

    for split_name, split in payload["splits"].items():
        lines.extend(
            [
                f"## {split_name.upper()} Split",
                "",
                f"Rows: **{split['n']}**; modalities: "
                + ", ".join(f"`{k}`={v}" for k, v in split["modality_counts"].items()),
                "",
                "| Policy | Accuracy | FT | Code | Structured | Unstructured | Delta acc | Delta FT |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in split["policies"].items():
            metrics = row["metrics"]
            delta = row.get("delta_vs_generalist", {})
            delta_acc = "" if not delta else f"{delta['accuracy'] * 100.0:+.2f} pp"
            delta_ft = "" if not delta else f"{delta['false_trustworthy_rate'] * 100.0:+.2f} pp"
            lines.append(
                f"| `{policy}` | {pct(metrics['accuracy'])} | "
                f"{pct(metrics['false_trustworthy_rate'])} | "
                f"{modality_cell(row, 'code')} | "
                f"{modality_cell(row, 'structured')} | "
                f"{modality_cell(row, 'unstructured')} | "
                f"{delta_acc} | {delta_ft} |"
            )
        lines.append("")

    test = payload["splits"].get("test")
    if test:
        baseline = test["policies"]["retry_patch_generalist"]["metrics"]
        routed = [
            (name, row)
            for name, row in test["policies"].items()
            if name != "retry_patch_generalist"
        ]
        better = [
            name
            for name, row in routed
            if row["metrics"]["accuracy"] > baseline["accuracy"]
            and row["metrics"]["false_trustworthy_rate"] <= baseline["false_trustworthy_rate"]
        ]
        lines.extend(["## Readout", ""])
        if better:
            lines.append(
                "At least one routed policy beats the seed-matched generalist on accuracy without raising FT: "
                + ", ".join(f"`{name}`" for name in better)
                + "."
            )
        else:
            lines.append(
                "No routed specialist policy beats the seed-matched retry-patch generalist on test accuracy while keeping FT no worse."
            )
        lines.append(
            "This is a one-seed specialist diagnostic, not release evidence; full blind-label QA remains required before merge or publish."
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start_time = time.time()
    args = parse_args()
    device = resolve_device(args.device)
    thresholds = {
        "generalist": load_threshold(args.generalist_threshold, args.generalist_threshold_source),
        "code": load_threshold(args.code_threshold, args.code_threshold_source),
        "structured": load_threshold(args.structured_threshold, args.structured_threshold_source),
    }
    checkpoints = {
        "generalist": args.generalist_checkpoint,
        "code": args.code_checkpoint,
        "structured": args.structured_checkpoint,
    }

    ds = load_processed(args.data_dir)
    missing = [split for split in args.splits if split not in ds]
    if missing:
        raise ValueError(f"Requested splits not present in {args.data_dir}: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "device": device,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "label_trusted_local_only": True,
        "checkpoints": {
            name: {"path": str(path), "threshold": thresholds[name]}
            for name, path in checkpoints.items()
        },
        "splits": {},
    }

    for split_name in args.splits:
        split = ds[split_name]
        texts = list(split["text"])
        print(f"[{split_name}] rows={len(split)}")
        model_preds: dict[str, np.ndarray] = {}
        for name, checkpoint in checkpoints.items():
            print(f"  scoring {name}: {checkpoint} (tau={thresholds[name]:.2f})")
            model_preds[name] = infer_preds(
                checkpoint=checkpoint,
                texts=texts,
                threshold=thresholds[name],
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                device=device,
            )
        payload["splits"][split_name] = evaluate_split(split_name, split, model_preds)

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
            "task": "modality_specialist_compare",
            "data_dir": str(args.data_dir),
            "splits": list(args.splits),
            "checkpoints": payload["checkpoints"],
            "label_trusted_local_only": True,
        },
        start_time=start_time,
    )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report : {report_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
