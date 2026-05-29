"""Score pyrrho on fitz-gov structured/code candidate packs.

The candidate rows live in the fitz-gov repo and are not active training rows.
This script builds reproducible candidate manifests, then evaluates a pyrrho
encoder checkpoint on a selected manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL, LABEL2ID, build_encoder_text
from pyrrho.metrics import compute_classification_metrics, gated_predictions

DEFAULT_INPUTS = [
    Path(
        "C:/Users/yanfi/PycharmProjects/fitz-gov/"
        "data/_workspaces/handoff/modality_structured_v1_20260527/cases.jsonl"
    ),
    Path(
        "C:/Users/yanfi/PycharmProjects/fitz-gov/"
        "data/_workspaces/handoff/modality_code_v1_20260527/cases.jsonl"
    ),
]
DEFAULT_MODEL = Path("models/pyrrho-nano-g3")
DEFAULT_THRESHOLD_SOURCE = Path("outputs/multi_seed_g3_v8/seed_1337/final_metrics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/modality_candidate_probe/g3_release")

MANIFEST_KEYS = {
    "full_20k": (),
    "balanced_pattern": ("pattern",),
    "balanced_modality_pattern": ("modality", "pattern"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-source", type=Path, default=DEFAULT_THRESHOLD_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset", choices=sorted(MANIFEST_KEYS), default="full_20k")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--manifests-only",
        action="store_true",
        help="Write derived manifests and summaries, then skip model scoring.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def scalar(case: Mapping[str, Any], *path: str, default: str = "unknown") -> str:
    value: Any = case
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key)
    return str(value if value not in (None, "") else default)


def contexts(case: Mapping[str, Any]) -> list[str]:
    input_obj = case.get("input") if isinstance(case.get("input"), Mapping) else {}
    raw_contexts = input_obj.get("contexts") if isinstance(input_obj, Mapping) else []
    out: list[str] = []
    if not isinstance(raw_contexts, list):
        return out
    for item in raw_contexts:
        if isinstance(item, Mapping):
            out.append(str(item.get("text") or ""))
        else:
            out.append(str(item))
    return out


def row_from_case(case: Mapping[str, Any]) -> dict[str, Any]:
    label = scalar(case, "governance", "classification", default="UNKNOWN").upper()
    if label not in LABEL2ID:
        raise ValueError(f"{case.get('id')}: invalid label {label!r}")
    return {
        "case_id": str(case.get("id") or ""),
        "query": scalar(case, "input", "query", default=""),
        "contexts": contexts(case),
        "label": label,
        "label_id": LABEL2ID[label],
        "modality": scalar(case, "meta", "modality"),
        "pattern": scalar(case, "taxonomy", "pattern"),
        "domain": scalar(case, "routing", "expert_fired"),
        "difficulty": scalar(case, "meta", "difficulty"),
        "mechanism": scalar(case, "meta", "mechanism"),
        "dataset_version": scalar(case, "meta", "dataset_version"),
    }


def load_rows(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for case in read_jsonl(path):
            row = row_from_case(case)
            if not row["case_id"]:
                raise ValueError(f"{path}: row without id")
            if row["case_id"] in seen:
                raise ValueError(f"duplicate case id: {row['case_id']}")
            seen.add(row["case_id"])
            rows.append(row)
    return rows


def manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "label": row["label"],
        "modality": row["modality"],
        "pattern": row["pattern"],
        "domain": row["domain"],
        "difficulty": row["difficulty"],
        "mechanism": row["mechanism"],
        "dataset_version": row["dataset_version"],
    }


def stable_sample(rows: list[dict[str, Any]], *, keys: tuple[str, ...], seed: int) -> list[dict[str, Any]]:
    if not keys:
        return sorted(rows, key=lambda row: str(row["case_id"]))

    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[key]) for key in keys)].append(row)
    target = min(len(bucket) for bucket in buckets.values())

    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda row: str(row["case_id"]))
        indexes = rng.permutation(len(bucket))[:target]
        selected.extend(bucket[int(i)] for i in indexes)
    return sorted(selected, key=lambda row: str(row["case_id"]))


def counter(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def summarize_manifest(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": counter(rows, "label"),
        "modalities": counter(rows, "modality"),
        "patterns": counter(rows, "pattern"),
        "domains": counter(rows, "domain"),
        "difficulties": counter(rows, "difficulty"),
        "mechanisms": counter(rows, "mechanism"),
    }


def write_manifests(
    rows: list[dict[str, Any]], *, output_dir: Path, seed: int
) -> dict[str, dict[str, Any]]:
    manifest_dir = output_dir / "manifests"
    summaries: dict[str, dict[str, Any]] = {}
    for name, keys in MANIFEST_KEYS.items():
        selected = stable_sample(rows, keys=keys, seed=seed)
        path = manifest_dir / f"{name}.jsonl"
        write_jsonl(path, [manifest_row(row) for row in selected])
        summary = summarize_manifest(selected)
        summary["path"] = str(path)
        summary["balance_keys"] = list(keys)
        if keys:
            buckets = Counter(tuple(str(row[key]) for key in keys) for row in rows)
            summary["rows_per_balanced_bucket"] = min(buckets.values())
            summary["balanced_bucket_count"] = len(buckets)
        summaries[name] = summary
    write_json(manifest_dir / "summary.json", summaries)
    return summaries


def load_manifest_ids(path: Path) -> set[str]:
    return {str(row.get("case_id") or "") for row in read_jsonl(path)}


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return name


def load_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.threshold_source.exists():
        metrics = json.loads(args.threshold_source.read_text(encoding="utf-8"))
        return float(metrics["threshold"])
    raise FileNotFoundError(f"threshold source not found: {args.threshold_source}")


def softmax_batch(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def group_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row[key])].append(idx)

    out = {}
    for value, indexes in sorted(grouped.items()):
        labels = np.asarray([rows[i]["label_id"] for i in indexes], dtype=np.int64)
        preds = np.asarray([predictions[i]["calibrated_pred_id"] for i in indexes], dtype=np.int64)
        out[value] = {
            "n": len(indexes),
            **compute_classification_metrics(preds, labels),
        }
    return out


def confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, int]]:
    out = {ID2LABEL[g]: {ID2LABEL[p]: 0 for p in range(3)} for g in range(3)}
    for gold, pred in zip(labels.tolist(), preds.tolist(), strict=True):
        out[ID2LABEL[int(gold)]][ID2LABEL[int(pred)]] += 1
    return out


def score_rows(
    *,
    rows: list[dict[str, Any]],
    model_path: Path,
    threshold: float,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    predictions: list[dict[str, Any]] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch_no, batch in enumerate(batched(rows, batch_size), start=1):
            texts = [build_encoder_text(row["query"], row["contexts"]) for row in batch]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float().cpu().numpy()
            all_logits.append(logits)
            if batch_no % 50 == 0:
                done = min(batch_no * batch_size, len(rows))
                print(f"  scored {done}/{len(rows)} rows", flush=True)

    logits = np.concatenate(all_logits, axis=0)
    probs = softmax_batch(logits)
    raw_preds = gated_predictions(logits, 0.0, num_classes=logits.shape[-1])
    cal_preds = gated_predictions(logits, threshold, num_classes=logits.shape[-1])
    labels = np.asarray([row["label_id"] for row in rows], dtype=np.int64)

    for idx, row in enumerate(rows):
        predictions.append(
            {
                "case_id": row["case_id"],
                "label": row["label"],
                "raw_pred": ID2LABEL[int(raw_preds[idx])],
                "calibrated_pred": ID2LABEL[int(cal_preds[idx])],
                "raw_pred_id": int(raw_preds[idx]),
                "calibrated_pred_id": int(cal_preds[idx]),
                "ok": bool(int(cal_preds[idx]) == int(row["label_id"])),
                "p_abstain": float(probs[idx, 0]),
                "p_disputed": float(probs[idx, 1]),
                "p_trustworthy": float(probs[idx, 2]),
                "modality": row["modality"],
                "pattern": row["pattern"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "mechanism": row["mechanism"],
            }
        )

    summary = {
        "model": str(model_path),
        "threshold": threshold,
        "device": device,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "rows": len(rows),
        "manifest": summarize_manifest(rows),
        "raw": compute_classification_metrics(raw_preds, labels),
        "calibrated": compute_classification_metrics(cal_preds, labels),
        "confusion_calibrated": confusion_matrix(labels, cal_preds),
        "breakdowns_calibrated": {
            "modality": group_metrics(rows, predictions, "modality"),
            "label": group_metrics(rows, predictions, "label"),
            "pattern": group_metrics(rows, predictions, "pattern"),
            "domain": group_metrics(rows, predictions, "domain"),
            "difficulty": group_metrics(rows, predictions, "difficulty"),
            "mechanism": group_metrics(rows, predictions, "mechanism"),
        },
    }
    return summary, predictions


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def metric_row(name: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| `{name}` | {metrics['n']} | {pct(float(metrics['accuracy']))} | "
        f"{pct(float(metrics['false_trustworthy_rate']))} | "
        f"{pct(float(metrics['recall_abstain']))} | "
        f"{pct(float(metrics['recall_disputed']))} | "
        f"{pct(float(metrics['recall_trustworthy']))} |"
    )


def write_report(path: Path, summary: Mapping[str, Any], predictions: list[dict[str, Any]]) -> None:
    cal = summary["calibrated"]
    raw = summary["raw"]
    lines = [
        "# Modality Candidate Probe",
        "",
        f"- Model: `{summary['model']}`",
        f"- Threshold: `{summary['threshold']:.4f}`",
        f"- Rows: **{summary['rows']}**",
        f"- Calibrated accuracy: **{pct(float(cal['accuracy']))}**",
        f"- Calibrated false-TRUSTWORTHY: **{pct(float(cal['false_trustworthy_rate']))}**",
        f"- Raw accuracy: **{pct(float(raw['accuracy']))}**",
        f"- Raw false-TRUSTWORTHY: **{pct(float(raw['false_trustworthy_rate']))}**",
        "",
        "## By Modality",
        "",
        "| Bucket | n | accuracy | FT | A recall | D recall | T recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["breakdowns_calibrated"]["modality"].items():
        lines.append(metric_row(name, metrics))

    lines.extend(
        [
            "",
            "## By Label",
            "",
            "| Bucket | n | accuracy | FT | A recall | D recall | T recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in summary["breakdowns_calibrated"]["label"].items():
        lines.append(metric_row(name, metrics))

    lines.extend(
        [
            "",
            "## By Pattern",
            "",
            "| Bucket | n | accuracy | FT | A recall | D recall | T recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in summary["breakdowns_calibrated"]["pattern"].items():
        lines.append(metric_row(name, metrics))

    wrong = [row for row in predictions if not row["ok"]]
    lines.extend(
        [
            "",
            "## Error Preview",
            "",
            "| Case | Modality | Pattern | Gold | Pred | P(A) | P(D) | P(T) |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in wrong[:100]:
        lines.append(
            f"| `{row['case_id']}` | `{row['modality']}` | `{row['pattern']}` | "
            f"`{row['label']}` | `{row['calibrated_pred']}` | "
            f"{row['p_abstain']:.3f} | {row['p_disputed']:.3f} | {row['p_trustworthy']:.3f} |"
        )
    if not wrong:
        lines.append("| none |  |  |  |  |  |  |  |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    inputs = args.input or DEFAULT_INPUTS
    rows = load_rows(inputs)
    manifest_summaries = write_manifests(rows, output_dir=args.output_dir, seed=args.seed)
    if args.manifests_only:
        print(f"Wrote manifests: {args.output_dir / 'manifests'}")
        return 0

    subset_manifest = args.output_dir / "manifests" / f"{args.subset}.jsonl"
    subset_ids = load_manifest_ids(subset_manifest)
    selected = [row for row in rows if row["case_id"] in subset_ids]
    selected.sort(key=lambda row: str(row["case_id"]))

    device = resolve_device(args.device)
    threshold = load_threshold(args)
    run_dir = args.output_dir / args.subset
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inputs      : {len(inputs)}")
    print(f"Rows        : {len(rows)}")
    print(f"Subset      : {args.subset} ({len(selected)} rows)")
    print(f"Subset info : {manifest_summaries[args.subset]}")
    print(f"Model       : {args.model}")
    print(f"Device      : {device}")
    print(f"Threshold   : {threshold:.4f}")

    summary, predictions = score_rows(
        rows=selected,
        model_path=args.model,
        threshold=threshold,
        device=device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    write_json(run_dir / "summary.json", summary)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_report(run_dir / "report.md", summary, predictions)

    cal = summary["calibrated"]
    print(f"Accuracy    : {pct(float(cal['accuracy']))}")
    print(f"False-T rate: {pct(float(cal['false_trustworthy_rate']))}")
    print(f"Wrote       : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
