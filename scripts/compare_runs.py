"""compare_runs.py — Diff metrics across two pyrrho runs side-by-side.

Each `run` is either:
- a path to a `final_metrics.json` (single-run output of `scripts/train_encoder.py`)
- a path to a `summary.json` (multi-seed output of `scripts/run_seeds.py`)
- the literal string `baseline` (uses the fitz-sage v0.11 sklearn cascade as constant numbers)

Output: a markdown comparison table to stdout + (optional) JSON file.

Run from project root:
    python scripts/compare_runs.py baseline outputs/modernbert_base_v1/final_metrics.json
    python scripts/compare_runs.py outputs/multi_seed/summary.json baseline
    python scripts/compare_runs.py outputs/a/final_metrics.json outputs/b/final_metrics.json --output diff.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# fitz-sage v0.11 (sklearn cascade) on fitz-gov v5, 5-fold CV. Per README.md L340-346.
BASELINE = {
    "label": "fitz-sage v0.11 sklearn baseline",
    "calibrated": {
        "accuracy": 0.787,
        "macro_f1": None,
        "recall_abstain": 0.865,
        "recall_disputed": 0.861,
        "recall_trustworthy": 0.700,
        "precision_abstain": None,
        "precision_disputed": None,
        "precision_trustworthy": None,
        "false_trustworthy_rate": 0.057,
    },
    "uncalibrated": None,
    "tier0_calibrated": None,
    "tier0_uncalibrated": None,
    "multi_seed": False,
}


METRIC_ORDER = (
    "accuracy",
    "macro_f1",
    "recall_abstain",
    "recall_disputed",
    "recall_trustworthy",
    "precision_abstain",
    "precision_disputed",
    "precision_trustworthy",
    "false_trustworthy_rate",
)


def load_run(spec: str) -> dict[str, Any]:
    """Normalize a CLI spec to a uniform run dict with `calibrated`, `uncalibrated`, tier0 variants."""
    if spec.lower() == "baseline":
        return BASELINE

    path = Path(spec).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Detect format: single-run final_metrics.json vs multi-seed summary.json
    if "aggregate" in data and "per_seed" in data:
        # multi-seed summary.json
        agg = data["aggregate"]
        seeds = data.get("seeds", [])
        return {
            "label": f"{path.parent.name} (seeds {seeds})",
            "calibrated": {k: agg["eval_calibrated"][k] for k in agg["eval_calibrated"]},
            "uncalibrated": {k: agg["eval_uncalibrated"][k] for k in agg["eval_uncalibrated"]},
            "tier0_calibrated": agg.get("tier0_calibrated"),
            "tier0_uncalibrated": agg.get("tier0_uncalibrated"),
            "multi_seed": True,
        }
    elif "eval_calibrated" in data:
        # single final_metrics.json
        return {
            "label": str(path.parent.name),
            "calibrated": data["eval_calibrated"],
            "uncalibrated": data["eval_uncalibrated"],
            "tier0_calibrated": data.get("tier0_calibrated"),
            "tier0_uncalibrated": data.get("tier0_uncalibrated"),
            "multi_seed": False,
        }
    else:
        raise ValueError(f"Unrecognized JSON schema in {path}. Expected final_metrics.json or summary.json.")


def fmt_value(v: Any, multi_seed: bool) -> str:
    if v is None:
        return "    —    "
    if isinstance(v, dict):
        m = v.get("mean")
        s = v.get("std")
        if m is None:
            return "    —    "
        if multi_seed and s is not None:
            return f"{m * 100:5.2f} ± {s * 100:4.2f}"
        return f"{m * 100:7.2f}  "
    if isinstance(v, (int, float)):
        return f"{v * 100:7.2f}  "
    return str(v)


def get_metric(run: dict, split: str, metric: str) -> Any:
    block = run.get(split)
    if block is None:
        return None
    return block.get(metric)


def get_mean(run: dict, split: str, metric: str) -> float | None:
    v = get_metric(run, split, metric)
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("mean")
    return float(v)


def fmt_delta(delta: float | None, polarity: str) -> str:
    """polarity='good_up' for metrics where higher is better; 'good_down' for FT."""
    if delta is None:
        return "    —    "
    arrow = ""
    if polarity == "good_up":
        arrow = "✓" if delta > 0 else ("✗" if delta < 0 else "·")
    else:
        arrow = "✓" if delta < 0 else ("✗" if delta > 0 else "·")
    return f"{delta * 100:+6.2f} {arrow}"


def print_comparison(run_a: dict, run_b: dict, split: str, header: str) -> None:
    print(f"\n## {header}\n")
    print(f"| Metric | {run_a['label']} | {run_b['label']} | Δ (b − a) |")
    print(f"|---|---|---|---|")
    for metric in METRIC_ORDER:
        a_v = get_metric(run_a, split, metric)
        b_v = get_metric(run_b, split, metric)
        a_mean = get_mean(run_a, split, metric)
        b_mean = get_mean(run_b, split, metric)
        delta = (b_mean - a_mean) if (a_mean is not None and b_mean is not None) else None
        polarity = "good_down" if metric == "false_trustworthy_rate" else "good_up"
        print(
            f"| {metric} | {fmt_value(a_v, run_a['multi_seed'])} | "
            f"{fmt_value(b_v, run_b['multi_seed'])} | {fmt_delta(delta, polarity)} |"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=str, help="Path to final_metrics.json or summary.json, or 'baseline'")
    p.add_argument("run_b", type=str, help="Path to final_metrics.json or summary.json, or 'baseline'")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON to write the comparison to")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)

    print(f"# pyrrho run comparison")
    print(f"\n- **A**: {run_a['label']}")
    print(f"- **B**: {run_b['label']}")
    print("\nAll cells show percentages. Δ column is `(B - A)`.  "
          "`✓` = improvement, `✗` = regression, `·` = tie. FT rate is graded with lower-is-better.")

    print_comparison(run_a, run_b, "calibrated", "EVAL — calibrated")
    print_comparison(run_a, run_b, "uncalibrated", "EVAL — uncalibrated")

    if run_a.get("tier0_calibrated") or run_b.get("tier0_calibrated"):
        print_comparison(run_a, run_b, "tier0_calibrated", "TIER0 — calibrated")

    if args.output:
        payload = {"run_a": run_a, "run_b": run_b}
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote comparison JSON -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
