"""tabular_ood_probe.py - serialized structured-evidence OOD probe.

Builds a small hand-labeled tabular test set, serializes each scenario in
three common retrieval formats, and scores a pyrrho encoder checkpoint.

Run from project root:
    python scripts/tabular_ood_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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

SERIALIZATIONS = ("markdown_table", "csv_extract", "evidence_packet")
DEFAULT_MODEL = Path("models/pyrrho-nano-g3")
DEFAULT_THRESHOLD_SOURCE = Path("outputs/multi_seed_g3_v8/seed_1337/final_metrics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/tabular_ood_probe/g3_release")


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "tab_01_revenue_exact",
        "expected": "TRUSTWORTHY",
        "pattern": "direct_answer",
        "mechanism": "exact_filtered_row",
        "query": "What was Acme Europe's net revenue in Q4 2025?",
        "sources": [
            {
                "title": "warehouse.finance_revenue_quarterly",
                "filters": "company=Acme; region=Europe; quarter=2025-Q4; metric=net_revenue",
                "columns": ["company", "region", "quarter", "net_revenue_usd_millions", "reporting_basis"],
                "rows": [["Acme", "Europe", "2025-Q4", "42.1", "net revenue"]],
                "notes": "Retrieved result table for the exact company, region, quarter, and metric.",
            }
        ],
    },
    {
        "id": "tab_02_active_paid_customers",
        "expected": "TRUSTWORTHY",
        "pattern": "direct_answer",
        "mechanism": "exact_filtered_row",
        "query": "How many active paid customers did Northwind have in March 2026?",
        "sources": [
            {
                "title": "mart.customer_monthly_status",
                "filters": "account=Northwind; month=2026-03; customer_status=active; plan_type=paid",
                "columns": ["account", "month", "active_paid_customers", "definition"],
                "rows": [["Northwind", "2026-03", "18,420", "active paid customers at month end"]],
                "notes": "The extract is already filtered to active paid customers only.",
            }
        ],
    },
    {
        "id": "tab_03_reorder_sku",
        "expected": "TRUSTWORTHY",
        "pattern": "consistent_chain",
        "mechanism": "row_comparison",
        "query": "Which SKU was below its reorder point at DC-West on May 1 2026?",
        "sources": [
            {
                "title": "inventory.daily_stock_position",
                "filters": "warehouse=DC-West; date=2026-05-01",
                "columns": ["sku", "warehouse", "date", "on_hand_units", "reorder_point_units"],
                "rows": [
                    ["SKU-R17", "DC-West", "2026-05-01", "44", "75"],
                    ["SKU-M04", "DC-West", "2026-05-01", "310", "250"],
                ],
                "notes": "Only SKU-R17 has on_hand_units below reorder_point_units.",
            }
        ],
    },
    {
        "id": "tab_04_sla_availability",
        "expected": "TRUSTWORTHY",
        "pattern": "quantitative_consensus",
        "mechanism": "threshold_comparison",
        "query": "Did the payments API meet the 99.9% availability SLA in April 2026?",
        "sources": [
            {
                "title": "observability.monthly_sla_rollup",
                "filters": "service=payments-api; month=2026-04",
                "columns": ["service", "month", "availability_percent", "sla_target_percent", "status"],
                "rows": [["payments-api", "2026-04", "99.95", "99.90", "met"]],
                "notes": "The availability value is higher than the target and the rollup status is met.",
            }
        ],
    },
    {
        "id": "tab_05_revenue_conflict",
        "expected": "DISPUTED",
        "pattern": "numerical_conflict",
        "mechanism": "same_metric_different_values",
        "query": "What was Acme Europe's net revenue in Q4 2025?",
        "sources": [
            {
                "title": "warehouse.finance_revenue_quarterly",
                "filters": "company=Acme; region=Europe; quarter=2025-Q4; metric=net_revenue",
                "columns": ["company", "region", "quarter", "net_revenue_usd_millions"],
                "rows": [["Acme", "Europe", "2025-Q4", "42.1"]],
                "notes": "Finance warehouse published value.",
            },
            {
                "title": "bi_export.board_revenue_pack",
                "filters": "company=Acme; region=Europe; quarter=2025-Q4; metric=net_revenue",
                "columns": ["company", "region", "quarter", "net_revenue_usd_millions"],
                "rows": [["Acme", "Europe", "2025-Q4", "38.9"]],
                "notes": "Board pack published value for the same metric and filters.",
            },
        ],
    },
    {
        "id": "tab_06_churn_conflict",
        "expected": "DISPUTED",
        "pattern": "numerical_conflict",
        "mechanism": "same_metric_different_values",
        "query": "What was Northwind's logo churn rate for paid accounts in April 2026?",
        "sources": [
            {
                "title": "mart.retention_monthly",
                "filters": "account=Northwind; month=2026-04; cohort=paid_accounts",
                "columns": ["account", "month", "logo_churn_rate_percent"],
                "rows": [["Northwind", "2026-04", "2.4"]],
                "notes": "Retention mart output.",
            },
            {
                "title": "finance.kpi_close_packet",
                "filters": "account=Northwind; month=2026-04; cohort=paid_accounts",
                "columns": ["account", "month", "logo_churn_rate_percent"],
                "rows": [["Northwind", "2026-04", "5.1"]],
                "notes": "Closed KPI packet output for the same paid-account cohort.",
            },
        ],
    },
    {
        "id": "tab_07_status_conflict",
        "expected": "DISPUTED",
        "pattern": "verdict_conflict",
        "mechanism": "same_run_incompatible_status",
        "query": "Was ETL job FIN-LOAD-2026-04-30 successful?",
        "sources": [
            {
                "title": "orchestrator.job_runs",
                "filters": "job_id=FIN-LOAD-2026-04-30",
                "columns": ["job_id", "run_date", "final_status"],
                "rows": [["FIN-LOAD-2026-04-30", "2026-04-30", "SUCCESS"]],
                "notes": "Scheduler final status.",
            },
            {
                "title": "data_quality.close_checks",
                "filters": "job_id=FIN-LOAD-2026-04-30",
                "columns": ["job_id", "run_date", "close_status", "blocking_error_count"],
                "rows": [["FIN-LOAD-2026-04-30", "2026-04-30", "FAILED", "3"]],
                "notes": "Close checks for the same run marked it failed.",
            },
        ],
    },
    {
        "id": "tab_08_scope_conflict",
        "expected": "DISPUTED",
        "pattern": "scope_conflict",
        "mechanism": "cohort_definition_conflict",
        "query": "How many active users did Product X report for April 2026?",
        "sources": [
            {
                "title": "product.analytics_monthly",
                "filters": "product=Product X; month=2026-04",
                "columns": ["product", "month", "active_users", "cohort_definition"],
                "rows": [["Product X", "2026-04", "812,000", "includes trial users"]],
                "notes": "Product analytics published active_users including trial users.",
            },
            {
                "title": "finance.operating_metrics",
                "filters": "product=Product X; month=2026-04",
                "columns": ["product", "month", "active_users", "cohort_definition"],
                "rows": [["Product X", "2026-04", "694,000", "paid users only"]],
                "notes": "Finance published active_users excluding trial users under the same metric name.",
            },
        ],
    },
    {
        "id": "tab_09_wrong_period",
        "expected": "ABSTAIN",
        "pattern": "temporal_mismatch",
        "mechanism": "wrong_partition",
        "query": "What was Acme Europe's net revenue in Q4 2025?",
        "sources": [
            {
                "title": "warehouse.finance_revenue_quarterly",
                "filters": "company=Acme; region=Europe; quarter=2025-Q3",
                "columns": ["company", "region", "quarter", "net_revenue_usd_millions"],
                "rows": [["Acme", "Europe", "2025-Q3", "39.7"]],
                "notes": "The retrieved row is for Q3 2025, not Q4 2025.",
            }
        ],
    },
    {
        "id": "tab_10_wrong_region",
        "expected": "ABSTAIN",
        "pattern": "wrong_specificity",
        "mechanism": "wrong_filter_value",
        "query": "How many active paid customers did Northwind have in Europe in March 2026?",
        "sources": [
            {
                "title": "mart.customer_monthly_status",
                "filters": "account=Northwind; month=2026-03; region=North America; customer_status=active; plan_type=paid",
                "columns": ["account", "region", "month", "active_paid_customers"],
                "rows": [["Northwind", "North America", "2026-03", "18,420"]],
                "notes": "The retrieved row is for North America, not Europe.",
            }
        ],
    },
    {
        "id": "tab_11_metric_mismatch",
        "expected": "ABSTAIN",
        "pattern": "wrong_specificity",
        "mechanism": "schema_metric_mismatch",
        "query": "What was Acme Europe's net revenue in Q4 2025?",
        "sources": [
            {
                "title": "warehouse.sales_bookings_quarterly",
                "filters": "company=Acme; region=Europe; quarter=2025-Q4",
                "columns": ["company", "region", "quarter", "gross_bookings_usd_millions"],
                "rows": [["Acme", "Europe", "2025-Q4", "57.3"]],
                "notes": "The table contains gross bookings only, not net revenue.",
            }
        ],
    },
    {
        "id": "tab_12_missing_result",
        "expected": "ABSTAIN",
        "pattern": "missing_execution_result",
        "mechanism": "query_without_result_grid",
        "query": "What was the average refund processing time for Enterprise customers in April 2026?",
        "sources": [
            {
                "title": "query_history.saved_sql",
                "filters": "segment=Enterprise; month=2026-04; metric=refund_processing_time",
                "columns": ["sql_text", "referenced_tables", "requested_metric"],
                "rows": [
                    [
                        "SELECT AVG(processing_hours) FROM refunds WHERE segment='Enterprise' AND month='2026-04'",
                        "refunds",
                        "average refund processing time",
                    ]
                ],
                "notes": "Only the saved SQL text and schema reference were retrieved; the result grid/value is not present.",
            }
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-source", type=Path, default=DEFAULT_THRESHOLD_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


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
    return 0.58


def markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def csv_table(columns: list[str], rows: list[list[str]]) -> str:
    return "\n".join([",".join(columns), *[",".join(str(v) for v in row) for row in rows]])


def packet_rows(columns: list[str], rows: list[list[str]]) -> str:
    rendered = []
    for idx, row in enumerate(rows, start=1):
        fields = "; ".join(f"{col}={value}" for col, value in zip(columns, row, strict=True))
        rendered.append(f"row_{idx}: {fields}")
    return "\n".join(rendered)


def serialize_source(source: dict[str, Any], style: str) -> str:
    if style == "markdown_table":
        return (
            f"Retrieved table: {source['title']}\n"
            f"Applied filters: {source['filters']}\n"
            f"{markdown_table(source['columns'], source['rows'])}\n"
            f"Note: {source['notes']}"
        )
    if style == "csv_extract":
        return (
            f"CSV extract from {source['title']}\n"
            f"# applied_filters: {source['filters']}\n"
            f"{csv_table(source['columns'], source['rows'])}\n"
            f"# note: {source['notes']}"
        )
    if style == "evidence_packet":
        return (
            "Structured evidence packet\n"
            f"table={source['title']}\n"
            f"filters={source['filters']}\n"
            f"columns={', '.join(source['columns'])}\n"
            f"{packet_rows(source['columns'], source['rows'])}\n"
            f"interpretation_note={source['notes']}"
        )
    raise ValueError(f"unknown serialization: {style}")


def build_cases() -> list[dict[str, Any]]:
    cases = []
    for scenario in SCENARIOS:
        for style in SERIALIZATIONS:
            cases.append(
                {
                    "id": f"{scenario['id']}__{style}",
                    "scenario_id": scenario["id"],
                    "serialization": style,
                    "expected": scenario["expected"],
                    "label_id": LABEL2ID[scenario["expected"]],
                    "taxonomy_pattern": scenario["pattern"],
                    "mechanism": scenario["mechanism"],
                    "query": scenario["query"],
                    "contexts": [serialize_source(source, style) for source in scenario["sources"]],
                    "meta": {
                        "modality": "structured",
                        "probe": "tabular_ood_v1",
                    },
                }
            )
    return cases


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def evaluate_cases(
    *,
    model_path: Path,
    threshold: float,
    device: str,
    cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    predictions = []
    labels = []
    raw_preds = []
    calibrated_preds = []

    with torch.no_grad():
        for case in cases:
            text = build_encoder_text(case["query"], case["contexts"])
            enc = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt").to(device)
            logits = model(**enc).logits[0].float().cpu().numpy()
            num_classes = int(logits.shape[-1])
            raw_id = int(gated_predictions(logits.reshape(1, -1), 0.0, num_classes=num_classes)[0])
            cal_id = int(
                gated_predictions(logits.reshape(1, -1), threshold, num_classes=num_classes)[0]
            )
            probs = softmax(logits)
            label_id = int(case["label_id"])
            labels.append(label_id)
            raw_preds.append(raw_id)
            calibrated_preds.append(cal_id)
            predictions.append(
                {
                    "id": case["id"],
                    "scenario_id": case["scenario_id"],
                    "serialization": case["serialization"],
                    "expected": case["expected"],
                    "raw_pred": ID2LABEL[raw_id],
                    "calibrated_pred": ID2LABEL[cal_id],
                    "ok": ID2LABEL[cal_id] == case["expected"],
                    "p_abstain": float(probs[0]),
                    "p_disputed": float(probs[1]),
                    "p_trustworthy": float(probs[2]),
                    "taxonomy_pattern": case["taxonomy_pattern"],
                    "mechanism": case["mechanism"],
                }
            )

    labels_arr = np.array(labels)
    raw_arr = np.array(raw_preds)
    cal_arr = np.array(calibrated_preds)
    summary = {
        "model": str(model_path),
        "device": device,
        "threshold": threshold,
        "rows": len(cases),
        "scenarios": len(SCENARIOS),
        "serializations": list(SERIALIZATIONS),
        "label_counts": dict(Counter(case["expected"] for case in cases)),
        "raw": compute_classification_metrics(raw_arr, labels_arr),
        "calibrated": compute_classification_metrics(cal_arr, labels_arr),
        "by_serialization": group_metrics(cases, predictions, "serialization"),
        "by_expected": group_metrics(cases, predictions, "expected"),
        "by_mechanism": group_metrics(cases, predictions, "mechanism"),
    }
    return summary, predictions


def group_metrics(cases: list[dict[str, Any]], preds: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, case in enumerate(cases):
        grouped[str(case[key])].append(idx)

    out = {}
    for value, indexes in sorted(grouped.items()):
        rows = [preds[i] for i in indexes]
        out[value] = {
            "n": len(rows),
            "correct": sum(int(row["ok"]) for row in rows),
            "accuracy": sum(int(row["ok"]) for row in rows) / len(rows),
            "false_trustworthy_rate": false_trustworthy(rows),
        }
    return out


def false_trustworthy(rows: list[dict[str, Any]]) -> float:
    non_t = [row for row in rows if row["expected"] != "TRUSTWORTHY"]
    if not non_t:
        return 0.0
    return sum(row["calibrated_pred"] == "TRUSTWORTHY" for row in non_t) / len(non_t)


def write_report(path: Path, summary: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    cal = summary["calibrated"]
    raw = summary["raw"]
    lines = [
        "# Tabular OOD Probe",
        "",
        f"- Model: `{summary['model']}`",
        f"- Threshold: `{summary['threshold']:.2f}`",
        f"- Rows: **{summary['rows']}** ({summary['scenarios']} scenarios x {len(summary['serializations'])} serializations)",
        f"- Calibrated accuracy: **{cal['accuracy']:.2%}**",
        f"- Calibrated false-TRUSTWORTHY: **{cal['false_trustworthy_rate']:.2%}**",
        f"- Raw accuracy: **{raw['accuracy']:.2%}**",
        f"- Raw false-TRUSTWORTHY: **{raw['false_trustworthy_rate']:.2%}**",
        "",
        "## By Serialization",
        "",
        "| Serialization | n | correct | accuracy | FT |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["by_serialization"].items():
        lines.append(
            f"| `{name}` | {row['n']} | {row['correct']} | {row['accuracy']:.2%} | {row['false_trustworthy_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## By Label",
            "",
            "| Expected | n | correct | accuracy | FT |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["by_expected"].items():
        lines.append(
            f"| `{name}` | {row['n']} | {row['correct']} | {row['accuracy']:.2%} | {row['false_trustworthy_rate']:.2%} |"
        )

    wrong = [row for row in predictions if not row["ok"]]
    lines.extend(
        [
            "",
            "## Errors",
            "",
            "| Case | Serialization | Expected | Predicted | P(A) | P(D) | P(T) |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    if wrong:
        for row in wrong:
            lines.append(
                f"| `{row['id']}` | `{row['serialization']}` | `{row['expected']}` | `{row['calibrated_pred']}` | "
                f"{row['p_abstain']:.3f} | {row['p_disputed']:.3f} | {row['p_trustworthy']:.3f} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    threshold = load_threshold(args)
    cases = build_cases()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "cases.jsonl", cases)

    summary, predictions = evaluate_cases(
        model_path=args.model,
        threshold=threshold,
        device=device,
        cases=cases,
    )
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", summary, predictions)

    cal = summary["calibrated"]
    print(f"Device       : {device}")
    print(f"Model        : {args.model}")
    print(f"Threshold    : {threshold:.2f}")
    print(f"Rows         : {summary['rows']}")
    print(f"Accuracy     : {cal['accuracy']:.2%}")
    print(f"False-T rate : {cal['false_trustworthy_rate']:.2%}")
    print(f"Wrote        : {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
