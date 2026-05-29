"""code_ood_probe.py - hand-authored code-evidence OOD probe.

Builds a small, fixed code-evidence test set, serializes each scenario in
three retrieval styles, and scores a pyrrho encoder checkpoint.

Run from project root:
    python scripts/code_ood_probe.py
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

SERIALIZATIONS = ("code_excerpt", "review_packet", "diff_context")
DEFAULT_MODEL = Path("models/pyrrho-nano-g3")
DEFAULT_THRESHOLD_SOURCE = Path("outputs/multi_seed_g3_v8/seed_1337/final_metrics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/code_ood_probe/g3_release")


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "code_01_expired_jwt",
        "expected": "TRUSTWORTHY",
        "pattern": "direct_answer",
        "mechanism": "exact_symbol_support",
        "query": "Does `validate_token` reject expired JWTs?",
        "sources": [
            {
                "path": "services/auth/tokens.py",
                "language": "python",
                "note": "The function raises before returning when exp is not in the future.",
                "content": """
def validate_token(payload: dict, now: datetime) -> str:
    exp = payload.get("exp")
    if exp is None:
        raise TokenError("missing exp")
    if int(exp) <= int(now.timestamp()):
        raise TokenError("expired token")
    return str(payload["sub"])
""",
            }
        ],
    },
    {
        "id": "code_02_retry_limit",
        "expected": "TRUSTWORTHY",
        "pattern": "direct_answer",
        "mechanism": "constant_flow_support",
        "query": "What retry limit is used by `sync_customer`?",
        "sources": [
            {
                "path": "jobs/customer_sync.py",
                "language": "python",
                "note": "The loop can run four attempts because the range includes 1..4.",
                "content": """
MAX_SYNC_RETRIES = 4

def sync_customer(customer_id: str) -> None:
    for attempt in range(1, MAX_SYNC_RETRIES + 1):
        try:
            push_customer(customer_id)
            return
        except TemporarySyncError:
            if attempt == MAX_SYNC_RETRIES:
                raise
""",
            }
        ],
    },
    {
        "id": "code_03_idempotency",
        "expected": "TRUSTWORTHY",
        "pattern": "consistent_chain",
        "mechanism": "control_flow_support",
        "query": "Does `create_charge` persist an idempotency key before calling the gateway?",
        "sources": [
            {
                "path": "billing/charges.py",
                "language": "python",
                "note": "The pending request row is created before gateway.charge is called.",
                "content": """
def create_charge(account_id: str, key: str, cents: int) -> str:
    existing = ChargeRequest.get_by_idempotency_key(key)
    if existing:
        return existing.charge_id
    request = ChargeRequest.create(
        account_id=account_id,
        idempotency_key=key,
        status="pending",
    )
    request.charge_id = gateway.charge(account_id=account_id, cents=cents)
    request.save()
    return request.charge_id
""",
            }
        ],
    },
    {
        "id": "code_04_admin_guard",
        "expected": "TRUSTWORTHY",
        "pattern": "direct_answer",
        "mechanism": "decorator_support",
        "query": "Is the `/admin/rebuild-index` route restricted to admin users?",
        "sources": [
            {
                "path": "web/admin_routes.py",
                "language": "python",
                "note": "The route has the admin role guard directly above the handler.",
                "content": """
@router.post("/admin/rebuild-index")
@require_role("admin")
def rebuild_index(request: Request) -> Response:
    enqueue_job("search.rebuild_index", requested_by=request.user.id)
    return Response(status_code=202)
""",
            }
        ],
    },
    {
        "id": "code_05_retry_conflict",
        "expected": "DISPUTED",
        "pattern": "numerical_conflict",
        "mechanism": "constant_config_conflict",
        "query": "What retry limit is configured for `sync_customer`?",
        "sources": [
            {
                "path": "jobs/customer_sync.py",
                "language": "python",
                "note": "Application code sets the retry limit to four.",
                "content": """
MAX_SYNC_RETRIES = 4

def sync_customer(customer_id: str) -> None:
    for attempt in range(1, MAX_SYNC_RETRIES + 1):
        push_customer(customer_id)
""",
            },
            {
                "path": "deploy/prod/customer-sync.yaml",
                "language": "yaml",
                "note": "The production deployment overrides the same retry setting to two.",
                "content": """
env:
  MAX_SYNC_RETRIES: "2"
  SYNC_QUEUE: "customer-sync-prod"
""",
            },
        ],
    },
    {
        "id": "code_06_public_conflict",
        "expected": "DISPUTED",
        "pattern": "authority_status_conflict",
        "mechanism": "security_metadata_conflict",
        "query": "Is `/reports/export` a public endpoint?",
        "sources": [
            {
                "path": "web/report_routes.py",
                "language": "python",
                "note": "The code requires an authenticated user.",
                "content": """
@router.get("/reports/export")
@login_required
def export_reports(request: Request) -> FileResponse:
    return build_report_export(request.user.account_id)
""",
            },
            {
                "path": "openapi/reports.yaml",
                "language": "yaml",
                "note": "The OpenAPI file marks the same path as public with no security scheme.",
                "content": """
/reports/export:
  get:
    x-public: true
    security: []
""",
            },
        ],
    },
    {
        "id": "code_07_hash_conflict",
        "expected": "DISPUTED",
        "pattern": "factual_contradiction",
        "mechanism": "implementation_doc_conflict",
        "query": "Which algorithm hashes password reset tokens?",
        "sources": [
            {
                "path": "security/reset_tokens.py",
                "language": "python",
                "note": "The implementation hashes reset tokens with SHA-256.",
                "content": """
def digest_reset_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
""",
            },
            {
                "path": "docs/security.md",
                "language": "markdown",
                "note": "The security doc names a different algorithm for reset tokens.",
                "content": """
Password reset tokens are never stored in plaintext. The reset-token digest is
computed with SHA-512 before the value is written to the database.
""",
            },
        ],
    },
    {
        "id": "code_08_flag_conflict",
        "expected": "DISPUTED",
        "pattern": "verdict_conflict",
        "mechanism": "default_value_conflict",
        "query": "Is the beta checkout flow enabled by default?",
        "sources": [
            {
                "path": "checkout/flags.py",
                "language": "python",
                "note": "The application default is disabled.",
                "content": """
FLAGS = {
    "beta_checkout": FeatureFlag(default=False, owner="checkout"),
}
""",
            },
            {
                "path": "config/rollout-defaults.json",
                "language": "json",
                "note": "The rollout default enables the same flag.",
                "content": """
{
  "beta_checkout": {
    "default": true,
    "rollout": "all_accounts"
  }
}
""",
            },
        ],
    },
    {
        "id": "code_09_wrong_symbol",
        "expected": "ABSTAIN",
        "pattern": "wrong_entity",
        "mechanism": "wrong_symbol",
        "query": "Does `send_invoice_email` attach the invoice PDF?",
        "sources": [
            {
                "path": "notifications/email.py",
                "language": "python",
                "note": "The retrieved function is for welcome email, not invoice email.",
                "content": """
def send_welcome_email(user: User) -> None:
    msg = EmailMessage(to=user.email, template="welcome")
    msg.attach("welcome_guide.pdf", build_welcome_pdf(user))
    mailer.send(msg)
""",
            }
        ],
    },
    {
        "id": "code_10_missing_audit",
        "expected": "ABSTAIN",
        "pattern": "evidence_absent",
        "mechanism": "missing_specific_field",
        "query": "Which audit event name does `refund_order` write?",
        "sources": [
            {
                "path": "orders/refunds.py",
                "language": "python",
                "note": "The excerpt shows the refund path but no audit event name.",
                "content": """
def refund_order(order_id: str, cents: int) -> Refund:
    order = Order.get(order_id)
    gateway.refund(payment_id=order.payment_id, amount_cents=cents)
    return Refund.create(order_id=order_id, amount_cents=cents)
""",
            }
        ],
    },
    {
        "id": "code_11_version_mismatch",
        "expected": "ABSTAIN",
        "pattern": "version_build_mismatch",
        "mechanism": "wrong_api_version",
        "query": "In API v2, does `parse_amount` reject negative values?",
        "sources": [
            {
                "path": "api/v1/payments.py",
                "language": "python",
                "note": "The retrieved implementation is explicitly for API v1, not v2.",
                "content": """
def parse_amount(raw: str) -> int:
    cents = int(Decimal(raw) * 100)
    if cents < 0:
        raise ValueError("amount must be positive")
    return cents
""",
            }
        ],
    },
    {
        "id": "code_12_missing_result",
        "expected": "ABSTAIN",
        "pattern": "missing_execution_result",
        "mechanism": "test_definition_without_run",
        "query": "Did the migration test pass for `202605_add_invoice_index`?",
        "sources": [
            {
                "path": "tests/migrations/test_invoice_index.py",
                "language": "python",
                "note": "Only the test definition is retrieved; no run status is present.",
                "content": """
def test_202605_add_invoice_index(migrator):
    migrator.apply("202605_add_invoice_index")
    indexes = migrator.inspect_indexes("invoice")
    assert "idx_invoice_account_created_at" in indexes
""",
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


def cleaned_code(text: str) -> str:
    return text.strip("\n")


def numbered_lines(text: str) -> str:
    lines = cleaned_code(text).splitlines()
    return "\n".join(f"{idx:03d}: {line}" for idx, line in enumerate(lines, start=1))


def diff_lines(text: str) -> str:
    lines = cleaned_code(text).splitlines()
    return "\n".join(f"+ {line}" if line.strip() else "+" for line in lines)


def serialize_source(source: dict[str, Any], style: str) -> str:
    content = cleaned_code(source["content"])
    if style == "code_excerpt":
        return (
            f"Retrieved file: {source['path']}\n"
            f"Language: {source['language']}\n"
            f"```{source['language']}\n{content}\n```\n"
            f"Note: {source['note']}"
        )
    if style == "review_packet":
        return (
            "Code review evidence packet\n"
            f"path={source['path']}\n"
            f"language={source['language']}\n"
            f"note={source['note']}\n"
            "numbered_excerpt:\n"
            f"{numbered_lines(content)}"
        )
    if style == "diff_context":
        return (
            "Retrieved diff context\n"
            f"+++ b/{source['path']}\n"
            "@@ relevant excerpt @@\n"
            f"{diff_lines(content)}\n"
            f"review_note={source['note']}"
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
                    "contexts": [
                        serialize_source(source, style) for source in scenario["sources"]
                    ],
                    "meta": {"modality": "code", "probe": "code_ood_v1"},
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
        "# Code OOD Probe",
        "",
        f"- Model: `{summary['model']}`",
        f"- Threshold: `{summary['threshold']:.2f}`",
        f"- Rows: **{summary['rows']}** "
        f"({summary['scenarios']} scenarios x {len(summary['serializations'])} serializations)",
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
            f"| `{name}` | {row['n']} | {row['correct']} | "
            f"{row['accuracy']:.2%} | {row['false_trustworthy_rate']:.2%} |"
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
            f"| `{name}` | {row['n']} | {row['correct']} | "
            f"{row['accuracy']:.2%} | {row['false_trustworthy_rate']:.2%} |"
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
                f"| `{row['id']}` | `{row['serialization']}` | `{row['expected']}` | "
                f"`{row['calibrated_pred']}` | {row['p_abstain']:.3f} | "
                f"{row['p_disputed']:.3f} | {row['p_trustworthy']:.3f} |"
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
