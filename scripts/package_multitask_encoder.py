"""Package and verify the pyrrho-nano-g3.1 multitask encoder.

This release shape is intentionally separate from the single-head encoder
release scripts. g3.1 is a custom multitask model with governance,
query-contract, route/domain, taxonomy, and scalar heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

from pyrrho.multitask import PyrrhoMultiTaskModernBert
from pyrrho.multitask_inference import PyrrhoMultiTaskPredictor


DEFAULT_MODEL_NAME = "pyrrho-nano-g3.1"
DEFAULT_SEED = 7
DEFAULT_SOURCE = Path("outputs/pyrrho-nano-g3_1_multitask/seed_7/best_model")
DEFAULT_OUTPUT = Path("models/pyrrho-nano-g3.1")
DEFAULT_SUMMARY = Path("outputs/pyrrho-nano-g3_1_multitask/summary.json")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_g3_1_multitask.yaml")
DEFAULT_DATA_DIR = Path("data/multitask_v8_1_query_contract")


SMOKE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "trustworthy_speed_of_light",
        "query": "What is the exact speed of light?",
        "contexts": ["NIST defines the speed of light as exactly 299,792,458 m/s."],
        "expected_governance": "TRUSTWORTHY",
    },
    {
        "id": "disputed_current_release",
        "query": "Which app release is the current production version?",
        "contexts": [
            "The deployment dashboard says version 4.2.1 is the current production release.",
            "The incident handoff says version 4.1.9 is the current production release.",
        ],
        "expected_governance": "DISPUTED",
    },
    {
        "id": "abstain_missing_result",
        "query": "Did the cache warmup job succeed?",
        "contexts": [
            "The job runbook describes how to start the cache warmup job, but it does not include a run result."
        ],
        "expected_governance": "ABSTAIN",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a local multitask encoder package")
    create.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    create.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    create.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    create.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    create.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    create.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    create.add_argument("--seed", type=int, default=DEFAULT_SEED)
    create.add_argument("--device", type=str, default="cpu")
    create.add_argument("--overwrite", action="store_true")

    verify = sub.add_parser("verify", help="Verify an existing local package")
    verify.add_argument("--package-dir", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--device", type=str, default="cpu")
    verify.add_argument("--skip-smoke", action="store_true")

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(package_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rel = path.relative_to(package_dir).as_posix()
        entries.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def reset_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
    resolved = output_dir.resolve()
    project_root = Path.cwd().resolve()
    if resolved == project_root or resolved == project_root.parent:
        raise ValueError(f"refusing to remove unsafe output path: {resolved}")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def seed_report(summary: dict[str, Any], seed: int) -> dict[str, Any]:
    reports = summary.get("seed_reports") or {}
    key = str(seed)
    if key not in reports:
        raise KeyError(f"seed {seed} not present in summary")
    return reports[key]


def release_readme(
    *,
    model_name: str,
    seed: int,
    threshold: float,
    report: dict[str, Any],
) -> str:
    test = report["test"]
    return f"""---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-classification
language:
  - en
base_model: answerdotai/ModernBERT-base
tags:
  - rag
  - governance
  - hallucination-detection
  - classification
  - fitz-gov
  - pyrrho
  - multitask
  - modernbert
datasets:
  - yafitzdev/fitz-gov
metrics:
  - accuracy
  - f1
---

# {model_name}

{model_name} is a small multitask RAG governance co-processor for anti-hallucination
and retrieval-quality pipelines. It reads a user question plus retrieved source
passages, then returns a calibrated evidence-state decision and auxiliary signals
that fitz-sage can use before answer generation.

It is not an answer generator and not an open-world fact checker. It sits between
retrieval and generation, or beside a retrieval package as a fast evidence
quality layer. Compared with `pyrrho-nano-g3`, this package adds multitask heads
for pre-retrieval query-contract classification, semantic route/domain, taxonomy
pattern, and six scalar governance signals.

## Governance Labels

| Label | Meaning |
|---|---|
| `ABSTAIN` | The retrieved sources do not contain enough evidence to answer the question. |
| `DISPUTED` | The retrieved sources conflict on the answer. |
| `TRUSTWORTHY` | The retrieved sources consistently support answering the question. |

## Multitask Heads

| Head | Labels / values | Intended use |
|---|---|---|
| `governance` | `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY` | Post-retrieval evidence sufficiency and conflict decision. |
| `query_contract` | `evidence_sufficiency`, `structured_lookup`, `temporal_grounding`, `exhaustive_coverage`, `comparison_coverage`, `representative_overview` | Pre-retrieval routing signal for what kind of evidence the query needs. |
| `route` | `science_medicine`, `law_policy`, `history_geography`, `technology_computing`, `economics_finance`, `culture_society`, `general_commonsense` | Semantic route/domain signal for retrieval policy and logging. |
| `taxonomy` | 23 fitz-gov taxonomy patterns | Failure/support pattern signal for audit and diagnostics. |
| `scalars` | `evidence_sufficiency`, `query_evidence_alignment`, `answer_coverage`, `conflict_density`, `retrieval_retry_value`, `false_trustworthy_risk` | Continuous governance signals for retry, ranking, and monitoring. |

## Outputs

This is a custom multitask package, not a standard single-head
`AutoModelForSequenceClassification` artifact. The recommended runtime is
`pyrrho.multitask_inference.PyrrhoMultiTaskPredictor` from the pyrrho repository.

The predictor returns a structured object:

| Field | Meaning |
|---|---|
| `governance.final_label` | Final calibrated label after the TRUSTWORTHY threshold rule. |
| `governance.raw_label` | Highest-probability governance label before threshold calibration. |
| `governance.probabilities` | Probability distribution over `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY`. |
| `governance.threshold` | TRUSTWORTHY probability threshold used by the package. |
| `query_contract.final_label` | Query-only contract prediction. |
| `route.final_label` | Query-only semantic route/domain prediction. |
| `taxonomy.final_label` | Query+evidence taxonomy-pattern prediction. |
| `scalars` | Six bounded scalar governance signals. |
| `timing_ms` | Local inference timing for the call. |

Example normalized output shape:

```json
{{
  "schema_version": "pyrrho_multitask_prediction_v1",
  "governance": {{
    "raw_label": "TRUSTWORTHY",
    "final_label": "TRUSTWORTHY",
    "used_threshold_fallback": false,
    "threshold": {threshold:.2f},
    "confidence": 0.84,
    "probabilities": {{
      "ABSTAIN": 0.08,
      "DISPUTED": 0.08,
      "TRUSTWORTHY": 0.84
    }}
  }},
  "query_contract": {{
    "final_label": "structured_lookup"
  }},
  "route": {{
    "final_label": "economics_finance"
  }},
  "taxonomy": {{
    "final_label": "direct_answer"
  }},
  "scalars": {{
    "evidence_sufficiency": 0.91,
    "query_evidence_alignment": 0.88,
    "answer_coverage": 0.86,
    "conflict_density": 0.08,
    "retrieval_retry_value": 0.12,
    "false_trustworthy_risk": 0.09
  }}
}}
```

The model does not generate answers, citations, source spans, retrieval results,
or natural-language explanations. It classifies and scores the `(query,
retrieved_contexts)` evidence state.

## Intended Use

Use this model when a RAG or retrieval package needs fast local signals about:

- whether retrieved evidence is enough to answer,
- whether retrieved evidence conflicts,
- what kind of evidence the query needs before retrieval,
- which semantic/domain route the query belongs to,
- which fitz-gov support/failure pattern is active,
- whether retrieval should retry, broaden, or escalate.

This model is not intended to write answers, verify facts outside the provided
sources, replace a retriever, or replace human review in high-stakes settings.

## Quick Start

Install the pyrrho package from the repository that contains this runtime, then
load the package with the multitask predictor:

```python
from huggingface_hub import snapshot_download

from pyrrho.multitask_inference import PyrrhoMultiTaskPredictor

MODEL_ID = "yafitzdev/{model_name}"
PACKAGE_DIR = snapshot_download(MODEL_ID)

query = "Which quarterly report is relevant?"
contexts = [
    "The Q2 report lists revenue, churn, and roadmap changes.",
]

predictor = PyrrhoMultiTaskPredictor.from_pretrained(PACKAGE_DIR, device="cpu")
result = predictor.predict(query, contexts)

print(result["governance"]["final_label"])
print(result["query_contract"]["final_label"])
print(result["route"]["final_label"])
print(result["taxonomy"]["final_label"])
print(result["scalars"])
```

For local package testing:

```powershell
python scripts/package_multitask_encoder.py verify --package-dir models/{model_name} --device cpu
```

## Release Selection

- Seed: `{seed}`
- TRUSTWORTHY threshold: `{threshold:.2f}`
- Selection reason: seed `{seed}` had the strongest composite release score while
  retaining strong governance, query-contract, route, taxonomy, and scalar metrics.

## Held-Out Test Metrics

| Metric | Result |
|---|---:|
| Governance accuracy | `{test["gov_accuracy"]:.4f}` |
| False-TRUSTWORTHY rate | `{test["gov_false_trustworthy_rate"]:.4f}` |
| Query-contract accuracy | `{test["query_contract_accuracy"]:.4f}` |
| Query-contract macro F1 | `{test["query_contract_macro_f1"]:.4f}` |
| Route accuracy | `{test["route_accuracy"]:.4f}` |
| Route macro F1 | `{test["route_macro_f1"]:.4f}` |
| Taxonomy accuracy | `{test["taxonomy_accuracy"]:.4f}` |
| Taxonomy macro F1 | `{test["taxonomy_macro_f1"]:.4f}` |
| Scalar MAE | `{test["scalar_mae"]:.4f}` |

Three-seed headline from the local release summary:

| Metric | Mean +/- std |
|---|---:|
| Governance accuracy | `97.84 +/- 0.15%` |
| False-TRUSTWORTHY rate | `0.85 +/- 0.07%` |
| Query-contract macro F1 | `94.24 +/- 0.28%` |
| Route accuracy | `93.41 +/- 0.32%` |
| Taxonomy accuracy | `89.26 +/- 0.23%` |
| Scalar MAE | `0.0592 +/- 0.0005` |

## Training Data

Trained on fitz-gov V8.1-style rows prepared from the V8.0.1 row set plus the
mandatory `routing.query_contract` field. The release package records the local
training config in `training_config.yaml` and detailed metrics in
`reports/summary.json`.

## Limitations

- This is a governance and routing co-processor, not a generator.
- The auxiliary heads are useful signals, not ground-truth explanations.
- Query-contract and route predictions are query-only and can be wrong when the
  user query is underspecified.
- Taxonomy and scalar outputs are trained on fitz-gov labels/signals and should
  be treated as decision-support metadata, not universal factual judgments.
- The license is CC BY-NC 4.0. Commercial use requires a separate license.
"""


def run_smoke(
    package_dir: Path,
    *,
    device: str,
    trustworthy_threshold: float | None = None,
) -> dict[str, Any]:
    predictor = PyrrhoMultiTaskPredictor.from_pretrained(
        package_dir,
        trustworthy_threshold=trustworthy_threshold,
        device=device,
    )
    rows = []
    ok = True
    for case in SMOKE_CASES:
        prediction = predictor.predict(case["query"], case["contexts"])
        governance = prediction["governance"]["final_label"]
        case_ok = governance == case["expected_governance"]
        ok = ok and case_ok
        rows.append(
            {
                "id": case["id"],
                "expected_governance": case["expected_governance"],
                "governance": prediction["governance"],
                "query_contract": prediction["query_contract"],
                "route": prediction["route"],
                "taxonomy": prediction["taxonomy"],
                "scalars": prediction["scalars"],
                "timing_ms": prediction["timing_ms"],
                "ok": case_ok,
            }
        )
    return {
        "schema_version": "pyrrho_multitask_package_smoke_v1",
        "package_dir": str(package_dir.resolve()),
        "device": device,
        "ok": ok,
        "cases": rows,
    }


def create_package(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    summary = read_json(args.summary)
    report = seed_report(summary, args.seed)
    threshold = float(report["eval"]["threshold"])

    reset_output_dir(output_dir, overwrite=bool(args.overwrite))
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = PyrrhoMultiTaskModernBert.from_pretrained(source_dir, map_location="cpu")
    model.save_pretrained(output_dir)
    backbone_config_path = output_dir / "config.json"
    backbone_config = read_json(backbone_config_path)
    backbone_config["architectures"] = ["PyrrhoMultiTaskModernBert"]
    backbone_config["pyrrho_package_type"] = "multitask_encoder"
    backbone_config["pyrrho_model_name"] = args.model_name
    write_json(backbone_config_path, backbone_config)
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    tokenizer.save_pretrained(output_dir)

    final_metrics_src = source_dir.parent / "final_metrics.json"
    shutil.copy2(args.summary, reports_dir / "summary.json")
    shutil.copy2(final_metrics_src, reports_dir / f"final_metrics_seed_{args.seed}.json")
    shutil.copy2(args.config, output_dir / "training_config.yaml")

    readme = release_readme(
        model_name=args.model_name,
        seed=args.seed,
        threshold=threshold,
        report=report,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    smoke = run_smoke(output_dir, device=args.device, trustworthy_threshold=threshold)
    write_json(reports_dir / "package_smoke.json", smoke)

    manifest = {
        "schema_version": "pyrrho_multitask_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "architecture": "PyrrhoMultiTaskModernBert",
        "source_checkpoint": str(source_dir),
        "training_config": "training_config.yaml",
        "data_dir": str(args.data_dir.resolve()),
        "release": {
            "seed": int(args.seed),
            "trustworthy_threshold": threshold,
            "selection_reason": (
                "Seed 7 had the strongest composite and governance accuracy while "
                "retaining strong query-contract, route, taxonomy, and scalar metrics."
            ),
        },
        "heads": {
            "governance_id2label": model.pyrrho_config.id2label,
            "query_contract_id2label": model.pyrrho_config.query_contract_id2label,
            "route_id2label": model.pyrrho_config.route_id2label,
            "taxonomy_id2label": model.pyrrho_config.taxonomy_id2label,
            "scalar_fields": list(model.pyrrho_config.scalar_fields),
        },
        "metrics": report,
        "reports": {
            "summary": "reports/summary.json",
            "seed_metrics": f"reports/final_metrics_seed_{args.seed}.json",
            "package_smoke": "reports/package_smoke.json",
        },
        "files": package_files(output_dir),
    }
    write_json(output_dir / "manifest.json", manifest)
    return verify_package_dir(output_dir, device=args.device, skip_smoke=False)


def verify_hashes(package_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for entry in manifest.get("files", []):
        path = package_dir / entry["path"]
        exists = path.exists()
        actual_sha = sha256_file(path) if exists else None
        actual_bytes = path.stat().st_size if exists else None
        checks.append(
            {
                "path": entry["path"],
                "exists": exists,
                "bytes_ok": exists and actual_bytes == int(entry["bytes"]),
                "sha256_ok": exists and actual_sha == entry["sha256"],
            }
        )
    return checks


def verify_package_dir(package_dir: Path, *, device: str, skip_smoke: bool) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest = read_json(package_dir / "manifest.json")
    required = [
        "config.json",
        "model.safetensors",
        "pyrrho_multitask_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
        "manifest.json",
    ]
    required_checks = [{"path": name, "exists": (package_dir / name).exists()} for name in required]
    hash_checks = verify_hashes(package_dir, manifest)
    smoke = None if skip_smoke else run_smoke(package_dir, device=device)
    ok = (
        all(check["exists"] for check in required_checks)
        and all(check["bytes_ok"] and check["sha256_ok"] for check in hash_checks)
        and (skip_smoke or bool(smoke and smoke["ok"]))
    )
    report = {
        "schema_version": "pyrrho_multitask_package_verify_v1",
        "package_dir": str(package_dir),
        "ok": ok,
        "device": device,
        "required_checks": required_checks,
        "hash_checks": hash_checks,
        "smoke": smoke,
    }
    write_json(package_dir / "release_verify_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    if args.command == "create":
        report = create_package(args)
        print(
            "ok={ok} package={package}".format(
                ok=report["ok"],
                package=report["package_dir"],
            )
        )
        return 0 if report["ok"] else 1

    report = verify_package_dir(
        args.package_dir,
        device=args.device,
        skip_smoke=bool(args.skip_smoke),
    )
    print(
        "ok={ok} package={package}".format(
            ok=report["ok"],
            package=report["package_dir"],
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
