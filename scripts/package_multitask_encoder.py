"""Package and verify pyrrho multitask encoder releases.

This release shape is intentionally separate from the single-head encoder
release scripts. The multitask encoder is a custom model with governance,
query-contract, route/domain, taxonomy, scalar, and optional retrieval-control
heads.
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
    create.add_argument("--selection-reason", type=str, default=None)

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
    if isinstance(reports, dict):
        if key not in reports:
            raise KeyError(f"seed {seed} not present in summary")
        return normalize_report_for_packaging(reports[key])
    if isinstance(reports, list):
        for entry in reports:
            if int(entry.get("seed", -1)) != int(seed):
                continue
            metrics_path = entry.get("path")
            if metrics_path and Path(metrics_path).exists():
                return normalize_report_for_packaging(read_json(Path(metrics_path)))
            if "test_key_metrics" in entry:
                threshold = float(entry["test_key_metrics"].get("threshold", 0.5))
                return {
                    "best_epoch": entry.get("best_epoch"),
                    "eval": {"threshold": threshold},
                    "test": dict(entry["test_key_metrics"]),
                }
        raise KeyError(f"seed {seed} not present in summary")
    raise TypeError("summary seed_reports must be a dict or list")


def flatten_split_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize nested trainer metrics to the older flat package format."""
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(value, dict):
            flat[key] = value

    governance = metrics.get("governance_calibrated")
    if isinstance(governance, dict):
        for key, value in governance.items():
            if key == "threshold":
                flat["threshold"] = value
                flat["gov_threshold"] = value
            elif key == "target_met":
                flat["gov_target_met"] = value
            else:
                flat[f"gov_{key}"] = value

    for group in (
        "query_contract",
        "route",
        "taxonomy",
        "scalars",
        "retrieval_action",
        "gap_type",
        "answerability_shape",
        "retrieval_modality",
        "retrieval_obligation",
    ):
        group_metrics = metrics.get(group)
        if isinstance(group_metrics, dict):
            flat.update(group_metrics)
    return flat


def normalize_report_for_packaging(report: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(report)
    for split in ("eval", "test"):
        split_metrics = normalized.get(split)
        if isinstance(split_metrics, dict):
            normalized[split] = flatten_split_metrics(split_metrics)
    return normalized


def default_selection_reason(model_name: str, seed: int) -> str:
    if model_name.endswith("g4"):
        return (
            f"Seed {seed} was selected from the completed official fitz-gov "
            "V9.0.0 3-seed run because it has the strongest validation "
            "composite score and the best balanced auxiliary-head tradeoff."
        )
    if model_name.endswith("g3.2"):
        return (
            f"Seed {seed} was selected after reviewing the completed 3-seed V8.2 "
            "run; it has the strongest held-out governance accuracy / "
            "false-trustworthy tradeoff and the strongest retrieval-modality "
            "held-out macro F1 among the candidate seeds."
        )
    return (
        f"Seed {seed} was selected from the completed local 3-seed summary for "
        "the strongest overall release tradeoff across governance and auxiliary heads."
    )


def has_retrieval_control(report: dict[str, Any]) -> bool:
    test = report.get("test", {})
    return any(
        key in test
        for key in (
            "retrieval_action_macro_f1",
            "gap_type_macro_f1",
            "answerability_shape_macro_f1",
            "retrieval_modality_macro_f1",
            "retrieval_obligation_macro_f1",
        )
    )


def metric_row(metrics: dict[str, Any], label: str, key: str) -> str:
    if key not in metrics:
        return ""
    return f"| {label} | `{float(metrics[key]):.4f}` |"


def aggregate_row(summary: dict[str, Any], label: str, key: str, *, percent: bool) -> str:
    aggregate = summary.get("aggregate", {}).get("test", {})
    aggregate_key = key
    if aggregate_key not in aggregate:
        aliases = {
            "gov_accuracy": ("governance_calibrated.accuracy", "governance_accuracy"),
            "gov_false_trustworthy_rate": (
                "governance_calibrated.false_trustworthy_rate",
                "false_trustworthy_rate",
            ),
            "query_contract_macro_f1": ("query_contract.query_contract_macro_f1",),
            "route_accuracy": ("route.route_accuracy",),
            "taxonomy_accuracy": ("taxonomy.taxonomy_accuracy",),
            "scalar_mae": ("scalars.scalar_mae",),
            "retrieval_action_macro_f1": ("retrieval_action.retrieval_action_macro_f1",),
            "gap_type_macro_f1": ("gap_type.gap_type_macro_f1",),
            "answerability_shape_macro_f1": (
                "answerability_shape.answerability_shape_macro_f1",
            ),
            "retrieval_modality_macro_f1": (
                "retrieval_modality.retrieval_modality_macro_f1",
            ),
            "retrieval_obligation_macro_f1": (
                "retrieval_obligation.retrieval_obligation_macro_f1",
            ),
        }
        for candidate in aliases.get(key, ()):
            if candidate in aggregate:
                aggregate_key = candidate
                break
    if aggregate_key not in aggregate:
        return ""
    row = aggregate[aggregate_key]
    mean = float(row["mean"])
    std = float(row["std"])
    if percent:
        return f"| {label} | `{mean * 100:.2f} +/- {std * 100:.2f}%` |"
    return f"| {label} | `{mean:.4f} +/- {std:.4f}` |"


def compact_table(rows: list[str]) -> str:
    return "\n".join(row for row in rows if row)


def release_readme(
    *,
    model_name: str,
    seed: int,
    threshold: float,
    report: dict[str, Any],
    summary: dict[str, Any],
    scalar_fields: list[str],
    selection_reason: str,
) -> str:
    test = report["test"]
    retrieval_control = has_retrieval_control(report)
    has_retrieval_obligation = "retrieval_obligation_macro_f1" in test
    is_g4_alpha = "g4-alpha" in model_name
    is_g4_target100 = "g4-target100" in model_name or "g4-v9-target100" in model_name
    is_g4_official = (
        model_name == "pyrrho-nano-g4"
        or "g4-v9-official" in model_name
        or "g4_v9_official" in model_name
    )
    is_g5_alpha = "g5-alpha" in model_name
    is_g5_repair = model_name == "pyrrho-nano-g5.1" or "g5_1" in model_name
    is_g5_5 = model_name == "pyrrho-nano-g5.5" or "g5_5" in model_name
    is_g5_6 = (
        model_name == "pyrrho-nano-g5.6"
        or "g5.6" in model_name
        or "g5_6" in model_name
    )
    is_g5_official = model_name == "pyrrho-nano-g5" or "g5_v10_target10" in model_name
    if is_g5_official or is_g5_repair or is_g5_5 or is_g5_6:
        if is_g5_6:
            comparison = (
                "Compared with `pyrrho-nano-g5.5`, this local candidate starts "
                "from the g5.5 checkpoint and adds 7,061 V12 focused-medium "
                "retrieval-planning rows targeted at the remaining strict-owner "
                "fitz-sage gaps."
            )
        elif is_g5_5:
            comparison = (
                "Compared with `pyrrho-nano-g5`, this package trains the same "
                "multitask surface on the official fitz-gov V11.0.0 repair "
                "release, adding targeted strict-owner retrieval-planning rows "
                "for class obligations, failure-focused cases, and larger "
                "retrieval evidence packs."
            )
        elif is_g5_repair:
            comparison = (
                "Compared with `pyrrho-nano-g5`, this local repair rebuilds "
                "the same V10 target-10 training rows after fixing retrieval-"
                "action alias normalization so TRUSTWORTHY rows with no "
                "evidence gap train as `answer_now`, not `retrieve_more`."
            )
        else:
            comparison = (
                "Compared with `pyrrho-nano-g5-alpha`, this package expands the "
                "V10 retrieval-obligation training block from the initial 5/5/5 "
                "candidate set to the full target-10 candidate set and uses a "
                "completed 3-seed run."
            )
        if is_g5_6:
            data_description = (
                "Trained from the local `pyrrho-nano-g5.5` checkpoint on the "
                "official fitz-gov V11.0.0 row set plus 7,061 local V12/g5.6 "
                "focused-medium candidate rows. Total prepared rows: 67,944 = "
                "2,980 V6 rows + 7,520 V7 rows + 14,092 V8 rows + 16,163 V9 "
                "rows + 12,748 V10 rows + 7,380 V11 rows + 7,061 V12 candidate "
                "rows. Prepared splits are train=54,416 / validation=6,778 / "
                "test=6,750. The V12 candidate block is structurally validated, "
                "duplicate-clean, and gap-closed against the saved focused-medium "
                "plan, but it has not received blind-label QA."
            )
            limitation_extra = (
                "- This is a local candidate, not a publishable official release.\n"
                "- The 7,061 V12/g5.6 candidate rows are structurally clean and "
                "gap-targeted, but not blind-label QAed.\n"
                "- This package exists to test whether the focused-medium data "
                "moves fitz-sage strict-owner behavior before committing to a "
                "larger V12 expansion.\n"
                "- The retrieval-obligation head is trained only on rows with a "
                "concrete retrieval obligation; rows with `retrieval_obligation=none` "
                "are masked for that head.\n"
            )
        elif is_g5_5:
            data_description = (
                "Trained on the published fitz-gov V11.0.0 Hugging Face release "
                "with official query-grouped splits. Total prepared rows: 60,883 "
                "= 2,980 V6 rows + 7,520 V7 rows + 14,092 V8 rows + 16,163 V9 "
                "rows + 12,748 V10 rows + 7,380 V11 rows. Splits are "
                "train=48,800 / validation=6,028 / test=6,055. Split assignments "
                "come from `v11/split_assignments.jsonl` at dataset commit "
                "`580809e42376d84284043689c702de4c500bca85`."
            )
            limitation_extra = (
                "- The retrieval-obligation head is trained only on rows with a "
                "concrete retrieval obligation; rows with `retrieval_obligation=none` "
                "are masked for that head.\n"
                "- Retrieval obligation and retrieval modality are planning heads. "
                "Low-confidence fine-grained obligations should be treated as "
                "retrieval hints, not hard guarantees.\n"
                "- This package is trained against the official V11 benchmark "
                "contract; fitz-sage integration still needs a separate strict-owner "
                "benchmark run before declaring a production upgrade.\n"
            )
        else:
            data_description = (
                "Trained on published fitz-gov V9.0.0 plus 12,748 local V10 "
                "target-10 retrieval-planning candidates. The V10 block combines "
                "6,058 initial 5/5/5 rows with 6,690 target-10 continuation rows; "
                "after repair, blind-label QA scored 12,748/12,748 agreement with "
                "0 triage, 0 missing, 0 invalid, and 0 error rows. Prepared splits "
                "are train=42,826 / validation=5,372 / test=5,305. Treat this as a "
                "local release candidate until the repaired V10 block is published "
                "as an official fitz-gov release."
            )
            limitation_extra = (
                "- This package depends on local V10 candidate rows that are QA-clean "
                "but not yet published as an official fitz-gov release.\n"
                "- The retrieval-obligation head is trained only on the V10 candidate "
                "rows; official V9 rows are masked for that head.\n"
                + (
                    "- This is a local repair checkpoint, not a published release. "
                    "fitz-sage benchmark pass rate improved versus g5, but mixed "
                    "table/prose/code retrieval remains unresolved.\n"
                    if is_g5_repair
                    else ""
                )
                + "- Retrieval obligation is the main remaining weak head. Low-support "
                "or fine-grained obligations such as row-key lookup, column-value "
                "lookup, stale-row versioning, and mixed-modality obligations should "
                "be treated as planning hints, not hard guarantees.\n"
            )
        answerability_head_row = (
            "| `answerability_shape` | `direct_answer`, `synthesis_answer`, "
            "`set_answer`, `structured_reasoning` | Query-only collapsed answer "
            "shape for retrieval planning. |"
        )
    elif is_g5_alpha:
        comparison = (
            "Compared with `pyrrho-nano-g4`, this alpha package keeps the "
            "official V9 governance/retrieval-control surface and adds a "
            "query-only V10 `retrieval_obligation` head for corpus-aware "
            "evidence-contract planning."
        )
        data_description = (
            "Trained on the published fitz-gov V9.0.0 Hugging Face release plus "
            "6,058 local V10 5/5/5 retrieval-planning candidates from "
            "`data/_workspaces/handoff/v10_cogeneration_5x5_20260612`. The V10 "
            "candidate block passed post-repair structural validation and quality "
            "audit, but has not received independent blind-label QA."
        )
        limitation_extra = (
            "- This is an alpha integration checkpoint, not a public release-quality "
            "benchmark replacement for `pyrrho-nano-g4`.\n"
            "- The V10 candidate block is structurally clean and targeted, but not "
            "blind-label QAed. Treat metrics as development signals.\n"
            "- The retrieval-obligation head is trained only on the V10 candidate "
            "rows; official V9 rows are masked for that head.\n"
        )
        answerability_head_row = (
            "| `answerability_shape` | `direct_answer`, `synthesis_answer`, "
            "`set_answer`, `structured_reasoning` | Query-only collapsed answer "
            "shape for retrieval planning. |"
        )
    elif is_g4_alpha:
        comparison = (
            "Compared with `pyrrho-nano-g3.2`, this alpha package keeps the "
            "retrieval-control head surface but collapses answerability into the "
            "four V9 planning labels: `direct_answer`, `synthesis_answer`, "
            "`set_answer`, and `structured_reasoning`."
        )
        data_description = (
            "Trained on a local alpha dataset made from published fitz-gov V8.2.0 "
            "plus local V9 rows: 24,592 V8.2 rows, 750 locally merged V9 rows, "
            "and 10,020 structurally clean V9 candidate rows from batches 100-433. "
            "The 10,020 candidate rows have not received blind-label QA."
        )
        limitation_extra = (
            "- This is an alpha integration checkpoint, not a public release-quality "
            "benchmark or replacement for `pyrrho-nano-g3.2`.\n"
            "- The four-way answerability head is intentionally collapsed for "
            "fitz-sage integration; it does not expose the old eleven detailed "
            "answerability labels.\n"
            "- The V9 candidate block was structurally normalized and label-trusted, "
            "but not blind-label QAed. Treat metrics as development signals.\n"
            "- Retrieval modality remains weak for sparse subclasses such as "
            "`pdf_layout`, `code`, and `configuration`.\n"
        )
        answerability_head_row = (
            "| `answerability_shape` | `direct_answer`, `synthesis_answer`, "
            "`set_answer`, `structured_reasoning` | Query-only collapsed answer "
            "shape for retrieval planning. |"
        )
    elif is_g4_official:
        comparison = (
            "Compared with `pyrrho-nano-g3.2`, this package keeps the "
            "retrieval-control head surface but collapses answerability into the "
            "four V9 planning labels: `direct_answer`, `synthesis_answer`, "
            "`set_answer`, and `structured_reasoning`."
        )
        data_description = (
            "Trained on the published fitz-gov V9.0.0 Hugging Face release with "
            "official query-grouped splits. Total prepared rows: 40,755 = 2,980 "
            "V6 rows + 7,520 V7 rows + 14,092 V8 rows + 16,163 V9 rows. Splits "
            "are train=32,625 / validation=4,104 / test=4,026. Split assignments "
            "come from `v9/split_assignments.jsonl` at dataset commit "
            "`874fd18d4952eec0e72b6df2264f8281615fd350`."
        )
        limitation_extra = (
            "- The four-way answerability head is intentionally collapsed for "
            "fitz-sage integration; it does not expose the old eleven detailed "
            "answerability labels.\n"
            "- Retrieval modality remains the weakest auxiliary head; sparse "
            "subclasses such as `pdf_layout`, `code`, and `configuration` should "
            "be treated as hints, not hard guarantees.\n"
        )
        answerability_head_row = (
            "| `answerability_shape` | `direct_answer`, `synthesis_answer`, "
            "`set_answer`, `structured_reasoning` | Query-only collapsed answer "
            "shape for retrieval planning. |"
        )
    elif is_g4_target100:
        comparison = (
            "Compared with `pyrrho-nano-g3.2`, this local candidate keeps the "
            "retrieval-control head surface but collapses answerability into the "
            "four V9 planning labels: `direct_answer`, `synthesis_answer`, "
            "`set_answer`, and `structured_reasoning`."
        )
        data_description = (
            "Trained on `data/multitask_g4_v9_target100`, built from published "
            "fitz-gov V8.2.0 plus the QA-gated local V9 target-100 vault. Total "
            "prepared rows: 40,755 = 24,592 published V8.2 rows + 16,163 local "
            "V9 rows. Splits are train=32,617 / eval=4,051 / test=4,087. The "
            "local V9 expansion closed all 189 answerability target cells at "
            "100/cell before training prep."
        )
        limitation_extra = (
            "- This is a pre-public-split local candidate, superseded by the "
            "official fitz-gov V9.0.0 `pyrrho-nano-g4` run; do not treat it as "
            "a final public release.\n"
            "- The four-way answerability head is intentionally collapsed for "
            "fitz-sage integration; it does not expose the old eleven detailed "
            "answerability labels.\n"
            "- The local V9 rows were QA-gated, but this package uses the "
            "pre-public local split rather than the official Hub split.\n"
            "- Retrieval modality remains weak for sparse subclasses such as "
            "`pdf_layout`, `code`, and `configuration`.\n"
        )
        answerability_head_row = (
            "| `answerability_shape` | `direct_answer`, `synthesis_answer`, "
            "`set_answer`, `structured_reasoning` | Query-only collapsed answer "
            "shape for retrieval planning. |"
        )
    elif retrieval_control:
        comparison = (
            "Compared with `pyrrho-nano-g3.1`, this package adds V8.2 "
            "retrieval-control heads for retrieval action, gap type, "
            "answerability shape, retrieval modality, and an additional "
            "`evidence_failure_severity` scalar."
        )
        data_description = (
            "Trained on fitz-gov V8.2.0 rows with mandatory "
            "`routing.query_contract` and `routing.retrieval_control` fields."
        )
        limitation_extra = (
            "- Retrieval modality is the weakest new head; sparse subclasses such as "
            "`pdf_layout` should be treated as hints, not authoritative decisions.\n"
            "- The V8.2 retrieval-control labels are mechanically validated, but did "
            "not receive a separate independent blind-label QA pass before this "
            "local candidate package.\n"
        )
        answerability_head_row = (
            "| `answerability_shape` | 11 answer-shape labels | Query-only hint "
            "for the answer shape the evidence must support. |"
        )
    else:
        comparison = (
            "Compared with `pyrrho-nano-g3`, this package adds multitask heads "
            "for pre-retrieval query-contract classification, semantic route/domain, "
            "taxonomy pattern, and scalar governance signals."
        )
        data_description = (
            "Trained on fitz-gov V8.1-style rows prepared from the V8.0.1 row set "
            "plus the mandatory `routing.query_contract` field."
        )
        limitation_extra = ""
        answerability_head_row = ""

    scalar_names = ", ".join(f"`{field}`" for field in scalar_fields)
    scalar_count = len(scalar_fields)
    head_rows = [
        "| `governance` | `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY` | Post-retrieval evidence sufficiency and conflict decision. |",
        "| `query_contract` | `evidence_sufficiency`, `structured_lookup`, `temporal_grounding`, `exhaustive_coverage`, `comparison_coverage`, `representative_overview` | Pre-retrieval routing signal for what kind of evidence the query needs. |",
        "| `route` | `science_medicine`, `law_policy`, `history_geography`, `technology_computing`, `economics_finance`, `culture_society`, `general_commonsense` | Semantic route/domain signal for retrieval policy and logging. |",
        "| `taxonomy` | 23 fitz-gov taxonomy patterns | Failure/support pattern signal for audit and diagnostics. |",
        f"| `scalars` | {scalar_names} | Continuous governance signals for retry, ranking, and monitoring. |",
    ]
    if retrieval_control:
        head_rows.extend(
            [
                "| `retrieval_action` | `answer_now`, `retrieve_more`, `broaden_search`, `resolve_conflict`, `ask_clarifying_question`, `structured_lookup` | Retrieval policy hint for the next pipeline action. |",
                "| `gap_type` | 12 evidence-gap labels | More specific reason why retrieval is insufficient or conflicting. |",
                answerability_head_row,
                "| `retrieval_modality` | `unstructured_text`, `structured_table`, `code`, `configuration`, `log_trace`, `pdf_layout`, `mixed` | Query-only hint for the preferred retrieval substrate. |",
            ]
        )
        if has_retrieval_obligation:
            head_rows.append(
                "| `retrieval_obligation` | 31 V10 obligation labels | Query-only target/closure obligation for corpus-aware retrieval planning. |"
            )
    output_rows = [
        "| `governance.final_label` | Final calibrated label after the TRUSTWORTHY threshold rule. |",
        "| `governance.raw_label` | Highest-probability governance label before threshold calibration. |",
        "| `governance.probabilities` | Probability distribution over `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY`. |",
        "| `governance.threshold` | TRUSTWORTHY probability threshold used by the package. |",
        "| `query_contract.final_label` | Query-only contract prediction. |",
        "| `route.final_label` | Query-only semantic route/domain prediction. |",
        "| `taxonomy.final_label` | Query+evidence taxonomy-pattern prediction. |",
        f"| `scalars` | {scalar_count} bounded scalar governance signals. |",
    ]
    if retrieval_control:
        output_rows.extend(
            [
                "| `retrieval_action.final_label` | Retrieval policy hint. |",
                "| `gap_type.final_label` | Evidence-gap type prediction. |",
                "| `answerability_shape.final_label` | Query-only answer-shape prediction. |",
                "| `retrieval_modality.final_label` | Query-only retrieval-modality prediction. |",
            ]
        )
        if has_retrieval_obligation:
            output_rows.append(
                "| `retrieval_obligation.final_label` | Query-only retrieval-obligation prediction. |"
            )
    output_rows.append("| `timing_ms` | Local inference timing for the call. |")
    single_seed_rows = compact_table(
        [
            metric_row(test, "Governance accuracy", "gov_accuracy"),
            metric_row(test, "False-TRUSTWORTHY rate", "gov_false_trustworthy_rate"),
            metric_row(test, "Query-contract accuracy", "query_contract_accuracy"),
            metric_row(test, "Query-contract macro F1", "query_contract_macro_f1"),
            metric_row(test, "Route accuracy", "route_accuracy"),
            metric_row(test, "Route macro F1", "route_macro_f1"),
            metric_row(test, "Taxonomy accuracy", "taxonomy_accuracy"),
            metric_row(test, "Taxonomy macro F1", "taxonomy_macro_f1"),
            metric_row(test, "Scalar MAE", "scalar_mae"),
            metric_row(test, "Retrieval-action macro F1", "retrieval_action_macro_f1"),
            metric_row(test, "Gap-type macro F1", "gap_type_macro_f1"),
            metric_row(test, "Answerability-shape macro F1", "answerability_shape_macro_f1"),
            metric_row(test, "Retrieval-modality macro F1", "retrieval_modality_macro_f1"),
            metric_row(test, "Retrieval-obligation macro F1", "retrieval_obligation_macro_f1"),
        ]
    )
    aggregate_rows = compact_table(
        [
            aggregate_row(summary, "Governance accuracy", "gov_accuracy", percent=True),
            aggregate_row(
                summary,
                "False-TRUSTWORTHY rate",
                "gov_false_trustworthy_rate",
                percent=True,
            ),
            aggregate_row(
                summary,
                "Query-contract macro F1",
                "query_contract_macro_f1",
                percent=True,
            ),
            aggregate_row(summary, "Route accuracy", "route_accuracy", percent=True),
            aggregate_row(summary, "Taxonomy accuracy", "taxonomy_accuracy", percent=True),
            aggregate_row(summary, "Scalar MAE", "scalar_mae", percent=False),
            aggregate_row(
                summary,
                "Retrieval-action macro F1",
                "retrieval_action_macro_f1",
                percent=True,
            ),
            aggregate_row(summary, "Gap-type macro F1", "gap_type_macro_f1", percent=True),
            aggregate_row(
                summary,
                "Answerability-shape macro F1",
                "answerability_shape_macro_f1",
                percent=True,
            ),
            aggregate_row(
                summary,
                "Retrieval-modality macro F1",
                "retrieval_modality_macro_f1",
                percent=True,
            ),
            aggregate_row(
                summary,
                "Retrieval-obligation macro F1",
                "retrieval_obligation_macro_f1",
                percent=True,
            ),
        ]
    )
    seed_count = len(summary.get("seeds") or [])
    aggregate_heading = (
        "Single-seed local candidate headline:"
        if seed_count == 1
        else "Three-seed headline from the local release summary:"
    )
    example_retrieval_action = (
        '  "retrieval_action": {\n    "final_label": "answer_now"\n  },\n'
        if retrieval_control
        else ""
    )
    example_scalar_rows = ",\n".join(
        f'    "{field}": {value:.2f}'
        for field, value in zip(
            scalar_fields,
            (0.91, 0.88, 0.86, 0.08, 0.12, 0.09, 0.07),
            strict=False,
        )
    )
    intended_use_rows = [
        "- whether retrieved evidence is enough to answer,",
        "- whether retrieved evidence conflicts,",
        "- what kind of evidence the query needs before retrieval,",
        "- which semantic/domain route the query belongs to,",
        "- which fitz-gov support/failure pattern is active,",
    ]
    if retrieval_control:
        intended_use_rows.append(
            "- what retrieval action and gap type the evidence state suggests,"
        )
    intended_use_rows.append("- whether retrieval should retry, broaden, or escalate.")
    quick_start_rows = [
        'print(result["governance"]["final_label"])',
        'print(result["query_contract"]["final_label"])',
        'print(result["route"]["final_label"])',
        'print(result["taxonomy"]["final_label"])',
    ]
    if retrieval_control:
        quick_start_rows.extend(
            [
                'print(result["retrieval_action"]["final_label"])',
                'print(result["gap_type"]["final_label"])',
            ]
        )
        if has_retrieval_obligation:
            quick_start_rows.append('print(result["retrieval_obligation"]["final_label"])')
    quick_start_rows.append('print(result["scalars"])')
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
quality layer. {comparison}

## Governance Labels

| Label | Meaning |
|---|---|
| `ABSTAIN` | The retrieved sources do not contain enough evidence to answer the question. |
| `DISPUTED` | The retrieved sources conflict on the answer. |
| `TRUSTWORTHY` | The retrieved sources consistently support answering the question. |

## Multitask Heads

| Head | Labels / values | Intended use |
|---|---|---|
{chr(10).join(head_rows)}

## Outputs

This is a custom multitask package, not a standard single-head
`AutoModelForSequenceClassification` artifact. The recommended runtime is
`pyrrho.multitask_inference.PyrrhoMultiTaskPredictor` from the pyrrho repository.

The predictor returns a structured object:

| Field | Meaning |
|---|---|
{chr(10).join(output_rows)}

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
{example_retrieval_action}  "scalars": {{
{example_scalar_rows}
  }}
}}
```

The model does not generate answers, citations, source spans, retrieval results,
or natural-language explanations. It classifies and scores the `(query,
retrieved_contexts)` evidence state.

## Intended Use

Use this model when a RAG or retrieval package needs fast local signals about:

{chr(10).join(intended_use_rows)}

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

{chr(10).join(quick_start_rows)}
```

For local package testing:

```powershell
python scripts/package_multitask_encoder.py verify --package-dir models/{model_name} --device cpu
```

## Release Selection

- Seed: `{seed}`
- TRUSTWORTHY threshold: `{threshold:.2f}`
- Selection reason: {selection_reason}

## Held-Out Test Metrics

| Metric | Result |
|---|---:|
{single_seed_rows}

{aggregate_heading}

| Metric | Mean +/- std |
|---|---:|
{aggregate_rows}

## Training Data

{data_description} The release package records the local training config in
`training_config.yaml` and detailed metrics in `reports/summary.json`.

## Limitations

- This is a governance and routing co-processor, not a generator.
- The auxiliary heads are useful signals, not ground-truth explanations.
- Query-contract and route predictions are query-only and can be wrong when the
  user query is underspecified.
- Taxonomy and scalar outputs are trained on fitz-gov labels/signals and should
  be treated as decision-support metadata, not universal factual judgments.
{limitation_extra}- The license is CC BY-NC 4.0. Commercial use requires a separate license.
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
        row = {
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
        for key in (
            "retrieval_action",
            "gap_type",
            "answerability_shape",
            "retrieval_modality",
            "retrieval_obligation",
        ):
            if key in prediction:
                row[key] = prediction[key]
        rows.append(row)
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

    selection_reason = args.selection_reason or default_selection_reason(
        args.model_name,
        args.seed,
    )
    readme = release_readme(
        model_name=args.model_name,
        seed=args.seed,
        threshold=threshold,
        report=report,
        summary=summary,
        scalar_fields=list(model.pyrrho_config.scalar_fields),
        selection_reason=selection_reason,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    smoke = run_smoke(output_dir, device=args.device, trustworthy_threshold=threshold)
    write_json(reports_dir / "package_smoke.json", smoke)

    heads = {
        "governance_id2label": model.pyrrho_config.id2label,
        "query_contract_id2label": model.pyrrho_config.query_contract_id2label,
        "route_id2label": model.pyrrho_config.route_id2label,
        "taxonomy_id2label": model.pyrrho_config.taxonomy_id2label,
        "scalar_fields": list(model.pyrrho_config.scalar_fields),
    }
    for key in (
        "retrieval_action_id2label",
        "gap_type_id2label",
        "answerability_shape_id2label",
        "retrieval_modality_id2label",
        "retrieval_obligation_id2label",
    ):
        mapping = getattr(model.pyrrho_config, key)
        if mapping:
            heads[key] = mapping

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
            "selection_reason": selection_reason,
        },
        "heads": heads,
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
