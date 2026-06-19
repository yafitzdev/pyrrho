"""Prepare fresh fitz-gov-v2 bulk rows for pyrrho multitask training.

The generator writes flat JSONL rows that already contain the pyrrho multitask
labels. This script is deliberately strict: it does not infer labels, it only
validates that each name/id pair matches the canonical v2 contract, normalizes
nullable obligation rows, and materializes train/eval/test JSONL plus metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


LABEL2ID = {"ABSTAIN": 0, "DISPUTED": 1, "TRUSTWORTHY": 2}
ROUTE2ID = {
    "science_medicine": 0,
    "law_policy": 1,
    "history_geography": 2,
    "technology_computing": 3,
    "economics_finance": 4,
    "culture_society": 5,
    "general_commonsense": 6,
}
QUERY_CONTRACT2ID = {
    "evidence_sufficiency": 0,
    "structured_lookup": 1,
    "temporal_grounding": 2,
    "exhaustive_coverage": 3,
    "comparison_coverage": 4,
    "representative_overview": 5,
}
RETRIEVAL_ACTION2ID = {
    "answer_now": 0,
    "retrieve_more": 1,
    "broaden_search": 2,
    "resolve_conflict": 3,
    "ask_clarifying_question": 4,
    "structured_lookup": 5,
}
GAP_TYPE2ID = {
    "none": 0,
    "missing_specific_fact": 1,
    "missing_timeframe": 2,
    "missing_comparison_side": 3,
    "missing_source_authority": 4,
    "conflicting_values": 5,
    "wrong_entity": 6,
    "wrong_version_or_scope": 7,
    "too_broad": 8,
    "incomplete_enumeration": 9,
    "unsupported_inference": 10,
    "ambiguous_query": 11,
}
ANSWERABILITY_SHAPE2ID = {
    "direct_answer": 0,
    "synthesis_answer": 1,
    "set_answer": 2,
    "structured_reasoning": 3,
}
RETRIEVAL_MODALITY2ID = {
    "unstructured_text": 0,
    "structured_table": 1,
    "code": 2,
    "configuration": 3,
    "log_trace": 4,
    "pdf_layout": 5,
    "mixed": 6,
}
RETRIEVAL_OBLIGATION2ID = {
    "row_key_lookup": 0,
    "column_value_lookup": 1,
    "multi_row_comparison": 2,
    "aggregate_or_count": 3,
    "stale_row_version": 4,
    "symbol_definition": 5,
    "constant_or_env_var": 6,
    "call_path_or_helper": 7,
    "test_or_execution_result": 8,
    "versioned_api_behavior": 9,
    "config_key_value": 10,
    "default_or_fallback": 11,
    "environment_override": 12,
    "version_scope": 13,
    "conflicting_config_sources": 14,
    "status_or_outcome": 15,
    "timestamp_ordering": 16,
    "error_signature": 17,
    "correlation_id": 18,
    "missing_run_result": 19,
    "table_or_figure_reference": 20,
    "footnote_or_caption": 21,
    "section_header_scope": 22,
    "page_or_revision_scope": 23,
    "form_or_field_value": 24,
    "prose_plus_table": 25,
    "prose_plus_code": 26,
    "table_plus_config": 27,
    "policy_plus_latest_row": 28,
    "log_plus_config": 29,
    "code_plus_changelog": 30,
}
TAXONOMY_PATTERN2ID = {
    "authority_conflict": 0,
    "authority_status_conflict": 1,
    "consistent_chain": 2,
    "definitional_conflict": 3,
    "direct_answer": 4,
    "evidence_absent": 5,
    "expert_consensus": 6,
    "factual_contradiction": 7,
    "missing_execution_result": 8,
    "multi_source_corroboration": 9,
    "numerical_conflict": 10,
    "partial_overlap": 11,
    "quantitative_consensus": 12,
    "resolved_candidate_selection": 13,
    "scope_conflict": 14,
    "single_authoritative": 15,
    "temporal_conflict": 16,
    "temporal_mismatch": 17,
    "too_general": 18,
    "verdict_conflict": 19,
    "version_build_mismatch": 20,
    "wrong_entity": 21,
    "wrong_specificity": 22,
}
SCALAR_FIELDS = (
    "evidence_sufficiency",
    "query_evidence_alignment",
    "answer_coverage",
    "conflict_density",
    "retrieval_retry_value",
    "false_trustworthy_risk",
    "evidence_failure_severity",
)
PROFILE_FIELDS = (
    "pack_shape",
    "failure_focus",
    "decoy_profile",
    "authority_profile",
    "temporal_profile",
    "source_format",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../fitz-gov-modern_generator/outputs/bulk_20000"),
        help="Directory containing worker_*.jsonl generated rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/multitask_v2_g1_alpha_20k"),
        help="Prepared pyrrho multitask output directory.",
    )
    parser.add_argument("--glob", default="worker_*.jsonl")
    parser.add_argument("--expected-rows", type=int, default=20000)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument(
        "--split-strategy",
        choices=("hash_id", "source"),
        default="hash_id",
        help="Use deterministic ID-hash splits by default because bulk generator rows are queued as train.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            row["_source_file"] = str(path)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def require_text(row: dict[str, Any], field: str, errors: list[str]) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{row.get('id')}: missing nonempty {field}")
        return ""
    return value.strip()


def check_pair(
    row: dict[str, Any],
    *,
    name_field: str,
    id_field: str,
    mapping: dict[str, int],
    errors: list[str],
) -> None:
    name = row.get(name_field)
    raw_id = row.get(id_field)
    if not isinstance(name, str) or name not in mapping:
        errors.append(f"{row.get('id')}: invalid {name_field}={name!r}")
        return
    if int(raw_id) != int(mapping[name]):
        errors.append(
            f"{row.get('id')}: {id_field}={raw_id!r} does not match {name_field}={name!r}"
        )


def normalize_obligation(row: dict[str, Any], errors: list[str]) -> None:
    obligation = row.get("retrieval_obligation")
    raw_id = row.get("retrieval_obligation_id", -1)
    if obligation in (None, "", "null"):
        row["retrieval_obligation"] = None
        row["retrieval_obligation_raw"] = None
        row["retrieval_obligation_id"] = -1
        return
    if not isinstance(obligation, str) or obligation not in RETRIEVAL_OBLIGATION2ID:
        errors.append(f"{row.get('id')}: invalid retrieval_obligation={obligation!r}")
        return
    if int(raw_id) != RETRIEVAL_OBLIGATION2ID[obligation]:
        errors.append(
            f"{row.get('id')}: retrieval_obligation_id={raw_id!r} "
            f"does not match retrieval_obligation={obligation!r}"
        )
    row["retrieval_obligation_raw"] = obligation


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("_source_file", None)
    out.pop("_source_line", None)
    out["version"] = out.get("schema_version") or "fitz-gov-v2-1.0-candidate"
    out["dataset_version"] = out.get("dataset_version") or "v2_candidate"
    out["query_contract_raw"] = out.get("query_contract")
    out["retrieval_action_raw"] = out.get("retrieval_action")
    out["gap_type_raw"] = out.get("gap_type")
    out["answerability_shape_detailed"] = out.get("answerability_shape")
    out["retrieval_modality_raw"] = out.get("retrieval_modality")
    out["taxonomy_pattern_raw"] = out.get("taxonomy_pattern")
    out["alpha_source"] = "fitz_gov_v2_bulk_20k_fresh_synthetic"
    if out.get("retrieval_obligation") in (None, "", "null"):
        out["retrieval_obligation"] = None
        out["retrieval_obligation_raw"] = None
        out["retrieval_obligation_id"] = -1
    return out


def deterministic_split(row_id: str) -> str:
    bucket = int(hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "eval"
    return "test"


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    row_id = require_text(row, "id", errors)
    split = row.get("split")
    if split not in {"train", "eval", "test"}:
        errors.append(f"{row_id}: invalid split={split!r}")
    require_text(row, "text", errors)
    require_text(row, "query_text", errors)
    require_text(row, "query", errors)
    contexts = row.get("contexts")
    if not isinstance(contexts, list) or not contexts or not all(
        isinstance(item, str) and item.strip() for item in contexts
    ):
        errors.append(f"{row_id}: contexts must be a nonempty list of strings")
    context_features = row.get("context_features")
    if not isinstance(context_features, list) or len(context_features) != len(contexts or []):
        errors.append(f"{row_id}: context_features count must match contexts")

    check_pair(row, name_field="label", id_field="label_id", mapping=LABEL2ID, errors=errors)
    check_pair(row, name_field="route", id_field="route_id", mapping=ROUTE2ID, errors=errors)
    check_pair(
        row,
        name_field="query_contract",
        id_field="query_contract_id",
        mapping=QUERY_CONTRACT2ID,
        errors=errors,
    )
    check_pair(
        row,
        name_field="retrieval_action",
        id_field="retrieval_action_id",
        mapping=RETRIEVAL_ACTION2ID,
        errors=errors,
    )
    check_pair(row, name_field="gap_type", id_field="gap_type_id", mapping=GAP_TYPE2ID, errors=errors)
    check_pair(
        row,
        name_field="answerability_shape",
        id_field="answerability_shape_id",
        mapping=ANSWERABILITY_SHAPE2ID,
        errors=errors,
    )
    check_pair(
        row,
        name_field="retrieval_modality",
        id_field="retrieval_modality_id",
        mapping=RETRIEVAL_MODALITY2ID,
        errors=errors,
    )
    check_pair(
        row,
        name_field="taxonomy_pattern",
        id_field="taxonomy_pattern_id",
        mapping=TAXONOMY_PATTERN2ID,
        errors=errors,
    )
    normalize_obligation(row, errors)

    scalar_targets = row.get("scalar_targets")
    if not isinstance(scalar_targets, dict):
        errors.append(f"{row_id}: scalar_targets must be an object")
    else:
        for field in SCALAR_FIELDS:
            value = scalar_targets.get(field)
            if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{row_id}: invalid scalar_targets.{field}={value!r}")
    return errors


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field)) for row in rows))


def main() -> int:
    args = parse_args()
    start = time.time()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = sorted(input_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no files matched {input_dir / args.glob}")

    raw_rows: list[dict[str, Any]] = []
    for path in paths:
        raw_rows.extend(read_jsonl(path))

    errors: list[str] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_queries: Counter[str] = Counter()
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}

    for row in raw_rows:
        row_errors = validate_row(row)
        errors.extend(row_errors)
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        seen_ids.add(row_id)
        query_key = str(row.get("query") or "").strip().lower()
        if query_key:
            duplicate_queries[query_key] += 1
        if len(errors) > args.max_errors:
            break
        if not row_errors:
            normalized = normalize_row(row)
            normalized["source_split"] = normalized.get("split")
            normalized["split"] = (
                deterministic_split(str(normalized["id"]))
                if args.split_strategy == "hash_id"
                else str(normalized["split"])
            )
            rows_by_split[str(normalized["split"])].append(normalized)

    repeated_queries = {query: count for query, count in duplicate_queries.items() if count > 1}
    if duplicate_ids:
        errors.extend(f"duplicate id: {row_id}" for row_id in duplicate_ids[: args.max_errors])
    if len(raw_rows) != args.expected_rows:
        errors.append(f"expected {args.expected_rows} rows, found {len(raw_rows)}")

    if errors:
        report = {
            "status": "failed",
            "input_dir": str(input_dir),
            "rows_read": len(raw_rows),
            "errors": errors[: args.max_errors],
            "error_count": len(errors),
            "duplicate_id_count": len(duplicate_ids),
            "duplicate_query_count": len(repeated_queries),
        }
        write_json(output_dir / "validation_report.json", report)
        raise SystemExit(f"validation failed with {len(errors)} errors; see {output_dir / 'validation_report.json'}")

    for split, split_rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", sorted(split_rows, key=lambda row: row["id"]))

    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    retrieval_obligation_counts = Counter(
        str(row.get("retrieval_obligation"))
        for row in all_rows
        if row.get("retrieval_obligation") is not None
    )
    metadata = {
        "source": {
            "dataset_family": "fitz-gov-v2",
            "generator_repo": str((Path.cwd().parent / "fitz-gov-modern_generator").resolve()),
            "input_dir": str(input_dir),
            "input_files": [str(path) for path in paths],
            "candidate_status": "fresh_synthetic_v2_round1_generator_validated",
            "blind_label_qa": "not_run",
            "split_strategy": args.split_strategy,
        },
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "label_counts": counts(all_rows, "label"),
        "dataset_version_counts": counts(all_rows, "dataset_version"),
        "route_counts": counts(all_rows, "route"),
        "difficulty_counts": counts(all_rows, "difficulty"),
        "query_contract_counts": counts(all_rows, "query_contract"),
        "answerability_shape_counts": counts(all_rows, "answerability_shape"),
        "retrieval_modality_counts": counts(all_rows, "retrieval_modality"),
        "retrieval_action_counts": counts(all_rows, "retrieval_action"),
        "gap_type_counts": counts(all_rows, "gap_type"),
        "taxonomy_pattern_counts": counts(all_rows, "taxonomy_pattern"),
        "retrieval_obligation_counts": dict(retrieval_obligation_counts),
        "retrieval_obligation_labeled_rows": sum(
            1 for row in all_rows if row.get("retrieval_obligation") is not None
        ),
        "retrieval_obligation_masked_rows": sum(
            1 for row in all_rows if row.get("retrieval_obligation") is None
        ),
        "profile_counts": {field: counts(all_rows, field) for field in PROFILE_FIELDS},
        "duplicate_query_count": len(repeated_queries),
        "duplicate_query_examples": list(repeated_queries.items())[:20],
        "route2id": ROUTE2ID,
        "taxonomy_pattern2id": TAXONOMY_PATTERN2ID,
        "query_contract2id": QUERY_CONTRACT2ID,
        "retrieval_action2id": RETRIEVAL_ACTION2ID,
        "gap_type2id": GAP_TYPE2ID,
        "answerability_shape2id": ANSWERABILITY_SHAPE2ID,
        "retrieval_modality2id": RETRIEVAL_MODALITY2ID,
        "retrieval_obligation2id": RETRIEVAL_OBLIGATION2ID,
        "require_query_contract": True,
        "require_retrieval_control": True,
        "head_input_sources": {
            "retrieval_action": "evidence",
            "gap_type": "evidence",
            "answerability_shape": "query",
            "retrieval_modality": "query",
            "retrieval_obligation": "query",
        },
        "scalar_fields": list(SCALAR_FIELDS),
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(
        output_dir / "validation_report.json",
        {
            "status": "passed",
            "input_dir": str(input_dir),
            "rows_read": len(raw_rows),
            "rows_written": len(all_rows),
            "splits": metadata["splits"],
            "duplicate_id_count": len(duplicate_ids),
            "duplicate_query_count": len(repeated_queries),
        },
    )

    print(f"input_dir   : {input_dir}")
    print(f"output_dir  : {output_dir}")
    print(f"rows        : {len(all_rows)}")
    print(f"splits      : {metadata['splits']}")
    print(f"labels      : {metadata['label_counts']}")
    print(f"modalities  : {metadata['retrieval_modality_counts']}")
    print(f"obligations : labeled={metadata['retrieval_obligation_labeled_rows']} masked={metadata['retrieval_obligation_masked_rows']}")
    print(f"metadata    : {output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
