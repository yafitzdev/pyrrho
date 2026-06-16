"""Prepare the local pyrrho-nano-g5-alpha multitask dataset.

This alpha keeps the official fitz-gov V9.0.0 rows as the stable base and adds
the V10 5/5/5 local candidate rows for corpus-aware retrieval planning.

V9 rows do not have retrieval-obligation labels, so they are masked with
`retrieval_obligation_id = -1`. V10 rows carry the new
`routing.retrieval_control.retrieval_obligation.kind` label and train the new
query-only obligation head.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pyrrho.data import RETRIEVAL_OBLIGATION_LABEL2ID
from pyrrho.manifest import write_manifest

from prepare_g4_alpha_data import (
    COLLAPSED_ANSWERABILITY_LABEL2ID,
    DETAILED_TO_COLLAPSED,
    deterministic_split,
    flatten_sdgp_case,
    get_path,
    read_json,
    read_jsonl,
    retrieval_kind,
    unwrap_case,
    write_jsonl,
)


DEFAULT_CANDIDATE_DIR = (
    Path("../fitz-gov")
    / "data"
    / "_workspaces"
    / "handoff"
    / "v10_cogeneration_5x5_20260612"
    / "subagent_outputs"
)


def fallback_shape_from_contract(contract: str) -> str:
    raw = contract.strip().lower()
    if raw == "exhaustive_coverage":
        return "set_answer"
    if raw in {"comparison_coverage", "structured_lookup", "temporal_grounding"}:
        return "structured_reasoning"
    if raw == "representative_overview":
        return "synthesis_answer"
    return "direct_answer"


def collapse_v10_answerability_shape(raw_shape: str, *, query_contract: str) -> str:
    raw = raw_shape.strip()
    if not raw:
        return fallback_shape_from_contract(query_contract)
    if raw in COLLAPSED_ANSWERABILITY_LABEL2ID:
        return raw
    if raw in DETAILED_TO_COLLAPSED:
        return DETAILED_TO_COLLAPSED[raw]
    lowered = raw.lower()
    if any(token in lowered for token in ("list", "enumeration", "coverage")):
        return "set_answer"
    if any(
        token in lowered
        for token in (
            "comparison",
            "comparative",
            "compare",
            "aggregate",
            "aggregation",
            "count",
            "temporal",
            "ordering",
            "order",
            "sequence",
            "event",
            "trace",
            "chain",
            "behavior",
            "correlation",
            "causal",
            "diagnostic",
            "diagnosis",
            "conflict",
            "resolution",
            "reconciliation",
            "disambiguation",
            "choose_between",
            "multi_row",
            "versioned_behavior",
            "process",
        )
    ):
        return "structured_reasoning"
    if any(
        token in lowered
        for token in (
            "summary",
            "explanation",
            "interpretation",
            "policy_determination",
            "policy_inference",
            "judgment",
            "causal_assessment",
            "causal_or_outcome_claim",
            "open_ended",
            "inference",
        )
    ):
        return "synthesis_answer"
    if any(
        token in lowered
        for token in (
            "exact",
            "lookup",
            "call_path",
            "value",
            "field",
            "row",
            "column",
            "key",
            "status",
            "state",
            "outcome",
            "result",
            "execution",
            "signature",
            "identifier",
            "definition",
            "boolean",
            "yes_no",
            "binary",
            "answerable",
            "complete_answer",
            "verification",
            "extract",
            "extraction",
            "layout",
            "caption",
            "footnote",
            "section",
            "page",
            "form",
            "scope",
            "current",
            "latest",
        )
    ):
        return "direct_answer"
    if any(
        token in lowered
        for token in (
            "insufficient",
            "missing",
            "unanswerable",
            "not_answerable",
            "underspecified",
            "ambiguous",
            "clarification",
            "wrong_aspect",
            "unsupported",
        )
    ):
        return fallback_shape_from_contract(query_contract)
    raise ValueError(f"unknown V10 answerability shape: {raw_shape!r}")


def default_retrieval_action(case: dict[str, Any], gap_type: str) -> str:
    label = str(get_path(case, ("governance", "classification")) or "").upper()
    gap = gap_type.strip().lower()
    if label == "TRUSTWORTHY":
        return "answer_now"
    if label == "DISPUTED" or "conflict" in gap:
        return "resolve_conflict"
    if gap == "ambiguous_query":
        return "ask_clarifying_question"
    return "retrieve_more"


def default_gap_type(case: dict[str, Any]) -> str:
    label = str(get_path(case, ("governance", "classification")) or "").upper()
    if label == "TRUSTWORTHY":
        return "none"
    if label == "DISPUTED":
        return "conflicting_values"
    return "missing_specific_fact"


def case_with_normalized_retrieval_control(
    case: dict[str, Any],
) -> tuple[dict[str, Any], str, str, list[str]]:
    out = copy.deepcopy(case)
    filled: list[str] = []
    retrieval_control = out.setdefault("routing", {}).setdefault("retrieval_control", {})

    gap_type_block = retrieval_control.setdefault("gap_type", {})
    gap_type = str(gap_type_block.get("kind") or "")
    if not gap_type:
        gap_type = default_gap_type(out)
        gap_type_block["kind"] = gap_type
        filled.append("gap_type")

    action_block = retrieval_control.setdefault("retrieval_action", {})
    if not str(action_block.get("kind") or ""):
        action_block["kind"] = default_retrieval_action(out, gap_type)
        filled.append("retrieval_action")

    modality_block = retrieval_control.setdefault("preferred_retrieval_modality", {})
    if not str(modality_block.get("kind") or ""):
        modality_block["kind"] = str(get_path(out, ("meta", "modality")) or "unstructured_text")
        filled.append("preferred_retrieval_modality")

    raw_shape = str(get_path(out, ("routing", "retrieval_control", "answerability_shape", "kind")) or "")
    query_contract = str(get_path(out, ("routing", "query_contract", "kind")) or "")
    collapsed = collapse_v10_answerability_shape(raw_shape, query_contract=query_contract)
    answerability_shape = retrieval_control.setdefault("answerability_shape", {})
    if not raw_shape:
        filled.append("answerability_shape")
    answerability_shape["kind"] = collapsed
    return out, raw_shape, collapsed, filled


def add_retrieval_obligation(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    raw = retrieval_kind(case, "retrieval_obligation")
    if raw not in RETRIEVAL_OBLIGATION_LABEL2ID:
        raise ValueError(f"{case.get('id')}: unknown retrieval_obligation {raw!r}")
    row["retrieval_obligation"] = raw
    row["retrieval_obligation_raw"] = raw
    row["retrieval_obligation_id"] = RETRIEVAL_OBLIGATION_LABEL2ID[raw]
    return row


def mask_official_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["retrieval_obligation"] = None
    out["retrieval_obligation_raw"] = None
    out["retrieval_obligation_id"] = -1
    out["alpha_source"] = str(out.get("alpha_source") or "fitz_gov_v9_0_0_public")
    return out


def add_row(
    rows_by_split: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
    seen_ids: set[str],
    source_counts: Counter[str],
) -> None:
    case_id = str(row["id"])
    if case_id in seen_ids:
        return
    seen_ids.add(case_id)
    rows_by_split[row["split"]].append(row)
    source_counts[str(row.get("alpha_source") or "unknown")] += 1


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    return {
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "label_counts": dict(Counter(str(row["label"]) for row in all_rows)),
        "dataset_version_counts": dict(
            Counter(str(row["dataset_version"]) for row in all_rows)
        ),
        "source_counts_by_split": {
            split: dict(Counter(str(row.get("alpha_source") or "unknown") for row in rows))
            for split, rows in rows_by_split.items()
        },
        "answerability_shape_counts": dict(
            Counter(str(row["answerability_shape"]) for row in all_rows)
        ),
        "retrieval_modality_counts": dict(
            Counter(str(row["retrieval_modality"]) for row in all_rows)
        ),
        "retrieval_obligation_counts": dict(
            Counter(
                str(row["retrieval_obligation"])
                for row in all_rows
                if int(row.get("retrieval_obligation_id", -1)) >= 0
            )
        ),
        "retrieval_obligation_labeled_rows": sum(
            1 for row in all_rows if int(row.get("retrieval_obligation_id", -1)) >= 0
        ),
        "retrieval_obligation_masked_rows": sum(
            1 for row in all_rows if int(row.get("retrieval_obligation_id", -1)) < 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path("data/multitask_g4_v9_official"),
        help="Official V9 multitask data dir.",
    )
    parser.add_argument(
        "--fitz-gov-repo",
        type=Path,
        default=Path("../fitz-gov"),
        help="Local fitz-gov repo, recorded in manifest.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="V10 5/5/5 candidate JSONL output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/multitask_g5_alpha_v10_5x5_candidate"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/encoder/modernbert_base_g5_alpha_v10_5x5_candidate.yaml"),
        help="Config path recorded in the output manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    base_dir = args.base_data_dir.resolve()
    fitz_gov_repo = args.fitz_gov_repo.resolve()
    candidate_dir = args.candidate_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata = read_json(base_dir / "metadata.json")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    seen_ids: set[str] = set()
    source_counts: Counter[str] = Counter()

    for split in ("train", "eval", "test"):
        for row in read_jsonl(base_dir / f"{split}.jsonl"):
            add_row(rows_by_split, mask_official_row(row), seen_ids, source_counts)

    candidate_files = sorted(candidate_dir.glob("batch_*.jsonl"))
    if not candidate_files:
        raise FileNotFoundError(f"no candidate files found in {candidate_dir}")

    candidate_rows = 0
    for path in candidate_files:
        for raw in read_jsonl(path):
            case = unwrap_case(raw)
            if get_path(case, ("meta", "dataset_version")) != "v10":
                raise ValueError(f"{case.get('id')}: expected meta.dataset_version='v10'")
            flattened_case, raw_shape, collapsed_shape, filled_fields = (
                case_with_normalized_retrieval_control(case)
            )
            split = deterministic_split(str(case.get("id") or ""))
            flat = flatten_sdgp_case(
                flattened_case,
                split=split,
                metadata=metadata,
                source_kind="fitz_gov_v10_5x5_candidate_no_blind_qa",
            )
            flat["answerability_shape_raw_v10"] = raw_shape
            flat["answerability_shape_collapsed_v10"] = collapsed_shape
            flat["v10_filled_retrieval_control_fields"] = filled_fields
            add_row(
                rows_by_split,
                add_retrieval_obligation(flat, case),
                seen_ids,
                source_counts,
            )
            candidate_rows += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = summarize(rows_by_split)
    out_metadata = {
        "source": {
            "base_data_dir": str(base_dir),
            "fitz_gov_repo": str(fitz_gov_repo),
            "base_fitz_gov_version": "v9.0.0",
            "v10_candidate_dir": str(candidate_dir),
            "v10_candidate_status": (
                "post_repair_validation_clean_quality_audit_clean_no_blind_label_qa"
            ),
        },
        **summary,
        "candidate_v10_rows_read": candidate_rows,
        "route2id": metadata["route2id"],
        "taxonomy_pattern2id": metadata["taxonomy_pattern2id"],
        "query_contract2id": metadata["query_contract2id"],
        "retrieval_action2id": metadata["retrieval_action2id"],
        "gap_type2id": metadata["gap_type2id"],
        "answerability_shape2id": metadata["answerability_shape2id"],
        "answerability_shape_collapse_map": metadata.get("answerability_shape_collapse_map", {}),
        "retrieval_modality2id": metadata["retrieval_modality2id"],
        "retrieval_obligation2id": RETRIEVAL_OBLIGATION_LABEL2ID,
        "require_query_contract": True,
        "require_retrieval_control": True,
        "scalar_fields": metadata["scalar_fields"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(out_metadata, indent=2), encoding="utf-8")
    write_manifest(
        output_dir,
        args.config,
        seed=0,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=fitz_gov_repo,
        extra={
            "script": "prepare_g5_alpha_data.py",
            "base_data_dir": str(base_dir),
            "candidate_dir": str(candidate_dir),
            "candidate_rows": candidate_rows,
        },
        start_time=start,
    )

    print(f"output       : {output_dir}")
    print(f"splits       : {out_metadata['splits']}")
    print(f"sources      : {dict(source_counts)}")
    print(f"v10 rows     : {candidate_rows}")
    print(
        "obligations  : labeled={labeled} masked={masked} labels={labels}".format(
            labeled=out_metadata["retrieval_obligation_labeled_rows"],
            masked=out_metadata["retrieval_obligation_masked_rows"],
            labels=len(out_metadata["retrieval_obligation_counts"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
