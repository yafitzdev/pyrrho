"""Prepare the local pyrrho-nano-g4-alpha multitask dataset.

The alpha data is intentionally local and label-trusted:

- published fitz-gov V8.2 flattened rows from pyrrho,
- locally merged V9 rows from the fitz-gov vault,
- optionally, structurally clean but not blind-QAed V9 candidate rows.

It collapses detailed V8.2 answerability labels into the four V9 planning
labels so fitz-sage can integrate the simpler retrieval-control head now.
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

from pyrrho.data import build_encoder_text, build_query_contract_text
from pyrrho.manifest import write_manifest


COLLAPSED_ANSWERABILITY_LABELS: tuple[str, ...] = (
    "direct_answer",
    "synthesis_answer",
    "set_answer",
    "structured_reasoning",
)
COLLAPSED_ANSWERABILITY_LABEL2ID: dict[str, int] = {
    label: idx for idx, label in enumerate(COLLAPSED_ANSWERABILITY_LABELS)
}
DETAILED_TO_COLLAPSED: dict[str, str] = {
    "single_fact": "direct_answer",
    "exact_lookup": "direct_answer",
    "yes_no": "direct_answer",
    "citation_required": "direct_answer",
    "explanation": "synthesis_answer",
    "summary": "synthesis_answer",
    "list": "set_answer",
    "exhaustive_list": "set_answer",
    "comparison": "structured_reasoning",
    "timeline": "structured_reasoning",
    "calculation": "structured_reasoning",
}
SCALAR_FIELDS: tuple[str, ...] = (
    "evidence_sufficiency",
    "query_evidence_alignment",
    "answer_coverage",
    "conflict_density",
    "retrieval_retry_value",
    "false_trustworthy_risk",
    "evidence_failure_severity",
)
CANONICAL_QUERY_CONTRACT_LABELS: tuple[str, ...] = (
    "evidence_sufficiency",
    "structured_lookup",
    "temporal_grounding",
    "exhaustive_coverage",
    "comparison_coverage",
    "representative_overview",
)
CANONICAL_RETRIEVAL_ACTION_LABELS: tuple[str, ...] = (
    "answer_now",
    "retrieve_more",
    "broaden_search",
    "resolve_conflict",
    "ask_clarifying_question",
    "structured_lookup",
)
CANONICAL_GAP_TYPE_LABELS: tuple[str, ...] = (
    "none",
    "missing_specific_fact",
    "missing_timeframe",
    "missing_comparison_side",
    "missing_source_authority",
    "conflicting_values",
    "wrong_entity",
    "wrong_version_or_scope",
    "too_broad",
    "incomplete_enumeration",
    "unsupported_inference",
    "ambiguous_query",
)
CANONICAL_RETRIEVAL_MODALITY_LABELS: tuple[str, ...] = (
    "unstructured_text",
    "structured_table",
    "code",
    "configuration",
    "log_trace",
    "pdf_layout",
    "mixed",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_path(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = row
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collapse_answerability_shape(shape: str) -> str:
    if shape in COLLAPSED_ANSWERABILITY_LABEL2ID:
        return shape
    if shape not in DETAILED_TO_COLLAPSED:
        raise ValueError(f"unknown answerability shape: {shape!r}")
    return DETAILED_TO_COLLAPSED[shape]


def normalize_query_contract(kind: str, *, collapsed_shape: str) -> str:
    raw = kind.strip()
    if raw in CANONICAL_QUERY_CONTRACT_LABELS:
        return raw
    lowered = raw.lower()
    if any(
        token in lowered
        for token in (
            "timeline",
            "temporal",
            "date",
            "deadline",
            "effective_time",
            "timeframe",
            "chronolog",
        )
    ):
        return "temporal_grounding"
    if any(
        token in lowered
        for token in (
            "comparison",
            "comparative",
            "compare",
            "benchmark",
            "threshold",
            "ranked",
            "candidate",
            "status_comparison",
        )
    ):
        return "comparison_coverage"
    if any(
        token in lowered
        for token in (
            "set",
            "list",
            "enumeration",
            "membership",
            "completeness",
            "coverage",
            "exhaustive",
            "items",
        )
    ):
        return "exhaustive_coverage"
    if any(
        token in lowered
        for token in (
            "calculation",
            "compute",
            "computation",
            "arithmetic",
            "numeric",
            "numerical",
            "quantitative",
            "derivation",
            "metric",
            "trace",
            "lookup",
            "structured",
            "configuration",
            "fee",
            "dose",
        )
    ):
        return "structured_lookup"
    if any(
        token in lowered
        for token in (
            "summary",
            "synthesis",
            "explanation",
            "explanatory",
            "causal",
            "rationale",
            "interpretation",
            "policy",
            "legal",
            "rule",
            "incident",
            "mechanism",
            "attribution",
        )
    ):
        return "representative_overview"
    if collapsed_shape == "set_answer":
        return "exhaustive_coverage"
    if collapsed_shape == "structured_reasoning":
        return "structured_lookup"
    if collapsed_shape == "synthesis_answer":
        return "representative_overview"
    return "evidence_sufficiency"


def normalize_retrieval_action(kind: str) -> str:
    raw = kind.strip()
    if raw in CANONICAL_RETRIEVAL_ACTION_LABELS:
        return raw
    lowered = raw.lower()
    if lowered.startswith("answer") or lowered in {
        "accept_answer",
        "direct_answer",
        "return_answer",
        "use_retrieved_context",
        "use_retrieved_evidence",
        "use_retrieved_sources",
    }:
        return "answer_now"
    if "conflict" in lowered or "resolution" in lowered or "escalate" in lowered:
        return "resolve_conflict"
    if "clarif" in lowered or "ambiguous" in lowered:
        return "ask_clarifying_question"
    if "structured" in lowered or "lookup" in lowered:
        return "structured_lookup"
    if "broaden" in lowered or "expand" in lowered:
        return "broaden_search"
    return "retrieve_more"


def normalize_gap_type(kind: str) -> str:
    raw = kind.strip()
    if raw in CANONICAL_GAP_TYPE_LABELS:
        return raw
    lowered = raw.lower()
    if lowered in {"", "none", "no_gap"}:
        return "none"
    if "ambiguous" in lowered or "underspecified" in lowered or "ambiguity" in lowered:
        return "ambiguous_query"
    if "unsupported" in lowered or "inference" in lowered:
        return "unsupported_inference"
    if "entity" in lowered:
        return "wrong_entity"
    if any(
        token in lowered
        for token in ("version", "build", "revision", "scope", "jurisdiction", "cohort")
    ):
        return "wrong_version_or_scope"
    if any(token in lowered for token in ("time", "temporal", "date", "deadline")):
        return "missing_timeframe"
    if "comparison" in lowered or "comparative" in lowered:
        return "missing_comparison_side"
    if "authority" in lowered or "source_authority" in lowered:
        return "missing_source_authority"
    if any(token in lowered for token in ("broad", "general", "partial_scope")):
        return "too_broad"
    if any(
        token in lowered
        for token in ("enumeration", "items", "set", "list", "completeness", "member")
    ):
        return "incomplete_enumeration"
    if any(
        token in lowered
        for token in (
            "conflict",
            "contradiction",
            "contradict",
            "verdict",
            "status",
            "definition",
            "claims",
            "values",
            "timestamps",
            "policies",
        )
    ):
        return "conflicting_values"
    return "missing_specific_fact"


def normalize_retrieval_modality(kind: str) -> str:
    raw = kind.strip()
    if raw in CANONICAL_RETRIEVAL_MODALITY_LABELS:
        return raw
    lowered = raw.lower()
    if "pdf" in lowered or "layout" in lowered:
        return "pdf_layout"
    if "code" in lowered:
        return "code"
    if "config" in lowered:
        return "configuration"
    if "log" in lowered or "trace" in lowered:
        return "log_trace"
    if "mixed" in lowered or "semi_structured" in lowered:
        return "mixed"
    if "structured" in lowered or "table" in lowered or "catalog" in lowered:
        return "structured_table"
    return "unstructured_text"


def normalize_taxonomy_pattern(pattern: str, *, gap_type: str, metadata: dict[str, Any]) -> str:
    raw = pattern.strip()
    if raw in metadata["taxonomy_pattern2id"]:
        return raw
    if raw == "missing_specific_fact":
        if gap_type == "missing_comparison_side":
            return "partial_overlap"
        return "evidence_absent"
    raise KeyError(f"unknown taxonomy pattern: {raw!r}")


def deterministic_split(case_id: str) -> str:
    bucket = int(hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "eval"
    return "test"


def context_features(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": ctx.get("id"),
            "text": ctx.get("text", ""),
            "summary": ctx.get("summary"),
            "authority_score": ctx.get("authority_score"),
            "authority_signal": ctx.get("authority_signal"),
            "relevance_to_query": ctx.get("relevance_to_query"),
            "boundary_quality": ctx.get("boundary_quality"),
            "evidence_bias_score": ctx.get("evidence_bias_score"),
            "temporality": ctx.get("temporality"),
        }
        for ctx in contexts
    ]


def retrieval_kind(row: dict[str, Any], field: str) -> str:
    block = get_path(row, ("routing", "retrieval_control", field))
    if not isinstance(block, dict):
        raise ValueError(f"{row.get('id')}: missing retrieval_control.{field}")
    kind = str(block.get("kind") or "")
    if not kind:
        raise ValueError(f"{row.get('id')}: empty retrieval_control.{field}.kind")
    return kind


def retrieval_severity(row: dict[str, Any]) -> float | None:
    value = get_path(row, ("routing", "retrieval_control", "evidence_failure_severity", "score"))
    return float(value) if isinstance(value, int | float) else None


def flatten_sdgp_case(
    row: dict[str, Any],
    *,
    split: str,
    metadata: dict[str, Any],
    source_kind: str,
) -> dict[str, Any]:
    case_id = str(row.get("id") or "")
    input_block = row.get("input") or {}
    contexts = input_block.get("contexts") or []
    query = str(input_block.get("query") or "").strip()
    context_texts = [str(ctx.get("text") or "").strip() for ctx in contexts]
    governance = row.get("governance") or {}
    routing = row.get("routing") or {}
    taxonomy = row.get("taxonomy") or {}
    meta = row.get("meta") or {}
    boundary = governance.get("boundary_proximity") or {}

    label = str(governance.get("classification") or "").upper()
    route = str(routing.get("expert_fired") or "")
    raw_pattern = str(taxonomy.get("pattern") or "")
    raw_query_contract = str(get_path(row, ("routing", "query_contract", "kind")) or "")
    raw_retrieval_action = retrieval_kind(row, "retrieval_action")
    raw_gap_type = retrieval_kind(row, "gap_type")
    detailed_shape = retrieval_kind(row, "answerability_shape")
    answerability_shape = collapse_answerability_shape(detailed_shape)
    query_contract = normalize_query_contract(
        raw_query_contract,
        collapsed_shape=answerability_shape,
    )
    raw_retrieval_modality = retrieval_kind(row, "preferred_retrieval_modality")
    retrieval_action = normalize_retrieval_action(raw_retrieval_action)
    gap_type = normalize_gap_type(raw_gap_type)
    if label == "TRUSTWORTHY" and gap_type == "none":
        retrieval_action = "answer_now"
    retrieval_modality = normalize_retrieval_modality(raw_retrieval_modality)
    pattern = normalize_taxonomy_pattern(raw_pattern, gap_type=gap_type, metadata=metadata)

    label2id = {"ABSTAIN": 0, "DISPUTED": 1, "TRUSTWORTHY": 2}
    scalar_targets = {}
    for field in SCALAR_FIELDS:
        value = retrieval_severity(row) if field == "evidence_failure_severity" else governance.get(field)
        if isinstance(value, int | float):
            scalar_targets[field] = float(value)

    return {
        "id": case_id,
        "split": split,
        "version": row.get("version", ""),
        "dataset_version": meta.get("dataset_version", ""),
        "text": build_encoder_text(query, context_texts),
        "query_text": build_query_contract_text(query),
        "query": query,
        "query_rewritten": input_block.get("query_rewritten"),
        "contexts": context_texts,
        "context_features": context_features(contexts),
        "label": label,
        "label_id": label2id[label],
        "route": route,
        "route_id": metadata["route2id"][route],
        "query_contract": query_contract,
        "query_contract_raw": raw_query_contract,
        "query_contract_id": metadata["query_contract2id"][query_contract],
        "retrieval_action": retrieval_action,
        "retrieval_action_raw": raw_retrieval_action,
        "retrieval_action_id": metadata["retrieval_action2id"][retrieval_action],
        "gap_type": gap_type,
        "gap_type_raw": raw_gap_type,
        "gap_type_id": metadata["gap_type2id"][gap_type],
        "answerability_shape": answerability_shape,
        "answerability_shape_id": COLLAPSED_ANSWERABILITY_LABEL2ID[answerability_shape],
        "answerability_shape_detailed": detailed_shape,
        "retrieval_modality": retrieval_modality,
        "retrieval_modality_raw": raw_retrieval_modality,
        "retrieval_modality_id": metadata["retrieval_modality2id"][retrieval_modality],
        "secondary_expert": routing.get("secondary_expert"),
        "routing_confidence": routing.get("routing_confidence"),
        "taxonomy_pattern": pattern,
        "taxonomy_pattern_raw": raw_pattern,
        "taxonomy_pattern_id": metadata["taxonomy_pattern2id"][pattern],
        "taxonomy_cell_id": taxonomy.get("cell_id", ""),
        "difficulty": meta.get("difficulty", ""),
        "confidence_level": meta.get("confidence_level"),
        "near_miss_class": meta.get("near_miss_class") or governance.get("near_miss_class"),
        "near_miss_reason": meta.get("near_miss_reason"),
        "boundary_nearest_class": boundary.get("nearest_class") if isinstance(boundary, dict) else None,
        "boundary_distance": boundary.get("distance") if isinstance(boundary, dict) else None,
        "scalar_targets": scalar_targets,
        "evidence_chain": input_block.get("evidence_chain"),
        "grounding_targets": meta.get("grounding_targets"),
        "alpha_source": source_kind,
    }


def collapse_flat_row(row: dict[str, Any], *, source_kind: str) -> dict[str, Any]:
    out = dict(row)
    detailed = str(out.get("answerability_shape") or "")
    collapsed = collapse_answerability_shape(detailed)
    out["answerability_shape"] = collapsed
    out["answerability_shape_id"] = COLLAPSED_ANSWERABILITY_LABEL2ID[collapsed]
    out["answerability_shape_detailed"] = detailed
    out["alpha_source"] = source_kind
    return out


def unwrap_case(row: dict[str, Any]) -> dict[str, Any]:
    case = row.get("case")
    return case if isinstance(case, dict) else row


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
    by_split_source = {
        split: dict(Counter(str(row.get("alpha_source") or "unknown") for row in rows))
        for split, rows in rows_by_split.items()
    }
    by_split_shape = {
        split: dict(Counter(str(row["answerability_shape"]) for row in rows))
        for split, rows in rows_by_split.items()
    }
    return {
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "label_counts": dict(Counter(str(row["label"]) for row in all_rows)),
        "dataset_version_counts": dict(Counter(str(row["dataset_version"]) for row in all_rows)),
        "answerability_shape_counts": dict(Counter(str(row["answerability_shape"]) for row in all_rows)),
        "source_counts_by_split": by_split_source,
        "answerability_shape_counts_by_split": by_split_shape,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, default=Path("data/multitask_v8_2_retrieval_control"))
    parser.add_argument("--fitz-gov-repo", type=Path, default=Path("../fitz-gov"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/multitask_g4_alpha_v9_candidate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/encoder/modernbert_base_g4_alpha_v9_candidate.yaml"),
        help="Config path recorded in the output manifest.",
    )
    parser.add_argument(
        "--skip-candidates",
        action="store_true",
        help="Use only published V8.2 rows plus active merged local V9 rows.",
    )
    parser.add_argument("--candidate-start", type=int, default=100)
    parser.add_argument("--candidate-end", type=int, default=433)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    base_dir = args.base_data_dir.resolve()
    fitz_gov_repo = args.fitz_gov_repo.resolve()
    output_dir = args.output_dir.resolve()
    metadata = read_json(base_dir / "metadata.json")
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    seen_ids: set[str] = set()
    source_counts: Counter[str] = Counter()

    for split in ("train", "eval", "test"):
        for row in read_jsonl(base_dir / f"{split}.jsonl"):
            add_row(
                rows_by_split,
                collapse_flat_row(row, source_kind="fitz_gov_v8_2_published"),
                seen_ids,
                source_counts,
            )

    cases_path = fitz_gov_repo / "data" / "fitz-gov" / "cases.jsonl"
    active_v9 = 0
    for raw in read_jsonl(cases_path):
        case = unwrap_case(raw)
        if get_path(case, ("meta", "dataset_version")) != "v9":
            continue
        split = deterministic_split(str(case.get("id") or ""))
        flat = flatten_sdgp_case(
            case,
            split=split,
            metadata=metadata,
            source_kind="fitz_gov_v9_local_merged",
        )
        add_row(rows_by_split, flat, seen_ids, source_counts)
        active_v9 += 1

    candidate_rows = 0
    missing_batches: list[int] = []
    if not args.skip_candidates:
        candidate_dir = (
            fitz_gov_repo
            / "data"
            / "_workspaces"
            / "handoff"
            / "v9_answerability"
            / "subagent_outputs"
        )
        for batch_id in range(args.candidate_start, args.candidate_end + 1):
            path = candidate_dir / f"batch_{batch_id:03d}.jsonl"
            if not path.exists():
                missing_batches.append(batch_id)
                continue
            for raw in read_jsonl(path):
                case = unwrap_case(raw)
                split = deterministic_split(str(case.get("id") or ""))
                flat = flatten_sdgp_case(
                    case,
                    split=split,
                    metadata=metadata,
                    source_kind="fitz_gov_v9_structural_candidate_no_blind_qa",
                )
                add_row(rows_by_split, flat, seen_ids, source_counts)
                candidate_rows += 1

    if missing_batches:
        raise FileNotFoundError(f"missing candidate batches: {missing_batches[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = summarize(rows_by_split)
    out_metadata = {
        "source": {
            "base_data_dir": str(base_dir),
            "fitz_gov_repo": str(fitz_gov_repo),
            "fitz_gov_cases": str(cases_path),
            "candidate_batches": None
            if args.skip_candidates
            else f"{args.candidate_start}-{args.candidate_end}",
            "candidate_status": "skipped_active_vault_only"
            if args.skip_candidates
            else "structural_dry_run_clean_no_blind_label_qa",
        },
        **summary,
        "active_v9_rows_read": active_v9,
        "candidate_v9_rows_read": candidate_rows,
        "route2id": metadata["route2id"],
        "taxonomy_pattern2id": metadata["taxonomy_pattern2id"],
        "query_contract2id": metadata["query_contract2id"],
        "retrieval_action2id": metadata["retrieval_action2id"],
        "gap_type2id": metadata["gap_type2id"],
        "answerability_shape2id": COLLAPSED_ANSWERABILITY_LABEL2ID,
        "answerability_shape_collapse_map": DETAILED_TO_COLLAPSED,
        "retrieval_modality2id": metadata["retrieval_modality2id"],
        "require_query_contract": True,
        "require_retrieval_control": True,
        "scalar_fields": list(SCALAR_FIELDS),
    }
    (output_dir / "metadata.json").write_text(json.dumps(out_metadata, indent=2), encoding="utf-8")
    write_manifest(
        output_dir,
        args.config,
        seed=0,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=fitz_gov_repo,
        extra={
            "script": "prepare_g4_alpha_data.py",
            "candidate_batches": None
            if args.skip_candidates
            else f"{args.candidate_start}-{args.candidate_end}",
        },
        start_time=start,
    )

    print(f"output     : {output_dir}")
    print(f"splits     : {out_metadata['splits']}")
    print(f"sources    : {dict(source_counts)}")
    print(f"shapes     : {out_metadata['answerability_shape_counts']}")
    print(f"active v9  : {active_v9}")
    print(f"candidate v9: {candidate_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
