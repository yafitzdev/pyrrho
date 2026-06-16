"""Prepare fitz-gov V8 for pyrrho-MoE multi-task training.

This is the data-audit/flattening scaffold for the Stage 0 route prototype and
later 4B upcycling runs. It preserves the published HF split contract and writes
JSONL records with governance label, semantic route, taxonomy pattern, scalar
targets, and context-level signals.

Run from project root:
    python scripts/prepare_moe_data.py --config configs/moe/pyrrho_moe_g3_alpha.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import yaml
from datasets import load_dataset

from pyrrho.data import (
    ANSWERABILITY_SHAPE_LABEL2ID,
    GAP_TYPE_LABEL2ID,
    LABEL2ID,
    QUERY_CONTRACT_LABEL2ID,
    RETRIEVAL_ACTION_LABEL2ID,
    RETRIEVAL_MODALITY_LABEL2ID,
    RETRIEVAL_OBLIGATION_LABEL2ID,
    build_encoder_text,
    build_query_contract_text,
)
from pyrrho.manifest import write_manifest
from pyrrho.moe import DEFAULT_SEMANTIC_EXPERT_GROUPS

DEFAULT_SCALAR_FIELDS: tuple[str, ...] = (
    "trustworthy",
    "disputed",
    "abstain",
    "confidence",
    "grounding",
    "conflict_density",
    "evidence_sufficiency",
    "domain_familiarity",
    "false_trustworthy_risk",
    "hallucination_pressure",
    "retrieval_retry_value",
    "human_escalation_score",
    "query_evidence_alignment",
    "answer_coverage",
    "evidence_bias_score",
    "evidence_failure_severity",
)

REQUIRED_PATHS: tuple[tuple[str, ...], ...] = (
    ("id",),
    ("input", "query"),
    ("input", "contexts"),
    ("governance", "classification"),
    ("routing", "expert_fired"),
    ("taxonomy", "pattern"),
    ("taxonomy", "cell_id"),
    ("meta", "difficulty"),
)


def get_path(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = row
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def missing_required(row: dict[str, Any]) -> list[str]:
    missing = []
    for path in REQUIRED_PATHS:
        value = get_path(row, path)
        if value is None or value == "" or value == []:
            missing.append(".".join(path))
    contexts = get_path(row, ("input", "contexts"))
    if isinstance(contexts, list):
        for i, ctx in enumerate(contexts):
            if not isinstance(ctx, dict) or not str(ctx.get("text") or "").strip():
                missing.append(f"input.contexts[{i}].text")
    return missing


def normalize_label(row: dict[str, Any]) -> str:
    label = str(get_path(row, ("governance", "classification")) or row.get("label") or "").upper()
    if label not in LABEL2ID:
        raise ValueError(f"case {row.get('id')!r} has invalid governance label {label!r}")
    return label


def normalize_query_contract(row: dict[str, Any], *, required: bool) -> str:
    routing = row.get("routing") or {}
    query_contract = routing.get("query_contract") if isinstance(routing, dict) else None
    kind = ""
    if isinstance(query_contract, dict):
        kind = str(query_contract.get("kind") or "")
    if kind in QUERY_CONTRACT_LABEL2ID:
        return kind
    if required:
        raise ValueError(f"case {row.get('id')!r} has invalid query_contract kind {kind!r}")
    return ""


def normalize_retrieval_control_label(
    row: dict[str, Any],
    field: str,
    label2id: dict[str, int],
    *,
    required: bool,
) -> str:
    routing = row.get("routing") or {}
    retrieval_control = routing.get("retrieval_control") if isinstance(routing, dict) else None
    block = retrieval_control.get(field) if isinstance(retrieval_control, dict) else None
    kind = str(block.get("kind") or "") if isinstance(block, dict) else ""
    if kind in label2id:
        return kind
    if required:
        raise ValueError(f"case {row.get('id')!r} has invalid retrieval_control.{field} kind {kind!r}")
    return ""


def retrieval_control_severity(row: dict[str, Any]) -> float | None:
    routing = row.get("routing") or {}
    retrieval_control = routing.get("retrieval_control") if isinstance(routing, dict) else None
    severity = (
        retrieval_control.get("evidence_failure_severity")
        if isinstance(retrieval_control, dict)
        else None
    )
    score = severity.get("score") if isinstance(severity, dict) else None
    return float(score) if isinstance(score, int | float) else None


def context_features(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ctx in contexts:
        out.append(
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
        )
    return out


def flatten_case(
    row: dict[str, Any],
    *,
    split: str,
    route2id: dict[str, int],
    taxonomy2id: dict[str, int],
    scalar_fields: tuple[str, ...],
    require_query_contract: bool,
    require_retrieval_control: bool,
) -> dict[str, Any]:
    label = normalize_label(row)
    input_block = row.get("input") or {}
    contexts = input_block.get("contexts") or []
    query = str(input_block.get("query") or "").strip()
    context_texts = [str(ctx.get("text") or "").strip() for ctx in contexts]
    governance = row.get("governance") or {}
    routing = row.get("routing") or {}
    taxonomy = row.get("taxonomy") or {}
    meta = row.get("meta") or {}
    query_contract = normalize_query_contract(row, required=require_query_contract)
    retrieval_action = normalize_retrieval_control_label(
        row,
        "retrieval_action",
        RETRIEVAL_ACTION_LABEL2ID,
        required=require_retrieval_control,
    )
    gap_type = normalize_retrieval_control_label(
        row,
        "gap_type",
        GAP_TYPE_LABEL2ID,
        required=require_retrieval_control,
    )
    answerability_shape = normalize_retrieval_control_label(
        row,
        "answerability_shape",
        ANSWERABILITY_SHAPE_LABEL2ID,
        required=require_retrieval_control,
    )
    retrieval_modality = normalize_retrieval_control_label(
        row,
        "preferred_retrieval_modality",
        RETRIEVAL_MODALITY_LABEL2ID,
        required=require_retrieval_control,
    )
    retrieval_obligation = normalize_retrieval_control_label(
        row,
        "retrieval_obligation",
        RETRIEVAL_OBLIGATION_LABEL2ID,
        required=False,
    )

    route = str(routing.get("expert_fired") or "")
    pattern = str(taxonomy.get("pattern") or "")
    scalar_targets = {}
    for field in scalar_fields:
        value = (
            retrieval_control_severity(row)
            if field == "evidence_failure_severity"
            else governance.get(field)
        )
        if isinstance(value, int | float):
            scalar_targets[field] = value
    boundary = governance.get("boundary_proximity") or {}

    return {
        "id": row.get("id", ""),
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
        "label_id": LABEL2ID[label],
        "route": route,
        "route_id": route2id[route],
        "query_contract": query_contract,
        "query_contract_id": QUERY_CONTRACT_LABEL2ID.get(query_contract, -1),
        "retrieval_action": retrieval_action,
        "retrieval_action_id": RETRIEVAL_ACTION_LABEL2ID.get(retrieval_action, -1),
        "gap_type": gap_type,
        "gap_type_id": GAP_TYPE_LABEL2ID.get(gap_type, -1),
        "answerability_shape": answerability_shape,
        "answerability_shape_id": ANSWERABILITY_SHAPE_LABEL2ID.get(answerability_shape, -1),
        "retrieval_modality": retrieval_modality,
        "retrieval_modality_id": RETRIEVAL_MODALITY_LABEL2ID.get(retrieval_modality, -1),
        "retrieval_obligation": retrieval_obligation,
        "retrieval_obligation_id": RETRIEVAL_OBLIGATION_LABEL2ID.get(retrieval_obligation, -1),
        "secondary_expert": routing.get("secondary_expert"),
        "routing_confidence": routing.get("routing_confidence"),
        "taxonomy_pattern": pattern,
        "taxonomy_pattern_id": taxonomy2id[pattern],
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
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g3_alpha.yaml"),
        help="MoE YAML config (default: configs/moe/pyrrho_moe_g3_alpha.yaml)",
    )
    p.add_argument("--hf", type=str, default=None, help="Override data.fitz_gov_hf")
    p.add_argument("--hf-config", type=str, default=None, help="Override data.fitz_gov_config")
    p.add_argument("--hf-revision", type=str, default=None, help="Override data.fitz_gov_revision")
    p.add_argument("--output", type=Path, default=None, help="Override data.moe_output_dir")
    p.add_argument("--strict", action="store_true", help="Fail on any required-field audit error")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    data_cfg = cfg.get("data", {})
    arch_cfg = cfg.get("architecture", {})

    repo_id = args.hf or data_cfg.get("fitz_gov_hf", "yafitzdev/fitz-gov")
    hf_config = args.hf_config or data_cfg.get("fitz_gov_config", "v8")
    revision = args.hf_revision or data_cfg.get("fitz_gov_revision", "v8.0.1")
    output_dir = (args.output or Path(data_cfg.get("moe_output_dir", "data/moe_v8"))).resolve()
    semantic_groups = tuple(arch_cfg.get("semantic_expert_groups") or DEFAULT_SEMANTIC_EXPERT_GROUPS)
    route2id = {name: i for i, name in enumerate(semantic_groups)}
    scalar_fields = tuple(data_cfg.get("scalar_fields") or DEFAULT_SCALAR_FIELDS)
    require_query_contract = bool(data_cfg.get("require_query_contract", False))
    require_retrieval_control = bool(data_cfg.get("require_retrieval_control", False))

    print(f"source          : HuggingFace {repo_id}")
    print(f"hf config       : {hf_config}")
    print(f"hf revision     : {revision}")
    print(f"output dir      : {output_dir}")

    ds = load_dataset(repo_id, hf_config, revision=revision)
    split_name_map = {"validation": "eval"}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    audit_missing: Counter[str] = Counter()
    audit_examples: dict[str, list[str]] = defaultdict(list)
    route_counts: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    scalar_missing: Counter[str] = Counter()

    all_rows: list[tuple[str, dict[str, Any]]] = []
    for raw_split in ds.keys():
        split = split_name_map.get(raw_split, raw_split)
        rows = [dict(row) for row in ds[raw_split]]
        split_rows[split] = rows
        for row in rows:
            all_rows.append((split, row))

    for _, row in all_rows:
        case_id = str(row.get("id") or "")
        for missing in missing_required(row):
            audit_missing[missing] += 1
            if len(audit_examples[missing]) < 5:
                audit_examples[missing].append(case_id)
        try:
            label_counts[normalize_label(row)] += 1
        except ValueError:
            audit_missing["governance.classification.invalid"] += 1
        try:
            normalize_query_contract(row, required=require_query_contract)
        except ValueError:
            audit_missing["routing.query_contract.invalid"] += 1
            if len(audit_examples["routing.query_contract.invalid"]) < 5:
                audit_examples["routing.query_contract.invalid"].append(case_id)
        for field, label2id in (
            ("retrieval_action", RETRIEVAL_ACTION_LABEL2ID),
            ("gap_type", GAP_TYPE_LABEL2ID),
            ("answerability_shape", ANSWERABILITY_SHAPE_LABEL2ID),
            ("preferred_retrieval_modality", RETRIEVAL_MODALITY_LABEL2ID),
            ("retrieval_obligation", RETRIEVAL_OBLIGATION_LABEL2ID),
        ):
            try:
                normalize_retrieval_control_label(
                    row,
                    field,
                    label2id,
                    required=require_retrieval_control,
                )
            except ValueError:
                key = f"routing.retrieval_control.{field}.invalid"
                audit_missing[key] += 1
                if len(audit_examples[key]) < 5:
                    audit_examples[key].append(case_id)
        route = str(get_path(row, ("routing", "expert_fired")) or "")
        pattern = str(get_path(row, ("taxonomy", "pattern")) or "")
        route_counts[route] += 1
        taxonomy_counts[pattern] += 1
        governance = row.get("governance") or {}
        for field in scalar_fields:
            value = (
                retrieval_control_severity(row)
                if field == "evidence_failure_severity"
                else governance.get(field)
            )
            if not isinstance(value, int | float):
                scalar_missing[field] += 1

    unknown_routes = sorted(r for r in route_counts if r and r not in route2id)
    if unknown_routes:
        audit_missing["routing.expert_fired.unknown"] += sum(route_counts[r] for r in unknown_routes)
        audit_examples["routing.expert_fired.unknown"].extend(unknown_routes[:5])

    if audit_missing and args.strict:
        print("\nAudit failed:")
        for key, count in audit_missing.most_common():
            print(f"  {key:40s}: {count} examples={audit_examples.get(key, [])}")
        return 1

    taxonomy_patterns = sorted(p for p in taxonomy_counts if p)
    taxonomy2id = {name: i for i, name in enumerate(taxonomy_patterns)}

    flat_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, rows in split_rows.items():
        flat_by_split[split] = [
            flatten_case(
                row,
                split=split,
                route2id=route2id,
                taxonomy2id=taxonomy2id,
                scalar_fields=scalar_fields,
                require_query_contract=require_query_contract,
                require_retrieval_control=require_retrieval_control,
            )
            for row in rows
            if not missing_required(row)
            and str(get_path(row, ("routing", "expert_fired")) or "") in route2id
            and str(get_path(row, ("taxonomy", "pattern")) or "") in taxonomy2id
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in flat_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = {
        "source": {"repo_id": repo_id, "config": hf_config, "revision": revision},
        "splits": {split: len(rows) for split, rows in flat_by_split.items()},
        "audit": {
            "missing_required": dict(audit_missing),
            "missing_required_examples": dict(audit_examples),
            "unknown_routes": unknown_routes,
            "scalar_missing": dict(scalar_missing),
        },
        "label_counts": dict(label_counts),
        "route2id": route2id,
        "route_counts": dict(route_counts),
        "taxonomy_pattern2id": taxonomy2id,
        "taxonomy_pattern_counts": dict(taxonomy_counts),
        "query_contract2id": QUERY_CONTRACT_LABEL2ID,
        "require_query_contract": require_query_contract,
        "retrieval_action2id": RETRIEVAL_ACTION_LABEL2ID,
        "gap_type2id": GAP_TYPE_LABEL2ID,
        "answerability_shape2id": ANSWERABILITY_SHAPE_LABEL2ID,
        "retrieval_modality2id": RETRIEVAL_MODALITY_LABEL2ID,
        "retrieval_obligation2id": RETRIEVAL_OBLIGATION_LABEL2ID,
        "require_retrieval_control": require_retrieval_control,
        "scalar_fields": list(scalar_fields),
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(
        output_dir,
        args.config,
        seed=0,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=(Path.cwd().parent / "fitz-gov"),
        extra={"script": "prepare_moe_data.py", **summary["source"]},
        start_time=start,
    )

    print("\nWrote MoE splits:")
    for split, rows in flat_by_split.items():
        print(f"  {split:10s}: {len(rows)}")
    print("\nTop routes:")
    for route, count in route_counts.most_common():
        print(f"  {route:24s}: {count}")
    print(f"\nTaxonomy patterns: {len(taxonomy2id)}")
    print(f"Audit missing required fields: {sum(audit_missing.values())}")
    if audit_missing:
        for key, count in audit_missing.most_common(10):
            print(f"  {key:40s}: {count}")
    print(f"\nMetadata: {output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
