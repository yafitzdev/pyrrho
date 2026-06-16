"""Prepare the local pyrrho-nano-g5.8 V12.2 boundary-repair candidate dataset.

This keeps the flattened rows and split assignments from
`data/multitask_g5_7_v12_1_failure_repair_candidate`, then appends the local
2,400-row V12.2/g5.8 boundary-repair candidate block. The candidate block is
structurally clean and duplicate clean, but it is not a published fitz-gov
release and has not gone through blind-label QA.
"""

from __future__ import annotations

import argparse
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
    deterministic_split,
    flatten_sdgp_case,
    get_path,
    read_json,
    read_jsonl,
    unwrap_case,
    write_jsonl,
)
from prepare_g5_alpha_data import case_with_normalized_retrieval_control, summarize


DEFAULT_CANDIDATE_DIR = (
    Path("../fitz-gov")
    / "data"
    / "_workspaces"
    / "handoff"
    / "v12_2_g5_8_boundary_repair_20260616"
    / "subagent_outputs"
)
DEFAULT_OUTPUT_DIR = Path("data/multitask_g5_8_v12_2_boundary_repair_candidate")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_g5_8_v12_2_boundary_repair_candidate.yaml")
SOURCE_KIND = "fitz_gov_v12_2_g5_8_boundary_repair_candidate_structural_clean_duplicate_clean_no_blind_qa"
CANDIDATE_STATUS = "structural_validation_clean_duplicate_clean_boundary_repair_no_blind_qa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path("data/multitask_g5_7_v12_1_failure_repair_candidate"),
        help="Flattened g5.7 multitask data dir.",
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
        help="V12.2/g5.8 boundary-repair candidate JSONL output directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config path recorded in the output manifest.",
    )
    return parser.parse_args()


def add_row_strict(
    rows_by_split: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
    seen_ids: set[str],
    source_counts: Counter[str],
) -> None:
    case_id = str(row["id"])
    if case_id in seen_ids:
        raise ValueError(f"duplicate row id while preparing g5.8 data: {case_id!r}")
    seen_ids.add(case_id)
    rows_by_split[row["split"]].append(row)
    source_counts[str(row.get("alpha_source") or "unknown")] += 1


def add_optional_retrieval_obligation(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    raw = str(get_path(case, ("routing", "retrieval_control", "retrieval_obligation", "kind")) or "")
    if not raw or raw == "none":
        row["retrieval_obligation"] = None
        row["retrieval_obligation_raw"] = raw or None
        row["retrieval_obligation_id"] = -1
        return row
    if raw not in RETRIEVAL_OBLIGATION_LABEL2ID:
        raise ValueError(f"{case.get('id')}: unknown retrieval_obligation {raw!r}")
    row["retrieval_obligation"] = raw
    row["retrieval_obligation_raw"] = raw
    row["retrieval_obligation_id"] = RETRIEVAL_OBLIGATION_LABEL2ID[raw]
    return row


def candidate_files(candidate_dir: Path) -> list[Path]:
    files = sorted(candidate_dir.glob("batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no candidate files found in {candidate_dir}")
    return files


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
            add_row_strict(rows_by_split, row, seen_ids, source_counts)

    candidate_rows_by_split: Counter[str] = Counter()
    candidate_rows_by_file: dict[str, int] = {}
    filled_field_counts: Counter[str] = Counter()
    raw_shape_counts: Counter[str] = Counter()
    collapsed_shape_counts: Counter[str] = Counter()
    candidate_rows = 0

    for path in candidate_files(candidate_dir):
        candidate_rows_by_file[str(path)] = 0
        for raw in read_jsonl(path):
            case = unwrap_case(raw)
            if get_path(case, ("meta", "dataset_version")) != "v12_2_candidate":
                raise ValueError(f"{case.get('id')}: expected meta.dataset_version='v12_2_candidate'")

            normalized_case, raw_shape, collapsed_shape, filled_fields = (
                case_with_normalized_retrieval_control(case)
            )
            split = deterministic_split(str(case.get("id") or ""))
            flat = flatten_sdgp_case(
                normalized_case,
                split=split,
                metadata=metadata,
                source_kind=SOURCE_KIND,
            )
            flat["answerability_shape_raw_v12_2"] = raw_shape
            flat["answerability_shape_collapsed_v12_2"] = collapsed_shape
            flat["v12_2_filled_retrieval_control_fields"] = filled_fields
            flat["v12_2_candidate_dir"] = str(candidate_dir)
            flat["v12_2_candidate_status"] = CANDIDATE_STATUS
            add_row_strict(
                rows_by_split,
                add_optional_retrieval_obligation(flat, case),
                seen_ids,
                source_counts,
            )
            candidate_rows += 1
            candidate_rows_by_file[str(path)] += 1
            candidate_rows_by_split[split] += 1
            raw_shape_counts[raw_shape or "<missing>"] += 1
            collapsed_shape_counts[collapsed_shape] += 1
            filled_field_counts.update(filled_fields)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = summarize(rows_by_split)
    out_metadata: dict[str, Any] = {
        "source": {
            "base_data_dir": str(base_dir),
            "fitz_gov_repo": str(fitz_gov_repo),
            "base_fitz_gov_version": "v11.0.0_plus_v12_g5_6_candidate_plus_v12_1_candidate",
            "candidate_dir": str(candidate_dir),
            "candidate_status": CANDIDATE_STATUS,
            "candidate_plan": (
                "data/_workspaces/reports/"
                "v12_2_g5_8_boundary_repair_20260616_plan.json"
            ),
            "candidate_validation_report": (
                "data/_workspaces/handoff/v12_2_g5_8_boundary_repair_20260616/"
                "validation_post_duplicate_repair.json"
            ),
            "candidate_quality_report": (
                "data/_workspaces/handoff/v12_2_g5_8_boundary_repair_20260616/"
                "quality_audit_post_duplicate_repair.json"
            ),
            "target_profile": "boundary_repair",
            "blind_label_qa": "not_run",
        },
        **summary,
        "candidate_v12_2_rows_read": candidate_rows,
        "candidate_v12_2_rows_by_split": dict(candidate_rows_by_split),
        "candidate_v12_2_rows_by_file": candidate_rows_by_file,
        "v12_2_raw_answerability_shape_counts": dict(raw_shape_counts),
        "v12_2_collapsed_answerability_shape_counts": dict(collapsed_shape_counts),
        "v12_2_filled_retrieval_control_field_counts": dict(filled_field_counts),
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
            "script": "prepare_g5_8_candidate_data.py",
            "base_data_dir": str(base_dir),
            "candidate_dir": str(candidate_dir),
            "candidate_rows": candidate_rows,
            "target_profile": "boundary_repair",
            "blind_label_qa": "not_run",
        },
        start_time=start,
    )

    print(f"output       : {output_dir}")
    print(f"splits       : {out_metadata['splits']}")
    print(f"sources      : {dict(source_counts)}")
    print(f"v12.2 rows   : {candidate_rows}")
    print(f"v12.2 split  : {dict(candidate_rows_by_split)}")
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
