"""Prepare the local pyrrho-nano-g5 multitask release-candidate dataset.

This keeps official fitz-gov V9.0.0 as the stable public base and appends the
local V10 target-10 retrieval-planning candidate block:

* V10 5/5/5 seed block from 2026-06-12
* V10 10/10/10 continuation block from 2026-06-13

V9 rows are masked for the retrieval-obligation head. V10 rows carry
`routing.retrieval_control.retrieval_obligation.kind` and train that head.
The V10 block is structurally clean, duplicate-audit clean, and blind-label
clean after the 2026-06-13 triage repair pass.
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
from prepare_g5_alpha_data import (
    add_retrieval_obligation,
    case_with_normalized_retrieval_control,
    mask_official_row,
    summarize,
)


DEFAULT_CANDIDATE_DIRS = (
    Path("../fitz-gov")
    / "data"
    / "_workspaces"
    / "handoff"
    / "v10_cogeneration_5x5_20260612"
    / "subagent_outputs",
    Path("../fitz-gov")
    / "data"
    / "_workspaces"
    / "handoff"
    / "v10_cogeneration_10x_after5x5_20260613"
    / "subagent_outputs",
)
DEFAULT_OUTPUT_DIR = Path("data/multitask_g5_v10_target10_candidate")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_g5_v10_target10_candidate.yaml")


def source_kind_for_candidate_dir(candidate_dir: Path) -> str:
    name = candidate_dir.parent.name
    if "10x_after5x5" in name:
        return "fitz_gov_v10_target10_continuation_candidate_blind_qa_clean"
    if "5x5" in name:
        return "fitz_gov_v10_5x5_candidate_blind_qa_clean"
    return "fitz_gov_v10_candidate_blind_qa_clean"


def add_row_strict(
    rows_by_split: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
    seen_ids: set[str],
    source_counts: Counter[str],
) -> None:
    case_id = str(row["id"])
    if case_id in seen_ids:
        raise ValueError(f"duplicate row id while preparing g5 data: {case_id!r}")
    seen_ids.add(case_id)
    rows_by_split[row["split"]].append(row)
    source_counts[str(row.get("alpha_source") or "unknown")] += 1


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
        action="append",
        default=None,
        help=(
            "V10 candidate JSONL output directory. May be passed multiple times. "
            "Defaults to the 5/5/5 seed block and the target-10 continuation block."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config path recorded in the output manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    base_dir = args.base_data_dir.resolve()
    fitz_gov_repo = args.fitz_gov_repo.resolve()
    candidate_dirs = [path.resolve() for path in (args.candidate_dir or DEFAULT_CANDIDATE_DIRS)]
    output_dir = args.output_dir.resolve()
    metadata = read_json(base_dir / "metadata.json")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    seen_ids: set[str] = set()
    source_counts: Counter[str] = Counter()

    for split in ("train", "eval", "test"):
        for row in read_jsonl(base_dir / f"{split}.jsonl"):
            add_row_strict(rows_by_split, mask_official_row(row), seen_ids, source_counts)

    candidate_rows_by_dir: dict[str, int] = {}
    filled_field_counts: Counter[str] = Counter()
    raw_shape_counts: Counter[str] = Counter()
    collapsed_shape_counts: Counter[str] = Counter()
    total_candidate_rows = 0

    for candidate_dir in candidate_dirs:
        candidate_files = sorted(candidate_dir.glob("batch_*.jsonl"))
        if not candidate_files:
            raise FileNotFoundError(f"no candidate files found in {candidate_dir}")

        source_kind = source_kind_for_candidate_dir(candidate_dir)
        candidate_rows_by_dir[str(candidate_dir)] = 0
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
                    source_kind=source_kind,
                )
                flat["answerability_shape_raw_v10"] = raw_shape
                flat["answerability_shape_collapsed_v10"] = collapsed_shape
                flat["v10_filled_retrieval_control_fields"] = filled_fields
                flat["v10_candidate_dir"] = str(candidate_dir)
                add_row_strict(
                    rows_by_split,
                    add_retrieval_obligation(flat, case),
                    seen_ids,
                    source_counts,
                )
                total_candidate_rows += 1
                candidate_rows_by_dir[str(candidate_dir)] += 1
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
            "base_fitz_gov_version": "v9.0.0",
            "v10_candidate_dirs": [str(path) for path in candidate_dirs],
            "v10_candidate_status": (
                "target10_structural_validation_clean_quality_audit_clean_blind_label_clean"
            ),
            "v10_blind_label_qa": {
                "score_dir": (
                    "data/_workspaces/qa/v10_target10_candidate_20260613/"
                    "score_after_blind_triage_repair_20260613"
                ),
                "agreement": "12748/12748",
                "triage": 0,
                "missing_invalid_error": "0/0/0",
                "repair_summary": (
                    "data/_workspaces/qa/v10_target10_candidate_20260613/"
                    "repair_blind_triage_summary.json"
                ),
            },
            "v10_target_profile": "10/10/10",
        },
        **summary,
        "candidate_v10_rows_read": total_candidate_rows,
        "candidate_v10_rows_by_dir": candidate_rows_by_dir,
        "v10_raw_answerability_shape_counts": dict(raw_shape_counts),
        "v10_collapsed_answerability_shape_counts": dict(collapsed_shape_counts),
        "v10_filled_retrieval_control_field_counts": dict(filled_field_counts),
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
            "script": "prepare_g5_data.py",
            "base_data_dir": str(base_dir),
            "candidate_dirs": [str(path) for path in candidate_dirs],
            "candidate_rows": total_candidate_rows,
            "v10_target_profile": "10/10/10",
        },
        start_time=start,
    )

    print(f"output       : {output_dir}")
    print(f"splits       : {out_metadata['splits']}")
    print(f"sources      : {dict(source_counts)}")
    print(f"v10 rows     : {total_candidate_rows}")
    print(f"v10 dirs     : {candidate_rows_by_dir}")
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
