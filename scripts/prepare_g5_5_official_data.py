"""Prepare the official pyrrho-nano-g5.5 multitask dataset from fitz-gov V11.0.0.

This consumes the public Hugging Face dataset `yafitzdev/fitz-gov` at revision
`v11.0.0`, verifies every row against the published `v11/split_assignments.jsonl`,
and writes the local train/eval/test JSONL files expected by
`scripts/train_multitask_encoder.py`.
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

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from pyrrho.data import RETRIEVAL_OBLIGATION_LABEL2ID
from pyrrho.manifest import write_manifest
from prepare_g4_alpha_data import (
    COLLAPSED_ANSWERABILITY_LABEL2ID,
    DETAILED_TO_COLLAPSED,
    SCALAR_FIELDS,
    flatten_sdgp_case,
    get_path,
    read_json,
    retrieval_kind,
    summarize,
    write_jsonl,
)


DEFAULT_REPO_ID = "yafitzdev/fitz-gov"
DEFAULT_CONFIG_NAME = "v11"
DEFAULT_REVISION = "v11.0.0"
DEFAULT_OUTPUT_DIR = Path("data/multitask_g5_5_v11_official")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_g5_5_v11_official.yaml")
DEFAULT_EXPECTED_SPLITS = {"train": 48800, "eval": 6028, "test": 6055}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path("data/multitask_g5_1_v10_repaired"),
        help="Existing multitask data dir used for stable label-id metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config path recorded in the output manifest.",
    )
    return parser.parse_args()


def read_split_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            case_id = str(row.get("case_id") or "")
            split = str(row.get("split") or "")
            if not case_id or split not in {"train", "validation", "test"}:
                raise ValueError(f"bad split assignment at line {line_no}: {row!r}")
            if case_id in assignments:
                raise ValueError(f"duplicate split assignment for {case_id!r}")
            assignments[case_id] = split
    return assignments


def local_split_name(hf_split: str) -> str:
    if hf_split == "validation":
        return "eval"
    if hf_split in {"train", "test"}:
        return hf_split
    raise ValueError(f"unexpected HF split: {hf_split!r}")


def dataset_sha(repo_id: str, revision: str) -> str | None:
    try:
        return HfApi().dataset_info(repo_id, revision=revision).sha
    except Exception:
        return None


def case_taxonomy_pattern(case: dict[str, Any]) -> str:
    raw = get_path(case, ("taxonomy", "pattern"))
    return str(raw or "")


def extend_taxonomy_metadata(
    metadata: dict[str, Any],
    cases_by_hf_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out = dict(metadata)
    mapping = {str(k): int(v) for k, v in dict(metadata["taxonomy_pattern2id"]).items()}
    additions: set[str] = set()
    for cases in cases_by_hf_split.values():
        for case in cases:
            pattern = case_taxonomy_pattern(case)
            if pattern and pattern not in mapping and pattern != "missing_specific_fact":
                additions.add(pattern)
    for pattern in sorted(additions):
        mapping[pattern] = len(mapping)
    out["taxonomy_pattern2id"] = mapping
    out["taxonomy_pattern_added_for_v11"] = sorted(additions)
    return out


def add_optional_retrieval_obligation(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = retrieval_kind(case, "retrieval_obligation")
    except ValueError:
        row["retrieval_obligation"] = None
        row["retrieval_obligation_raw"] = None
        row["retrieval_obligation_id"] = -1
        return row
    if raw == "none":
        row["retrieval_obligation"] = None
        row["retrieval_obligation_raw"] = raw
        row["retrieval_obligation_id"] = -1
        return row
    if raw not in RETRIEVAL_OBLIGATION_LABEL2ID:
        raise ValueError(f"{case.get('id')}: unknown retrieval_obligation {raw!r}")
    row["retrieval_obligation"] = raw
    row["retrieval_obligation_raw"] = raw
    row["retrieval_obligation_id"] = RETRIEVAL_OBLIGATION_LABEL2ID[raw]
    return row


def source_kind(case: dict[str, Any]) -> str:
    version = str(get_path(case, ("meta", "dataset_version")) or "unknown")
    return f"fitz_gov_{version}_from_v11_0_0_public"


def validate_splits(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    actual = {split: len(rows) for split, rows in rows_by_split.items()}
    if actual != DEFAULT_EXPECTED_SPLITS:
        raise ValueError(f"unexpected split sizes: actual={actual}, expected={DEFAULT_EXPECTED_SPLITS}")

    ids: list[str] = []
    for rows in rows_by_split.values():
        ids.extend(str(row["id"]) for row in rows)
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise ValueError(f"prepared data has {duplicate_count} duplicate ids")


def main() -> int:
    args = parse_args()
    start = time.time()
    base_dir = args.base_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    base_metadata = read_json(base_dir / "metadata.json")

    assignments_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename="v11/split_assignments.jsonl",
            revision=args.revision,
        )
    )
    assignments = read_split_assignments(assignments_path)
    assignment_counts = dict(Counter(assignments.values()))

    dataset = load_dataset(args.repo_id, args.config_name, revision=args.revision)
    cases_by_hf_split: dict[str, list[dict[str, Any]]] = {}
    for hf_split in ("train", "validation", "test"):
        if hf_split not in dataset:
            raise KeyError(f"HF dataset is missing split {hf_split!r}")
        cases_by_hf_split[hf_split] = [dict(row) for row in dataset[hf_split]]

    metadata = extend_taxonomy_metadata(base_metadata, cases_by_hf_split)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    seen_ids: set[str] = set()
    obligation_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for hf_split, cases in cases_by_hf_split.items():
        local_split = local_split_name(hf_split)
        for case in cases:
            case_id = str(case.get("id") or "")
            assigned = assignments.get(case_id)
            if assigned != hf_split:
                raise ValueError(
                    f"split mismatch for {case_id!r}: dataset={hf_split}, assignment={assigned}"
                )
            if case_id in seen_ids:
                raise ValueError(f"duplicate dataset id across public splits: {case_id!r}")
            seen_ids.add(case_id)
            flat = flatten_sdgp_case(
                case,
                split=local_split,
                metadata=metadata,
                source_kind=source_kind(case),
            )
            flat = add_optional_retrieval_obligation(flat, case)
            if int(flat.get("retrieval_obligation_id", -1)) >= 0:
                obligation_counts[str(flat["retrieval_obligation"])] += 1
            source_counts[str(flat.get("alpha_source") or "unknown")] += 1
            rows_by_split[local_split].append(flat)

    missing_assignments = sorted(set(assignments) - seen_ids)
    if missing_assignments:
        raise ValueError(f"{len(missing_assignments)} assignments are missing from HF splits")

    validate_splits(rows_by_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = summarize(rows_by_split)
    labeled_obligation_rows = sum(
        1
        for rows in rows_by_split.values()
        for row in rows
        if int(row.get("retrieval_obligation_id", -1)) >= 0
    )
    sha = dataset_sha(args.repo_id, args.revision)
    out_metadata = {
        "source": {
            "repo_id": args.repo_id,
            "config_name": args.config_name,
            "revision": args.revision,
            "resolved_commit": sha,
            "split_assignments": "v11/split_assignments.jsonl",
            "split_assignments_cache_path": str(assignments_path),
            "split_name_map": {"validation": "eval"},
            "base_metadata_dir": str(base_dir),
        },
        **summary,
        "assignment_counts": assignment_counts,
        "source_counts": dict(source_counts),
        "taxonomy_pattern_added_for_v11": metadata["taxonomy_pattern_added_for_v11"],
        "route2id": metadata["route2id"],
        "taxonomy_pattern2id": metadata["taxonomy_pattern2id"],
        "query_contract2id": metadata["query_contract2id"],
        "retrieval_action2id": metadata["retrieval_action2id"],
        "gap_type2id": metadata["gap_type2id"],
        "answerability_shape2id": COLLAPSED_ANSWERABILITY_LABEL2ID,
        "answerability_shape_collapse_map": DETAILED_TO_COLLAPSED,
        "retrieval_modality2id": metadata["retrieval_modality2id"],
        "retrieval_obligation2id": RETRIEVAL_OBLIGATION_LABEL2ID,
        "retrieval_obligation_counts": dict(obligation_counts),
        "retrieval_obligation_labeled_rows": labeled_obligation_rows,
        "retrieval_obligation_masked_rows": sum(
            1
            for rows in rows_by_split.values()
            for row in rows
            if int(row.get("retrieval_obligation_id", -1)) < 0
        ),
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
        fitz_gov_repo=Path.cwd().parent / "fitz-gov",
        extra={
            "script": "prepare_g5_5_official_data.py",
            "repo_id": args.repo_id,
            "config_name": args.config_name,
            "revision": args.revision,
            "resolved_commit": sha,
            "split_assignments": "v11/split_assignments.jsonl",
            "retrieval_obligation_labeled_rows": labeled_obligation_rows,
        },
        start_time=start,
    )

    print(f"output       : {output_dir}")
    print(f"source       : {args.repo_id}/{args.config_name}@{args.revision}")
    print(f"commit       : {sha}")
    print(f"splits       : {out_metadata['splits']}")
    print(f"assignments  : {assignment_counts}")
    print(f"versions     : {out_metadata['dataset_version_counts']}")
    print(f"sources      : {dict(source_counts)}")
    print(f"taxonomy +v11: {len(out_metadata['taxonomy_pattern_added_for_v11'])}")
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
