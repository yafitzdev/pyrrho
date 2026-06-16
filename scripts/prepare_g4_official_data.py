"""Prepare the official pyrrho-nano-g4 multitask dataset from fitz-gov V9.0.0.

This is the real g4 prep path. It consumes the public Hugging Face dataset
`yafitzdev/fitz-gov` at revision `v9.0.0`, verifies every row against the
published `v9/split_assignments.jsonl`, and writes the local train/eval/test
JSONL files expected by `scripts/train_multitask_encoder.py`.
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

from pyrrho.manifest import write_manifest
from prepare_g4_alpha_data import (
    COLLAPSED_ANSWERABILITY_LABEL2ID,
    DETAILED_TO_COLLAPSED,
    SCALAR_FIELDS,
    flatten_sdgp_case,
    read_json,
    summarize,
    write_jsonl,
)


DEFAULT_REPO_ID = "yafitzdev/fitz-gov"
DEFAULT_CONFIG_NAME = "v9"
DEFAULT_REVISION = "v9.0.0"
DEFAULT_OUTPUT_DIR = Path("data/multitask_g4_v9_official")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_g4_v9_official.yaml")
DEFAULT_EXPECTED_SPLITS = {"train": 32625, "eval": 4104, "test": 4026}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path("data/multitask_v8_2_retrieval_control"),
        help="Existing multitask data dir used only for stable label-id metadata.",
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
    metadata = read_json(base_dir / "metadata.json")

    assignments_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename="v9/split_assignments.jsonl",
            revision=args.revision,
        )
    )
    assignments = read_split_assignments(assignments_path)
    assignment_counts = dict(Counter(assignments.values()))

    dataset = load_dataset(args.repo_id, args.config_name, revision=args.revision)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    seen_ids: set[str] = set()

    for hf_split in ("train", "validation", "test"):
        if hf_split not in dataset:
            raise KeyError(f"HF dataset is missing split {hf_split!r}")
        local_split = local_split_name(hf_split)
        for row in dataset[hf_split]:
            case = dict(row)
            case_id = str(case.get("id") or "")
            assigned = assignments.get(case_id)
            if assigned != hf_split:
                raise ValueError(
                    f"split mismatch for {case_id!r}: dataset={hf_split}, assignment={assigned}"
                )
            if case_id in seen_ids:
                raise ValueError(f"duplicate dataset id across public splits: {case_id!r}")
            seen_ids.add(case_id)
            rows_by_split[local_split].append(
                flatten_sdgp_case(
                    case,
                    split=local_split,
                    metadata=metadata,
                    source_kind="fitz_gov_v9_0_0_public",
                )
            )

    missing_assignments = sorted(set(assignments) - seen_ids)
    if missing_assignments:
        raise ValueError(f"{len(missing_assignments)} assignments are missing from HF splits")

    validate_splits(rows_by_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    summary = summarize(rows_by_split)
    sha = dataset_sha(args.repo_id, args.revision)
    out_metadata = {
        "source": {
            "repo_id": args.repo_id,
            "config_name": args.config_name,
            "revision": args.revision,
            "resolved_commit": sha,
            "split_assignments": "v9/split_assignments.jsonl",
            "split_assignments_cache_path": str(assignments_path),
            "split_name_map": {"validation": "eval"},
        },
        **summary,
        "assignment_counts": assignment_counts,
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
        extra={
            "script": "prepare_g4_official_data.py",
            "repo_id": args.repo_id,
            "config_name": args.config_name,
            "revision": args.revision,
            "resolved_commit": sha,
            "split_assignments": "v9/split_assignments.jsonl",
        },
        start_time=start,
    )

    print(f"output     : {output_dir}")
    print(f"source     : {args.repo_id}/{args.config_name}@{args.revision}")
    print(f"commit     : {sha}")
    print(f"splits     : {out_metadata['splits']}")
    print(f"assignments: {assignment_counts}")
    print(f"versions   : {out_metadata['dataset_version_counts']}")
    print(f"shapes     : {out_metadata['answerability_shape_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
