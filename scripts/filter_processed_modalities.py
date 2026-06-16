"""Filter a prepared pyrrho DatasetDict by modality.

This is for local modality controls where the source split contract already
exists and only a subset of modalities should be trained/evaluated. It preserves
the original split membership for retained rows and writes a normal processed
data directory with JSONL files plus ``hf_dataset/``.

Example:
    python scripts/filter_processed_modalities.py \
      --input data/processed_v8_plus_structured_code_retry_patch_candidate \
      --output data/processed_v8_plus_code_retry_patch_candidate \
      --keep-modality unstructured --keep-modality code
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from pyrrho.data import ID2LABEL, load_processed
from pyrrho.manifest import write_manifest

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


SPLIT_FILE_NAMES = {
    "train": "train.jsonl",
    "eval": "eval.jsonl",
    "test": "test.jsonl",
    "tier0_sanity": "tier0_sanity.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Processed source dir containing hf_dataset/.")
    parser.add_argument("--output", type=Path, required=True, help="Processed output dir to write.")
    parser.add_argument(
        "--keep-modality",
        action="append",
        required=True,
        choices=("unstructured", "structured", "code"),
        help="Modality to keep. Repeat for multiple modalities.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output dir if it already exists.")
    parser.add_argument("--seed", type=int, default=42, help="Manifest seed for this deterministic filter run.")
    return parser.parse_args()


def safe_prepare_output(path: Path, overwrite: bool) -> None:
    path = path.resolve()
    if not path.exists():
        path.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    cwd = Path.cwd().resolve()
    if path == cwd or cwd not in path.parents:
        raise ValueError(f"Refusing to overwrite path outside workspace: {path}")
    shutil.rmtree(path)
    path.mkdir(parents=True)


def write_jsonl(path: Path, dataset: Dataset) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in dataset:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def counter(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize(dataset: Dataset) -> dict[str, Any]:
    labels = list(dataset["label"]) if "label" in dataset.column_names else []
    label_ids = list(dataset["label_id"]) if "label_id" in dataset.column_names else []
    if not labels and label_ids:
        labels = [ID2LABEL[int(label_id)] for label_id in label_ids]
    return {
        "rows": len(dataset),
        "labels": counter(labels),
        "modalities": counter(list(dataset["modality"])) if "modality" in dataset.column_names else {},
        "difficulties": counter(list(dataset["difficulty"])) if "difficulty" in dataset.column_names else {},
        "experts": counter(list(dataset["expert"])) if "expert" in dataset.column_names else {},
        "taxonomy_patterns": counter(list(dataset["taxonomy_pattern"]))
        if "taxonomy_pattern" in dataset.column_names
        else {},
    }


def main() -> int:
    started = time.time()
    args = parse_args()
    keep = sorted(set(args.keep_modality))

    if not args.input.exists():
        raise FileNotFoundError(f"Input processed dir not found: {args.input}")
    safe_prepare_output(args.output, args.overwrite)

    source = load_processed(args.input)
    filtered_splits: dict[str, Dataset] = {}
    for split_name, split in source.items():
        if "modality" not in split.column_names:
            raise ValueError(f"Split {split_name!r} has no modality column")
        filtered = split.filter(lambda row: row["modality"] in keep, desc=f"filter {split_name}")
        filtered_splits[split_name] = filtered
        file_name = SPLIT_FILE_NAMES.get(split_name, f"{split_name}.jsonl")
        write_jsonl(args.output / file_name, filtered)

    ds = DatasetDict(filtered_splits)
    hf_dir = args.output / "hf_dataset"
    ds.save_to_disk(str(hf_dir))

    summary = {
        "source": str(args.input),
        "output": str(args.output),
        "keep_modalities": keep,
        "splits": {name: summarize(split) for name, split in filtered_splits.items()},
    }
    (args.output / "prep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fitz_gov_repo = Path("C:/Users/yanfi/PycharmProjects/fitz-gov")
    manifest_path = write_manifest(
        output_dir=args.output,
        config_path=Path(__file__).resolve(),
        seed=args.seed,
        fitz_gov_repo=fitz_gov_repo if fitz_gov_repo.exists() else None,
        extra={
            "script": "filter_processed_modalities.py",
            "source": str(args.input),
            "keep_modalities": keep,
            "splits": {name: {"rows": len(split)} for name, split in filtered_splits.items()},
        },
        start_time=started,
    )

    for split_name, split in filtered_splits.items():
        modalities = summary["splits"][split_name]["modalities"]
        print(f"{split_name:12s}: {len(split):6d}  modalities={modalities}")
    print(f"Wrote HF DatasetDict -> {hf_dir}")
    print(f"Wrote summary        -> {args.output / 'prep_summary.json'}")
    print(f"Wrote manifest       -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
