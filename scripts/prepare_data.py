"""
prepare_data.py — Convert fitz-gov v5 JSON files into HF datasets for pyrrho training.

Maps the 4 fitz-gov category files to the 3-class governance label:

    abstention.json          -> ABSTAIN
    dispute.json             -> DISPUTED
    trustworthy_hedged.json  -> TRUSTWORTHY
    trustworthy_direct.json  -> TRUSTWORTHY

Produces:
    {output}/train.jsonl         — 80% of tier1, stratified by (label, difficulty)
    {output}/eval.jsonl          — 20% of tier1, stratified the same way
    {output}/tier0_sanity.jsonl  — full tier0 held out as sanity gate (>=95% threshold)
    {output}/hf_dataset/         — HF DatasetDict for direct use with Trainer

Run from project root:
    python scripts/prepare_data.py --fitz-gov ../fitz-gov/data --output data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


FILE_TO_LABEL_3: dict[str, str] = {
    "abstention.json": "ABSTAIN",
    "dispute.json": "DISPUTED",
    "trustworthy_hedged.json": "TRUSTWORTHY",
    "trustworthy_direct.json": "TRUSTWORTHY",
}

FILE_TO_LABEL_4: dict[str, str] = {
    "abstention.json": "ABSTAIN",
    "dispute.json": "DISPUTED",
    "trustworthy_hedged.json": "TRUSTWORTHY_HEDGED",
    "trustworthy_direct.json": "TRUSTWORTHY_DIRECT",
}

LABEL2ID_3: dict[str, int] = {"ABSTAIN": 0, "DISPUTED": 1, "TRUSTWORTHY": 2}
LABEL2ID_4: dict[str, int] = {
    "ABSTAIN": 0,
    "DISPUTED": 1,
    "TRUSTWORTHY_HEDGED": 2,
    "TRUSTWORTHY_DIRECT": 3,
}

# Module-level — set in main() based on --num-classes flag.
FILE_TO_LABEL: dict[str, str] = FILE_TO_LABEL_3
LABEL2ID: dict[str, int] = LABEL2ID_3
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def load_tier(fitz_gov_root: Path, tier: str) -> list[dict]:
    """Load all cases from a tier directory, attaching the 3-class governance label."""
    tier_dir = fitz_gov_root / tier
    if not tier_dir.exists():
        raise FileNotFoundError(f"fitz-gov tier directory not found: {tier_dir}")

    out: list[dict] = []
    for filename, label in FILE_TO_LABEL.items():
        path = tier_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing expected fitz-gov file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for case in data.get("cases", []):
            case["label"] = label
            case["label_id"] = LABEL2ID[label]
            case["source_file"] = filename
            out.append(case)
    return out


def build_text(case: dict) -> str:
    """Concatenate query + numbered contexts. Stable format reused by encoder + SLM."""
    query = (case.get("query") or "").strip()
    contexts = case.get("contexts") or []
    if not isinstance(contexts, list):
        contexts = [str(contexts)]

    parts = [f"Question: {query}", "", "Sources:"]
    for i, ctx in enumerate(contexts, start=1):
        ctx_str = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False)
        parts.append(f"[{i}] {ctx_str.strip()}")
    return "\n".join(parts)


def normalize_case(case: dict) -> dict:
    """Flatten a fitz-gov case to a record suitable for an HF Dataset."""
    contexts = case.get("contexts") or []
    return {
        "id": case.get("id", ""),
        "text": build_text(case),
        "query": (case.get("query") or "").strip(),
        "contexts": contexts,
        "context_count": case.get("context_count", len(contexts)),
        "label": case["label"],
        "label_id": case["label_id"],
        "expected_mode": (case.get("expected_mode") or "").upper(),
        "category": case.get("category", ""),
        "subcategory": case.get("subcategory", ""),
        "difficulty": case.get("difficulty", "unknown"),
        "domain": case.get("domain", "unknown"),
        "query_type": case.get("query_type", "unknown"),
        "source_type": case.get("source_type", "unknown"),
        "reasoning_type": case.get("reasoning_type", "unknown"),
        "evidence_pattern": case.get("evidence_pattern", "unknown"),
        "source_file": case["source_file"],
    }


def print_distribution(name: str, records: list[dict]) -> None:
    n = len(records)
    print(f"\n[{name}] total: {n}")

    label_counts = Counter(r["label"] for r in records)
    print("  by label:")
    for label in LABEL2ID.keys():
        c = label_counts.get(label, 0)
        pct = (c / n * 100) if n else 0.0
        print(f"    {label:20s}: {c:5d}  ({pct:5.1f}%)")

    diff_counts = Counter(r["difficulty"] for r in records)
    print("  by difficulty:")
    for diff, c in sorted(diff_counts.items()):
        pct = (c / n * 100) if n else 0.0
        print(f"    {diff:14s}: {c:5d}  ({pct:5.1f}%)")

    dom_counts = Counter(r["domain"] for r in records).most_common(5)
    print("  top 5 domains:")
    for dom, c in dom_counts:
        pct = (c / n * 100) if n else 0.0
        print(f"    {dom:14s}: {c:5d}  ({pct:5.1f}%)")


def stratified_split(
    records: list[dict],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratify by (label, difficulty) — preserves class balance AND difficulty balance."""
    strata = [f"{r['label']}|{r['difficulty']}" for r in records]
    train, eval_ = train_test_split(
        records,
        train_size=train_ratio,
        stratify=strata,
        random_state=seed,
        shuffle=True,
    )
    return train, eval_


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fitz-gov",
        type=Path,
        default=Path("../fitz-gov/data"),
        help="Path to fitz-gov data directory (default: ../fitz-gov/data, "
        "assumes you run this from the pyrrho project root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Where to write the prepared splits (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the stratified split (default: 42)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        choices=[3, 4],
        default=3,
        help="Label space cardinality: 3 = TRUSTWORTHY collapsed (matches fitz-sage baseline). "
        "4 = TRUSTWORTHY_HEDGED vs TRUSTWORTHY_DIRECT kept separate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fitz_gov_root = args.fitz_gov.resolve()
    output_dir = args.output.resolve()

    # Bind global mappings based on --num-classes
    global FILE_TO_LABEL, LABEL2ID, ID2LABEL
    if args.num_classes == 4:
        FILE_TO_LABEL = FILE_TO_LABEL_4
        LABEL2ID = LABEL2ID_4
    else:
        FILE_TO_LABEL = FILE_TO_LABEL_3
        LABEL2ID = LABEL2ID_3
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    print(f"fitz-gov source : {fitz_gov_root}")
    print(f"output dir      : {output_dir}")
    print(f"train ratio     : {args.train_ratio}")
    print(f"seed            : {args.seed}")
    print(f"num classes     : {args.num_classes} (labels: {list(LABEL2ID.keys())})")

    if not fitz_gov_root.exists():
        print(f"ERROR: fitz-gov directory not found: {fitz_gov_root}", file=sys.stderr)
        print(
            "Pass --fitz-gov pointing to your fitz-gov data directory, e.g. "
            "--fitz-gov C:/Users/yanfi/PycharmProjects/fitz-gov/data",
            file=sys.stderr,
        )
        return 1

    print("\nLoading tier1_core...")
    raw_tier1 = load_tier(fitz_gov_root, "tier1_core")
    tier1 = [normalize_case(c) for c in raw_tier1]
    print(f"  loaded {len(tier1)} tier1 cases")

    print("\nLoading tier0_sanity...")
    raw_tier0 = load_tier(fitz_gov_root, "tier0_sanity")
    tier0 = [normalize_case(c) for c in raw_tier0]
    print(f"  loaded {len(tier0)} tier0 cases")

    # Defensive: ensure tier0 and tier1 don't share IDs (would silently leak)
    tier0_ids = {r["id"] for r in tier0 if r["id"]}
    tier1_ids = {r["id"] for r in tier1 if r["id"]}
    overlap = tier0_ids & tier1_ids
    if overlap:
        print(
            f"WARNING: {len(overlap)} IDs overlap between tier0 and tier1 "
            f"(first 5: {sorted(overlap)[:5]}). Investigate before training.",
            file=sys.stderr,
        )

    print_distribution("tier1_core (full)", tier1)
    print_distribution("tier0_sanity (held out)", tier0)

    train, eval_ = stratified_split(tier1, args.train_ratio, args.seed)
    print_distribution(f"tier1 train ({args.train_ratio:.0%})", train)
    print_distribution(f"tier1 eval ({1 - args.train_ratio:.0%})", eval_)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(eval_, output_dir / "eval.jsonl")
    save_jsonl(tier0, output_dir / "tier0_sanity.jsonl")
    print(f"\nWrote JSONL splits to {output_dir}")

    hf_dir = output_dir / "hf_dataset"
    ds_dict = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "eval": Dataset.from_list(eval_),
            "tier0_sanity": Dataset.from_list(tier0),
        }
    )
    ds_dict.save_to_disk(str(hf_dir))
    print(f"Wrote HF DatasetDict to {hf_dir}")

    print(
        f"\nDONE  train={len(train)} eval={len(eval_)} tier0={len(tier0)} "
        f"labels={list(LABEL2ID.keys())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
