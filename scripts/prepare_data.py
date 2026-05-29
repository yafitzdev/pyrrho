"""
prepare_data.py — Convert fitz-gov into HF datasets for pyrrho training.

Default source is the published Hugging Face contract:
`yafitzdev/fitz-gov`, config `v8`, revision `v8.0.1`. The script preserves
the query-grouped split contract: train=19,674, validation/eval=2,459,
test=2,459.

Local vault mode remains available for development; when no published split
assignment is provided, it falls back to the historical stratified 80/20 split.
For V8 experiments, the script can preserve the published V7 split contract and
append locally-generated V8 rows from a fitz-gov vault using a QA manifest.

Label mapping (V6 `governance.classification` → pyrrho label):

    ABSTAIN       -> ABSTAIN
    DISPUTED      -> DISPUTED
    TRUSTWORTHY   -> TRUSTWORTHY            (3-class default)
                  -> TRUSTWORTHY_HEDGED     (4-class, from meta.category)
                  -> TRUSTWORTHY_DIRECT     (4-class, from meta.category)

Produces:
    {output}/train.jsonl         — training split
    {output}/eval.jsonl          — validation split used for checkpoint/threshold selection
    {output}/test.jsonl          — held-out test split when available (V7+)
    {output}/hf_dataset/         — HF DatasetDict for direct use with Trainer

Run from project root:
    python scripts/prepare_data.py --output data/processed_v8
    python scripts/prepare_data.py --hf yafitzdev/fitz-gov --hf-config v7 --hf-revision v7.0.1 --output data/processed_v7
    python scripts/prepare_data.py --vault ../fitz-gov/data/fitz-gov \
      --split-manifest ../fitz-gov/data/_workspaces/qa/sdgp_v8_qa/split_assignments.jsonl \
      --output data/processed_v8_offline
    python scripts/prepare_data.py --output data/processed_v8_probe \
      --append-local-vault ../fitz-gov/data/sdgp_vault_v51_enriched \
      --append-local-manifest ../fitz-gov/data/sdgp_v8_qa/blind_label_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

LABEL2ID_3: dict[str, int] = {"ABSTAIN": 0, "DISPUTED": 1, "TRUSTWORTHY": 2}
LABEL2ID_4: dict[str, int] = {
    "ABSTAIN": 0,
    "DISPUTED": 1,
    "TRUSTWORTHY_HEDGED": 2,
    "TRUSTWORTHY_DIRECT": 3,
}

# Module-level — set in main() based on --num-classes flag.
LABEL2ID: dict[str, int] = LABEL2ID_3
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

ALLOWED_MODALITIES = {"unstructured", "structured", "code"}
APPEND_SPLITS = ("train", "eval", "test")


def label_for(case: dict, num_classes: int) -> str:
    """Derive the pyrrho training label from a V6/V7 case."""
    cls = (case.get("governance", {}).get("classification") or case.get("label") or "").upper()
    if cls == "ABSTAIN":
        return "ABSTAIN"
    if cls == "DISPUTED":
        return "DISPUTED"
    if cls == "TRUSTWORTHY":
        if num_classes == 4:
            cat = (case.get("meta", {}).get("category") or "").lower()
            return "TRUSTWORTHY_DIRECT" if "direct" in cat else "TRUSTWORTHY_HEDGED"
        return "TRUSTWORTHY"
    raise ValueError(f"case {case.get('id')!r} has unknown classification={cls!r}")


def load_vault_jsonl(path: Path) -> list[dict]:
    """Load a SDGP vault cases.jsonl — one case per line."""
    if not path.exists():
        raise FileNotFoundError(f"V6 vault JSONL not found: {path}")
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize_split_name(split: str) -> str:
    """Normalize fitz-gov/HF split names to pyrrho's local split names."""
    if split == "validation":
        return "eval"
    if split in {"train", "eval", "test"}:
        return split
    raise ValueError(f"unsupported split {split!r}; expected train|validation|eval|test")


def load_split_manifest(path: Path) -> dict[str, str]:
    """Load `{case_id: split}` from a fitz-gov QA manifest or split assignment file."""
    if not path.exists():
        raise FileNotFoundError(f"split manifest not found: {path}")
    split_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            case_id = str(row.get("case_id") or "")
            split = str(row.get("split") or "")
            if not case_id or split not in {"train", "validation", "eval", "test"}:
                raise ValueError(
                    f"{path}:{line_no}: expected case_id and split=train|validation|eval|test"
                )
            split_by_id[case_id] = normalize_split_name(split)
    return split_by_id


def apply_split_manifest(cases: list[dict], manifest_path: Path) -> dict[str, list[dict]]:
    """Assign local vault rows to train/eval/test using a published split manifest."""
    split_by_id = load_split_manifest(manifest_path)
    raw_splits: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    seen_ids: set[str] = set()
    missing: list[str] = []
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id:
            raise ValueError("local split-manifest row is missing id")
        split = split_by_id.get(case_id)
        if split is None:
            missing.append(case_id)
            continue
        raw_splits.setdefault(split, []).append(case)
        seen_ids.add(case_id)

    if missing:
        raise ValueError(
            f"{len(missing)} local cases are missing from split manifest; "
            f"first 5: {sorted(missing)[:5]}"
        )
    unused = set(split_by_id) - seen_ids
    if unused:
        raise ValueError(
            f"{len(unused)} split-manifest IDs were not found in local cases; "
            f"first 5: {sorted(unused)[:5]}"
        )
    return raw_splits


def load_candidate_pack(path: Path) -> tuple[list[dict], dict]:
    """Load a candidate modality pack without touching the active fitz-gov vault.

    `path` may be a candidate directory containing `cases.jsonl` and `manifest.json`,
    or a direct JSONL path. Directory manifests are treated as source metadata, not
    split assignments.
    """
    if path.is_dir():
        cases_path = path / "cases.jsonl"
        manifest_path = path / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
    else:
        cases_path = path
        manifest = {}

    cases = load_vault_jsonl(cases_path)
    if manifest:
        expected_rows = manifest.get("rows")
        if expected_rows is not None and int(expected_rows) != len(cases):
            raise ValueError(
                f"{path}: manifest rows={expected_rows} but cases.jsonl has {len(cases)}"
            )
        expected_modality = manifest.get("modality")
        if expected_modality:
            mismatched = [
                str(case.get("id") or "")
                for case in cases
                if case.get("meta", {}).get("modality") != expected_modality
            ]
            if mismatched:
                raise ValueError(
                    f"{path}: {len(mismatched)} rows do not match manifest modality "
                    f"{expected_modality!r}; first 5: {mismatched[:5]}"
                )

    for case in cases:
        case_id = str(case.get("id") or "")
        modality = case.get("meta", {}).get("modality")
        if not case_id:
            raise ValueError(f"{path}: candidate row without id")
        if modality not in ALLOWED_MODALITIES - {"unstructured"}:
            raise ValueError(
                f"{path}: candidate row {case_id!r} has invalid candidate modality "
                f"{modality!r}; expected structured|code"
            )

    return cases, {"path": str(path), **manifest}


def load_candidate_selection_ids(paths: list[Path] | None) -> set[str] | None:
    """Load selected case IDs from one or more probe/training manifest JSONLs."""
    if not paths:
        return None
    selected: set[str] = set()
    for path in paths:
        for row in load_vault_jsonl(path):
            case_id = str(row.get("case_id") or row.get("id") or "")
            if not case_id:
                raise ValueError(f"{path}: selection manifest row without case_id/id")
            selected.add(case_id)
    return selected


def canonical_query(case: dict) -> str:
    query = str(case.get("input", {}).get("query") or "")
    return " ".join(query.casefold().split())


def candidate_group_key(case: dict, split_key: str) -> str:
    if split_key == "id":
        return str(case.get("id") or "")
    if split_key == "query":
        return canonical_query(case) or str(case.get("id") or "")
    raise ValueError(f"unsupported candidate split key: {split_key}")


def candidate_stratum(cases: list[dict]) -> str:
    """Derive a stable split-balancing stratum for a candidate query group."""
    labels = Counter(label_for(case, 3) for case in cases)
    modalities = Counter(str(case.get("meta", {}).get("modality") or "unknown") for case in cases)
    difficulties = Counter(str(case.get("meta", {}).get("difficulty") or "unknown") for case in cases)
    label = sorted(labels.items(), key=lambda item: (-item[1], item[0]))[0][0]
    modality = sorted(modalities.items(), key=lambda item: (-item[1], item[0]))[0][0]
    difficulty = sorted(difficulties.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return f"{modality}|{label}|{difficulty}"


def split_candidate_cases(
    cases: list[dict],
    *,
    eval_ratio: float,
    test_ratio: float,
    seed: int,
    split_key: str,
) -> dict[str, list[dict]]:
    """Deterministically split candidate rows, keeping query groups together."""
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("--candidate-eval-ratio must be in [0, 1)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--candidate-test-ratio must be in [0, 1)")
    if eval_ratio + test_ratio >= 1.0:
        raise ValueError("candidate eval+test ratios must sum to less than 1")

    groups_by_stratum: dict[str, list[list[dict]]] = defaultdict(list)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_group[candidate_group_key(case, split_key)].append(case)
    for group_cases in by_group.values():
        groups_by_stratum[candidate_stratum(group_cases)].append(group_cases)

    rng = random.Random(seed)
    out: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for stratum in sorted(groups_by_stratum):
        groups = sorted(groups_by_stratum[stratum], key=lambda rows: str(rows[0].get("id") or ""))
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_ratio))
        n_eval = int(round(n * eval_ratio))
        if n_test + n_eval > n:
            n_eval = max(0, n - n_test)

        for idx, group_cases in enumerate(groups):
            if idx < n_test:
                split = "test"
            elif idx < n_test + n_eval:
                split = "eval"
            else:
                split = "train"
            out[split].extend(group_cases)

    for split in out:
        out[split].sort(key=lambda case: str(case.get("id") or ""))
    return out


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_cases(cases: list[dict]) -> dict:
    return {
        "rows": len(cases),
        "labels": counter_dict([label_for(case, 3) for case in cases]),
        "modalities": counter_dict(
            [str(case.get("meta", {}).get("modality") or "unknown") for case in cases]
        ),
        "patterns": counter_dict(
            [str(case.get("taxonomy", {}).get("pattern") or "unknown") for case in cases]
        ),
        "experts": counter_dict(
            [str(case.get("routing", {}).get("expert_fired") or "unknown") for case in cases]
        ),
        "difficulties": counter_dict(
            [str(case.get("meta", {}).get("difficulty") or "unknown") for case in cases]
        ),
    }


def append_candidate_packs_to_splits(
    raw_splits: dict[str, list[dict]],
    *,
    candidate_paths: list[Path],
    selection_manifest_paths: list[Path] | None,
    eval_ratio: float,
    test_ratio: float,
    seed: int,
    split_key: str,
) -> dict:
    """Append candidate structured/code rows to an existing split contract."""
    if not {"train", "eval"} <= set(raw_splits.keys()):
        raise ValueError("--append-candidate-pack requires an existing train/eval split contract")

    selection_ids = load_candidate_selection_ids(selection_manifest_paths)
    candidate_cases: list[dict] = []
    source_manifests: list[dict] = []
    seen_candidate_ids: set[str] = set()
    for path in candidate_paths:
        cases, manifest = load_candidate_pack(path.resolve())
        source_manifests.append(manifest)
        for case in cases:
            case_id = str(case.get("id") or "")
            if selection_ids is not None and case_id not in selection_ids:
                continue
            if case_id in seen_candidate_ids:
                raise ValueError(f"duplicate candidate case id: {case_id}")
            seen_candidate_ids.add(case_id)
            candidate_cases.append(case)

    if selection_ids is not None:
        missing = selection_ids - seen_candidate_ids
        if missing:
            raise ValueError(
                f"{len(missing)} selected candidate IDs were not found in candidate packs; "
                f"first 5: {sorted(missing)[:5]}"
            )
    if not candidate_cases:
        raise ValueError("candidate pack selection produced 0 rows")

    existing_ids = {
        str(case.get("id") or "")
        for split_cases in raw_splits.values()
        for case in split_cases
        if isinstance(case, dict)
    }
    duplicated = existing_ids & seen_candidate_ids
    if duplicated:
        raise ValueError(
            f"{len(duplicated)} candidate IDs duplicate existing split IDs; "
            f"first 5: {sorted(duplicated)[:5]}"
        )

    base_queries = {
        canonical_query(case)
        for split_cases in raw_splits.values()
        for case in split_cases
        if isinstance(case, dict) and canonical_query(case)
    }
    candidate_queries = {canonical_query(case) for case in candidate_cases if canonical_query(case)}
    query_overlap = base_queries & candidate_queries
    if query_overlap:
        raise ValueError(
            f"{len(query_overlap)} candidate queries exactly overlap the base contract; "
            f"first 5: {sorted(query_overlap)[:5]}"
        )

    candidate_splits = split_candidate_cases(
        candidate_cases,
        eval_ratio=eval_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split_key=split_key,
    )
    for split in APPEND_SPLITS:
        raw_splits.setdefault(split, []).extend(candidate_splits[split])

    return {
        "sources": source_manifests,
        "selection_manifests": [str(path) for path in (selection_manifest_paths or [])],
        "split_key": split_key,
        "seed": seed,
        "eval_ratio": eval_ratio,
        "test_ratio": test_ratio,
        "total": summarize_cases(candidate_cases),
        "splits": {split: summarize_cases(candidate_splits[split]) for split in APPEND_SPLITS},
    }


def load_hf(
    repo_id: str,
    config: str,
    revision: str | None,
) -> dict[str, list[dict]]:
    """Load the published Hugging Face dataset splits.

    For V7, preserves the train/validation/test split contract.
    """
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision else {}
    ds = load_dataset(repo_id, config, **kwargs)
    splits: dict[str, list[dict]] = {}
    if {"train", "validation", "test"} <= set(ds.keys()):
        splits["train"] = ds["train"].to_list()
        splits["eval"] = ds["validation"].to_list()
        splits["test"] = ds["test"].to_list()
    elif "train" in ds:
        splits["all"] = ds["train"].to_list()
    else:
        raise ValueError(f"HF config {config!r} has unsupported splits: {list(ds.keys())}")

    return splits


def append_local_cohort_to_splits(
    raw_splits: dict[str, list[dict]],
    *,
    vault_root: Path,
    manifest_path: Path,
    cohort: str,
) -> dict[str, int]:
    """Append a local cohort to existing train/eval/test splits by manifest assignment."""
    if not {"train", "eval"} <= set(raw_splits.keys()):
        raise ValueError("--append-local-vault requires an existing train/eval split contract")

    cases = load_vault_jsonl(vault_root / "cases.jsonl")
    additions = [
        case
        for case in cases
        if isinstance(case.get("meta"), dict) and case["meta"].get("dataset_version") == cohort
    ]
    if not additions:
        raise ValueError(f"no {cohort!r} rows found in local vault {vault_root}")

    split_by_id = load_split_manifest(manifest_path)
    split_counts = {"train": 0, "eval": 0, "test": 0}
    existing_ids = {
        str(case.get("id") or "")
        for split_cases in raw_splits.values()
        for case in split_cases
        if isinstance(case, dict)
    }
    for case in additions:
        case_id = str(case.get("id") or "")
        if not case_id:
            raise ValueError("local cohort row is missing id")
        if case_id in existing_ids:
            raise ValueError(f"local cohort row duplicates existing split id: {case_id}")
        split = split_by_id.get(case_id)
        if split is None:
            raise ValueError(f"local cohort row missing from split manifest: {case_id}")
        raw_splits.setdefault(split, []).append(case)
        split_counts[split] += 1

    return split_counts


def build_text(case: dict) -> str:
    """Concatenate query + numbered contexts. Stable format reused by encoder + SLM."""
    query = (case.get("input", {}).get("query") or "").strip()
    contexts = case.get("input", {}).get("contexts") or []
    parts = [f"Question: {query}", "", "Sources:"]
    for i, ctx in enumerate(contexts, start=1):
        text = (ctx.get("text") if isinstance(ctx, dict) else str(ctx)).strip()
        parts.append(f"[{i}] {text}")
    return "\n".join(parts)


def normalize_case(case: dict, num_classes: int) -> dict:
    """Flatten a V6/V7 case to a record suitable for an HF Dataset + downstream training."""
    label = label_for(case, num_classes)
    meta = case.get("meta", {})
    taxonomy = case.get("taxonomy", {}) if isinstance(case.get("taxonomy"), dict) else {}
    routing = case.get("routing", {}) if isinstance(case.get("routing"), dict) else {}
    contexts = case.get("input", {}).get("contexts") or []
    # Keep contexts as a plain list[str] for compatibility with existing eval scripts.
    context_texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in contexts]
    modality = meta.get("modality", "unstructured")
    return {
        "id": case.get("id", ""),
        "text": build_text(case),
        "query": (case.get("input", {}).get("query") or "").strip(),
        "contexts": context_texts,
        "context_count": meta.get("context_count", len(context_texts)),
        "label": label,
        "label_id": LABEL2ID[label],
        "expected_mode": (case.get("governance", {}).get("classification") or "").upper(),
        "category": meta.get("category", ""),
        "difficulty": meta.get("difficulty", "unknown"),
        "taxonomy_pattern": taxonomy.get("pattern", ""),
        "taxonomy_cell_id": taxonomy.get("cell_id", ""),
        "expert": routing.get("expert_fired", ""),
        "dataset_version": meta.get("dataset_version", ""),
        "modality": modality,
        "source_file": (
            f"{modality}:{meta.get('dataset_version', 'unknown')}:"
            f"{meta.get('category', 'unknown')}"
        ),
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

    expert_counts = Counter(r["expert"] for r in records).most_common(5)
    print("  top 5 experts:")
    for expert, c in expert_counts:
        pct = (c / n * 100) if n else 0.0
        print(f"    {expert:22s}: {c:5d}  ({pct:5.1f}%)")

    modality_counts = Counter(r.get("modality", "unknown") for r in records)
    if modality_counts:
        print("  by modality:")
        for modality, c in sorted(modality_counts.items()):
            pct = (c / n * 100) if n else 0.0
            print(f"    {modality:14s}: {c:5d}  ({pct:5.1f}%)")


def summarize_records(records: list[dict]) -> dict:
    return {
        "rows": len(records),
        "labels": counter_dict([str(r["label"]) for r in records]),
        "modalities": counter_dict([str(r.get("modality", "unknown")) for r in records]),
        "difficulties": counter_dict([str(r.get("difficulty", "unknown")) for r in records]),
        "experts": counter_dict([str(r.get("expert", "unknown")) for r in records]),
        "taxonomy_patterns": counter_dict(
            [str(r.get("taxonomy_pattern", "unknown")) for r in records]
        ),
    }


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
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Path to the SDGP vault directory containing cases.jsonl "
        "(default: ../fitz-gov/data/sdgp_vault_v51_enriched)",
    )
    src.add_argument(
        "--hf",
        type=str,
        default=None,
        help="Alternative: load from HuggingFace, e.g. 'yafitzdev/fitz-gov'. "
        "Default uses the published fitz-gov dataset.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="v8",
        help="HuggingFace dataset config to load (default: v8).",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="v8.0.1",
        help="HuggingFace revision/tag to load (default: v8.0.1). Use empty string for main.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Offline split assignment JSONL for --vault mode; preserves train/eval/test splits "
        "without loading the HuggingFace dataset.",
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
    parser.add_argument(
        "--append-local-vault",
        type=Path,
        default=None,
        help="Append local cohort rows from this SDGP vault after loading the published split contract.",
    )
    parser.add_argument(
        "--append-local-manifest",
        type=Path,
        default=None,
        help="QA manifest/split assignment JSONL that gives splits for appended local rows.",
    )
    parser.add_argument(
        "--append-local-cohort",
        type=str,
        default="v8",
        help="meta.dataset_version cohort to append from --append-local-vault (default: v8).",
    )
    parser.add_argument(
        "--append-candidate-pack",
        type=Path,
        action="append",
        default=None,
        help="Append a structured/code candidate pack directory containing cases.jsonl + "
        "manifest.json, or a direct candidate cases.jsonl. Repeat for multiple modalities.",
    )
    parser.add_argument(
        "--candidate-selection-manifest",
        type=Path,
        action="append",
        default=None,
        help="Optional JSONL manifest with case_id/id values selecting a subset from the "
        "candidate packs. Repeatable; omitted means use every candidate row.",
    )
    parser.add_argument(
        "--candidate-eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of candidate query groups appended to eval (default: 0.1).",
    )
    parser.add_argument(
        "--candidate-test-ratio",
        type=float,
        default=0.1,
        help="Fraction of candidate query groups appended to test (default: 0.1).",
    )
    parser.add_argument(
        "--candidate-split-key",
        choices=["query", "id"],
        default="query",
        help="Grouping key for candidate splits; query prevents exact-query leakage "
        "(default: query).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()

    # Bind global mappings based on --num-classes
    global LABEL2ID, ID2LABEL
    LABEL2ID = LABEL2ID_4 if args.num_classes == 4 else LABEL2ID_3
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    if args.hf or args.vault is None:
        repo_id = args.hf or "yafitzdev/fitz-gov"
        revision = args.hf_revision or None
        print(f"source          : HuggingFace {repo_id}")
        print(f"hf config       : {args.hf_config}")
        print(f"hf revision     : {revision or '<default branch>'}")
        raw_splits = load_hf(
            repo_id,
            args.hf_config,
            revision,
        )
    else:
        vault_root = (args.vault or Path("../fitz-gov/data/sdgp_vault_v51_enriched")).resolve()
        print(f"source          : SDGP vault {vault_root}")
        local_cases = load_vault_jsonl(vault_root / "cases.jsonl")
        if args.split_manifest is not None:
            manifest_path = args.split_manifest.resolve()
            print(f"split manifest  : {manifest_path}")
            raw_splits = apply_split_manifest(local_cases, manifest_path)
        else:
            raw_splits = {"all": local_cases}

    append_summary = None
    candidate_summary = None

    if args.append_local_vault is not None:
        if args.append_local_manifest is None:
            raise ValueError("--append-local-manifest is required with --append-local-vault")
        append_counts = append_local_cohort_to_splits(
            raw_splits,
            vault_root=args.append_local_vault.resolve(),
            manifest_path=args.append_local_manifest.resolve(),
            cohort=args.append_local_cohort,
        )
        print(
            "appended local  : "
            f"cohort={args.append_local_cohort} "
            f"train={append_counts['train']} eval={append_counts['eval']} "
            f"test={append_counts['test']}"
        )
        append_summary = {
            "cohort": args.append_local_cohort,
            "splits": append_counts,
            "vault": str(args.append_local_vault.resolve()),
            "manifest": str(args.append_local_manifest.resolve()),
        }

    if args.append_candidate_pack:
        candidate_summary = append_candidate_packs_to_splits(
            raw_splits,
            candidate_paths=args.append_candidate_pack,
            selection_manifest_paths=args.candidate_selection_manifest,
            eval_ratio=args.candidate_eval_ratio,
            test_ratio=args.candidate_test_ratio,
            seed=args.seed,
            split_key=args.candidate_split_key,
        )
        split_counts = candidate_summary["splits"]
        print(
            "appended candidates: "
            f"train={split_counts['train']['rows']} "
            f"eval={split_counts['eval']['rows']} "
            f"test={split_counts['test']['rows']} "
            f"modalities={candidate_summary['total']['modalities']}"
        )

    print(f"output dir      : {output_dir}")
    print(f"train ratio     : {args.train_ratio}")
    print(f"seed            : {args.seed}")
    print(f"num classes     : {args.num_classes} (labels: {list(LABEL2ID.keys())})")

    print("\nLoaded raw splits:")
    for name, cases in raw_splits.items():
        print(f"  {name:12s}: {len(cases)}")

    if {"train", "eval"} <= set(raw_splits.keys()):
        train = [normalize_case(c, args.num_classes) for c in raw_splits["train"]]
        eval_ = [normalize_case(c, args.num_classes) for c in raw_splits["eval"]]
        test = [normalize_case(c, args.num_classes) for c in raw_splits.get("test", [])]
        tier0 = [normalize_case(c, args.num_classes) for c in raw_splits.get("tier0_sanity", [])]
        print("\nUsing published split contract; --train-ratio/--seed do not resplit HF data.")
    else:
        all_cases = raw_splits["all"]
        raw_tier1 = [c for c in all_cases if c.get("tier", 1) == 1 or c.get("id", "").startswith("t1_")]
        raw_tier0 = [c for c in all_cases if c.get("tier", 1) == 0 or c.get("id", "").startswith("t0_")]
        print(f"  tier1: {len(raw_tier1)}, tier0: {len(raw_tier0)}")
        tier1 = [normalize_case(c, args.num_classes) for c in raw_tier1]
        tier0 = [normalize_case(c, args.num_classes) for c in raw_tier0]
        train, eval_ = stratified_split(tier1, args.train_ratio, args.seed)
        test = []

    # Defensive: ensure published/derived splits don't share IDs.
    split_ids = {
        "train": {r["id"] for r in train if r["id"]},
        "eval": {r["id"] for r in eval_ if r["id"]},
        "test": {r["id"] for r in test if r["id"]},
        "tier0_sanity": {r["id"] for r in tier0 if r["id"]},
    }
    split_items = list(split_ids.items())
    for i, (left, left_ids) in enumerate(split_items):
        for right, right_ids in split_items[i + 1 :]:
            if not left_ids or not right_ids or "tier0_sanity" in (left, right):
                continue
            overlap = left_ids & right_ids
            if overlap:
                raise ValueError(
                    f"{len(overlap)} IDs overlap between {left} and {right} "
                    f"(first 5: {sorted(overlap)[:5]})."
                )

    print_distribution("train", train)
    print_distribution("eval / validation", eval_)
    if test:
        print_distribution("test", test)
    if tier0:
        print_distribution("tier0_sanity (diagnostic)", tier0)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(eval_, output_dir / "eval.jsonl")
    if test:
        save_jsonl(test, output_dir / "test.jsonl")
    if tier0:
        save_jsonl(tier0, output_dir / "tier0_sanity.jsonl")
    print(f"\nWrote JSONL splits to {output_dir}")

    prep_summary = {
        "source": {
            "hf": bool(args.hf or args.vault is None),
            "hf_repo": args.hf or "yafitzdev/fitz-gov",
            "hf_config": args.hf_config,
            "hf_revision": args.hf_revision or None,
            "vault": str(args.vault.resolve()) if args.vault else None,
            "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
        },
        "num_classes": args.num_classes,
        "base_append": append_summary,
        "candidate_append": candidate_summary,
        "splits": {
            "train": summarize_records(train),
            "eval": summarize_records(eval_),
            "test": summarize_records(test),
            "tier0_sanity": summarize_records(tier0),
        },
    }
    with (output_dir / "prep_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(prep_summary, fh, indent=2, ensure_ascii=False)
    print(f"Wrote prep summary -> {output_dir / 'prep_summary.json'}")

    hf_dir = output_dir / "hf_dataset"
    dataset_splits = {
        "train": Dataset.from_list(train),
        "eval": Dataset.from_list(eval_),
    }
    if tier0:
        dataset_splits["tier0_sanity"] = Dataset.from_list(tier0)
    if test:
        dataset_splits["test"] = Dataset.from_list(test)
    ds_dict = DatasetDict(dataset_splits)
    ds_dict.save_to_disk(str(hf_dir))
    print(f"Wrote HF DatasetDict to {hf_dir}")

    print(
        f"\nDONE  train={len(train)} eval={len(eval_)} test={len(test)} tier0={len(tier0)} "
        f"labels={list(LABEL2ID.keys())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
