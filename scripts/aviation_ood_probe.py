"""aviation_ood_probe.py - 10-case aviation maintenance OOD probe.

This mirrors the ECU OOD probe shape: verify exact-query absence in processed
datasets, then score one or more multi-seed runs using each seed's saved
calibrated TRUSTWORTHY threshold.

Run from project root:
    python scripts/aviation_ood_probe.py
    python scripts/aviation_ood_probe.py --run g2 outputs/multi_seed_g2 --run g2.2 outputs/multi_seed_g2_2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL, build_encoder_text
from pyrrho.metrics import gated_predictions


CASES = [
    {
        "id": "air_01_trustworthy_ad_terminating_action",
        "expected": "TRUSTWORTHY",
        "query": "Did aircraft N742PX comply with AD 2026-04-11 for the left elevator hinge inspection?",
        "contexts": [
            "Maintenance record WO-8841 for N742PX lists AD 2026-04-11, left elevator hinge inspection, completed on 2026-04-18. The record says terminating action paragraph (g)(2) was performed with no cracks found.",
            "The authorized release certificate for N742PX references WO-8841 and signs the aircraft back to service after AD 2026-04-11 compliance. Inspector badge QA-17 approved the entry.",
        ],
    },
    {
        "id": "air_02_trustworthy_superseded_sb_resolved",
        "expected": "TRUSTWORTHY",
        "query": "Which service bulletin revision is valid for the fuel pump wire clamp replacement on N319QJ?",
        "contexts": [
            "Planning note lists SB FP-22 Rev A for N319QJ but marks it superseded before execution.",
            "The maintenance release for N319QJ states SB FP-22 Rev C was used for the fuel pump wire clamp replacement and closes the Rev A planning note as obsolete.",
        ],
    },
    {
        "id": "air_03_trustworthy_mel_deferral_allowed",
        "expected": "TRUSTWORTHY",
        "query": "Was N508LM allowed to depart with the right landing light inoperative under the MEL?",
        "contexts": [
            "MEL item 33-41 permits dispatch with one landing light inoperative for day VFR operations, repair category C, if the defect is placarded and logged.",
            "N508LM dispatch release on 2026-05-03 was day VFR only. The right landing light defect was placarded, entered as MEL 33-41, and accepted by maintenance control.",
        ],
    },
    {
        "id": "air_04_disputed_release_to_service",
        "expected": "DISPUTED",
        "query": "Was N220RA released to service after the nose gear shimmy inspection?",
        "contexts": [
            "The line maintenance worksheet for N220RA says the nose gear shimmy inspection found no abnormal play and lists final status: released to service.",
            "The maintenance control export for the same N220RA inspection says final status: aircraft grounded pending repeat torque check, with no release-to-service signature.",
        ],
    },
    {
        "id": "air_05_disputed_ad_interval",
        "expected": "DISPUTED",
        "query": "Is the repetitive inspection interval for AD 2025-19-07 100 flight hours or 300 flight hours?",
        "contexts": [
            "Operator compliance matrix for AD 2025-19-07 lists repetitive inspection every 100 flight hours for the affected flap track bracket.",
            "The engineering order attached to the same AD 2025-19-07 package states the repetitive inspection interval is 300 flight hours and labels the 100-hour interval as preliminary.",
        ],
    },
    {
        "id": "air_06_disputed_component_status",
        "expected": "DISPUTED",
        "query": "Is hydraulic pump HP-77 serviceable for installation on N611VX?",
        "contexts": [
            "Component shop tag for HP-77 says serviceable, pressure-tested, and eligible for installation on A320-family aircraft.",
            "The operator quarantine log for HP-77 says the same serial number failed post-shop leak check and is blocked from installation on N611VX.",
        ],
    },
    {
        "id": "air_07_abstain_wrong_tail_number",
        "expected": "ABSTAIN",
        "query": "Did N742PX complete the pitot-static leak check after the avionics bay repair?",
        "contexts": [
            "Aircraft N742XP completed a pitot-static leak check after avionics bay repair on 2026-04-21. The result was within limits.",
            "N742PX has an avionics bay repair work package open, but the retrieved status page only lists parts staging and technician assignment, not the pitot-static test result.",
        ],
    },
    {
        "id": "air_08_abstain_wrong_serial_range",
        "expected": "ABSTAIN",
        "query": "Does AD 2026-02-09 apply to engine serial V17344 on N901CT?",
        "contexts": [
            "AD 2026-02-09 applies to engine serials V17000 through V17299 installed on the CT-900 fleet.",
            "N901CT has engine serial V17344. The fleet summary says CT-900 aircraft may be affected generally, but it does not list applicability for serials above V17299.",
        ],
    },
    {
        "id": "air_09_abstain_setup_no_signed_release",
        "expected": "ABSTAIN",
        "query": "Was N455KA signed off after the emergency locator transmitter battery replacement?",
        "contexts": [
            "The work package for N455KA lists ELT battery replacement steps, required tools, and inspection criteria.",
            "Parts issue record confirms a replacement ELT battery was delivered to the hangar. The retrieved documents do not include a signed maintenance release or final inspection result.",
        ],
    },
    {
        "id": "air_10_abstain_wrong_revision",
        "expected": "ABSTAIN",
        "query": "Did N319QJ comply with SB FP-22 Rev C?",
        "contexts": [
            "N319QJ completed SB FP-22 Rev B on 2026-03-14 for the fuel pump wire clamp replacement.",
            "A later planning memo says Rev C adds an inspection of clamp orientation, but the retrieved records do not show Rev C execution for N319QJ.",
        ],
    },
]


DEFAULT_RUNS = [
    ("g2", "outputs/multi_seed_g2"),
    ("g2.1-v8-probe", "outputs/multi_seed_g2_1_v8_probe"),
    ("g2.2", "outputs/multi_seed_g2_2"),
]

DEFAULT_DATASETS = [
    ("processed_v7", "data/processed_v7"),
    ("processed_v8_probe", "data/processed_v8_probe"),
    ("processed_v8_balanced_controls", "data/processed_v8_balanced_controls"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help="Multi-seed run directory to score, e.g. --run g2 outputs/multi_seed_g2",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help="Processed dataset dir to exact-query check, e.g. --dataset processed_v7 data/processed_v7",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/aviation_ood_probe/comparison.json"),
        help="Where to write the full JSON artifact",
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return name


def hf_dataset_path(path: Path) -> Path:
    if (path / "hf_dataset").exists():
        return path / "hf_dataset"
    return path


def query_match_report(path: Path) -> dict[str, list[str]]:
    ds = load_from_disk(str(hf_dataset_path(path)))
    matches: dict[str, set[str]] = {}
    for split in ds.keys():
        for row in ds[split]:
            query = str(row.get("query") or "").strip().lower()
            if not query:
                continue
            matches.setdefault(query, set()).add(split)
    out: dict[str, list[str]] = {}
    for case in CASES:
        out[case["id"]] = sorted(matches.get(case["query"].strip().lower(), set()))
    return out


def seed_dirs(run_dir: Path) -> list[Path]:
    out = []
    for path in run_dir.glob("seed_*"):
        if path.is_dir():
            out.append(path)
    out.sort(key=lambda p: int(p.name.split("_", 1)[1]))
    if not out:
        raise FileNotFoundError(f"no seed_* directories found under {run_dir}")
    return out


def collapsed_probs(logits: np.ndarray) -> tuple[float, float, float]:
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    if probs.shape[0] == 3:
        return float(probs[0]), float(probs[1]), float(probs[2])
    if probs.shape[0] == 4:
        return float(probs[0]), float(probs[1]), float(probs[2] + probs[3])
    raise ValueError(f"unsupported logits shape: {probs.shape}")


def score_checkpoint(seed_dir: Path, device: str) -> dict:
    ckpt = seed_dir / "best_model"
    final_metrics = json.loads((seed_dir / "final_metrics.json").read_text(encoding="utf-8"))
    threshold = float(final_metrics["threshold"])

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()

    rows = []
    with torch.no_grad():
        for case in CASES:
            text = build_encoder_text(case["query"], case["contexts"])
            enc = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt").to(device)
            logits = model(**enc).logits[0].float().cpu().numpy()
            num_classes = int(logits.shape[-1])
            argmax_id = int(gated_predictions(logits.reshape(1, -1), 0.0, num_classes=num_classes)[0])
            cal_id = int(gated_predictions(logits.reshape(1, -1), threshold, num_classes=num_classes)[0])
            p_a, p_d, p_t = collapsed_probs(logits)
            rows.append(
                {
                    "id": case["id"],
                    "expected": case["expected"],
                    "argmax": ID2LABEL[argmax_id],
                    "calibrated": ID2LABEL[cal_id],
                    "p_abstain": p_a,
                    "p_disputed": p_d,
                    "p_trustworthy": p_t,
                    "ok": ID2LABEL[cal_id] == case["expected"],
                }
            )

    score = sum(int(row["ok"]) for row in rows)
    return {
        "seed": int(seed_dir.name.split("_", 1)[1]),
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "score": score,
        "total": len(rows),
        "rows": rows,
    }


def summarize_run(label: str, run_dir: Path, device: str) -> dict:
    per_seed = [score_checkpoint(seed_dir, device) for seed_dir in seed_dirs(run_dir)]
    mean_score = float(np.mean([entry["score"] for entry in per_seed]))
    case_hits: dict[str, int] = {case["id"]: 0 for case in CASES}
    for entry in per_seed:
        for row in entry["rows"]:
            case_hits[row["id"]] += int(row["ok"])
    return {
        "label": label,
        "run_dir": str(run_dir),
        "mean_score": mean_score,
        "per_seed": per_seed,
        "case_hits": case_hits,
    }


def print_dataset_report(dataset_results: list[dict]) -> None:
    print("Exact-query match check:")
    for dataset in dataset_results:
        label = dataset["label"]
        matches = dataset["matches"]
        total_matches = sum(1 for splits in matches.values() if splits)
        print(f"  {label}: {total_matches}/{len(CASES)} queries matched exactly")
        for case in CASES:
            splits = matches[case["id"]]
            if splits:
                print(f"    {case['id']}: MATCH {','.join(splits)}")
            else:
                print(f"    {case['id']}: 0 matches")
    print()


def print_run_report(run_results: list[dict]) -> None:
    for run in run_results:
        print(f"{run['label']}  ({run['run_dir']})")
        print(f"{'seed':>6s} {'tau':>6s} {'score':>7s}")
        print("-" * 24)
        for entry in run["per_seed"]:
            print(f"{entry['seed']:6d} {entry['threshold']:6.2f} {entry['score']:>2d}/{entry['total']:<4d}")
        print("-" * 24)
        print(f"{'mean':>6s} {'':6s} {run['mean_score']:.2f}/{len(CASES)}")
        print()


def print_pairwise_delta(run_a: dict, run_b: dict) -> None:
    print(f"Pairwise delta: {run_a['label']} -> {run_b['label']}")
    print(f"  mean score: {run_a['mean_score']:.2f}/{len(CASES)} -> {run_b['mean_score']:.2f}/{len(CASES)}")
    print("  per-case hit counts:")
    for case in CASES:
        cid = case["id"]
        before = run_a["case_hits"][cid]
        after = run_b["case_hits"][cid]
        delta = after - before
        print(f"    {cid}: {before}/{len(run_a['per_seed'])} -> {after}/{len(run_b['per_seed'])} ({delta:+d})")
    print()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    dataset_specs = args.dataset or DEFAULT_DATASETS
    run_specs = args.run or DEFAULT_RUNS

    dataset_results = []
    for label, raw_path in dataset_specs:
        dataset_results.append(
            {
                "label": label,
                "path": str(raw_path),
                "matches": query_match_report(Path(raw_path)),
            }
        )

    run_results = [summarize_run(label, Path(raw_path), device) for label, raw_path in run_specs]

    print(f"Device: {device}\n")
    print_dataset_report(dataset_results)
    print_run_report(run_results)
    for i in range(1, len(run_results)):
        print_pairwise_delta(run_results[i - 1], run_results[i])

    payload = {
        "device": device,
        "datasets": dataset_results,
        "runs": run_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
