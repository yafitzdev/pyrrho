"""automotive_ood_probe.py — exact recovered 10-case ECU/test-management OOD probe.

Runs the same synthetic automotive/ECU cases previously used for the manual
g2 OOD check, verifies exact-string query absence in one or more processed
datasets, then scores one or more multi-seed run directories using each
seed's saved calibrated TRUSTWORTHY threshold.

Default comparison:
    - outputs/multi_seed_g2
    - outputs/multi_seed_g2_1_v8_probe

Default exact-query datasets:
    - data/processed_v7
    - data/processed_v8_probe

Run from project root:
    python scripts/automotive_ood_probe.py
    python scripts/automotive_ood_probe.py --run g2 outputs/multi_seed_g2 --run g2.1-v8-probe outputs/multi_seed_g2_1_v8_probe
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
        "id": "ecu_01_trustworthy_hil_boot_regression",
        "expected": "TRUSTWORTHY",
        "query": "Did ECU release R17.4 pass the HIL boot-time regression test after the watchdog reset patch?",
        "contexts": [
            "HIL campaign HIL-BOOT-774 for ECU release R17.4 executed on bench B12 after watchdog reset patch WD-219. The measured cold boot time was 820 ms, below the 950 ms requirement, and all 40 restart cycles completed without DTCs.",
            "Test management record TM-88421 marks HIL-BOOT-774 as PASS for ECU R17.4. The attached lab summary states that the watchdog reset patch did not introduce a boot-time regression.",
        ],
    },
    {
        "id": "ecu_02_trustworthy_acceptance_run",
        "expected": "TRUSTWORTHY",
        "query": "Which test run should be used as the valid acceptance evidence for the ADAS ECU lane-keep warning test?",
        "contexts": [
            "Test run LKAS-WARN-2026-04-18-03 is tagged VALIDATED in the test management system for ADAS ECU software 9.6.2. It uses production camera firmware CAM-4.11 and the approved scenario set SCN-LKAS-12.",
            "Run LKAS-WARN-2026-04-18-02 was invalidated because the camera firmware was an engineering build. The validation note says to use LKAS-WARN-2026-04-18-03 as the acceptance evidence.",
        ],
    },
    {
        "id": "ecu_03_trustworthy_diagnostic_traceability",
        "expected": "TRUSTWORTHY",
        "query": "Does the diagnostic session timeout test satisfy requirement DIAG-REQ-114 for body ECU build 5.8.12?",
        "contexts": [
            "Requirement DIAG-REQ-114 requires the body ECU to leave extended diagnostic session after 5.0 seconds plus or minus 0.5 seconds of inactivity. Test case TC-DIAG-114 measured timeouts of 5.1, 5.0, and 5.2 seconds on body ECU build 5.8.12.",
            "The test management trace links TC-DIAG-114 to DIAG-REQ-114 and records final verdict PASS for body ECU build 5.8.12. No deviations or waivers are attached.",
        ],
    },
    {
        "id": "ecu_04_disputed_dtc_powercycle",
        "expected": "DISPUTED",
        "query": "What was the final verdict of the DTC clear-after-power-cycle test for BCM build 5.8.12?",
        "contexts": [
            "The lab execution log for BCM build 5.8.12 says DTC clear-after-power-cycle failed because DTC B1321 remained active after ignition cycle 3. Final verdict: FAIL.",
            "The test management export for the same BCM build 5.8.12 and test case TC-BCM-DTC-044 lists final verdict PASS, with no open defects linked.",
        ],
    },
    {
        "id": "ecu_05_disputed_can_timeout",
        "expected": "DISPUTED",
        "query": "Is the powertrain ECU CAN message timeout threshold configured to 100 ms or 250 ms in release PT-22.1?",
        "contexts": [
            "Calibration file PT-22.1-CAN.arxml defines EngineStatus timeout supervision as 100 ms for the powertrain ECU network management test.",
            "The released test specification for PT-22.1 states that EngineStatus timeout supervision is 250 ms and flags any 100 ms timeout as an obsolete pre-release value.",
        ],
    },
    {
        "id": "ecu_06_disputed_test_management_status",
        "expected": "DISPUTED",
        "query": "Can the release manager mark the steering ECU torque plausibility regression as passed for SW 3.14.0?",
        "contexts": [
            "Jenkins job STEER-TQ-REG-991 for steering ECU SW 3.14.0 completed all torque plausibility regression steps with PASS and uploaded the result package to the release dashboard.",
            "The test management system rejects STEER-TQ-REG-991 for SW 3.14.0 because the bench used sensor simulator profile SIM-TQ-OLD instead of the approved SIM-TQ-2026 profile. The release dashboard status is BLOCKED.",
        ],
    },
    {
        "id": "ecu_07_abstain_wrong_ecu_release",
        "expected": "ABSTAIN",
        "query": "Did the inverter ECU INV-8.2 pass the high-voltage interlock loop open-circuit test?",
        "contexts": [
            "Battery management ECU BMS-8.2 passed the high-voltage contactor weld detection test on bench HV-03. The result does not include inverter ECU testing.",
            "Inverter ECU INV-8.1 passed the high-voltage interlock loop open-circuit test in March. The report predates INV-8.2 and is marked superseded.",
        ],
    },
    {
        "id": "ecu_08_abstain_setup_no_result",
        "expected": "ABSTAIN",
        "query": "What was the verdict of the UDS security-access negative-response test for gateway ECU GW-6.0?",
        "contexts": [
            "The gateway ECU GW-6.0 security-access test plan lists negative-response checks for invalid seed/key attempts, lockout duration, and NRC 0x35 handling.",
            "Bench setup notes confirm that CANoe project GW_SEC_ACC_2026 was loaded and the gateway ECU was flashed to GW-6.0 before execution. The execution result table is not included in the retrieved sources.",
        ],
    },
    {
        "id": "ecu_09_abstain_missing_testmanagement_evidence",
        "expected": "ABSTAIN",
        "query": "Which defect ID blocks closure of the ABS ECU wheel-speed dropout regression in test management?",
        "contexts": [
            "The ABS ECU wheel-speed dropout regression is part of the brake software qualification suite. The suite contains tests for left-front, right-front, and rear wheel sensor dropout.",
            "A project status slide says the brake team is reviewing open defects before release sign-off, but it does not list defect IDs or test management links for the wheel-speed dropout regression.",
        ],
    },
    {
        "id": "ecu_10_abstain_wrong_platform",
        "expected": "ABSTAIN",
        "query": "Did the AUTOSAR COM signal invalidation test pass on the zonal controller ZC-2 prototype?",
        "contexts": [
            "The AUTOSAR COM signal invalidation test passed on the central gateway CGW-1 production sample using software branch rel/cgw/2026.05.",
            "The zonal controller ZC-2 prototype integration note covers Ethernet wake-up behavior and power-mode transitions, but it does not mention AUTOSAR COM signal invalidation results.",
        ],
    },
]

DEFAULT_RUNS = [
    ("g2", "outputs/multi_seed_g2"),
    ("g2.1-v8-probe", "outputs/multi_seed_g2_1_v8_probe"),
]

DEFAULT_DATASETS = [
    ("processed_v7", "data/processed_v7"),
    ("processed_v8_probe", "data/processed_v8_probe"),
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
        default=Path("outputs/automotive_ood_probe/comparison.json"),
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
        delta_str = f"{delta:+d}"
        print(f"    {cid}: {before}/{len(run_a['per_seed'])} -> {after}/{len(run_b['per_seed'])} ({delta_str})")
    print("  per-seed flips:")
    rows_a = {entry["seed"]: {row["id"]: row for row in entry["rows"]} for entry in run_a["per_seed"]}
    rows_b = {entry["seed"]: {row["id"]: row for row in entry["rows"]} for entry in run_b["per_seed"]}
    for seed in sorted(set(rows_a) & set(rows_b)):
        flips = []
        for case in CASES:
            cid = case["id"]
            pred_a = rows_a[seed][cid]["calibrated"]
            pred_b = rows_b[seed][cid]["calibrated"]
            if pred_a != pred_b:
                flips.append(f"{cid}: {pred_a} -> {pred_b}")
        score_a = next(entry["score"] for entry in run_a["per_seed"] if entry["seed"] == seed)
        score_b = next(entry["score"] for entry in run_b["per_seed"] if entry["seed"] == seed)
        if flips:
            print(f"    seed {seed}: {score_a}/{len(CASES)} -> {score_b}/{len(CASES)}")
            for flip in flips:
                print(f"      {flip}")
        else:
            print(f"    seed {seed}: {score_a}/{len(CASES)} -> {score_b}/{len(CASES)}  (no prediction changes)")
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
    if len(run_results) >= 2:
        print_pairwise_delta(run_results[0], run_results[1])

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
