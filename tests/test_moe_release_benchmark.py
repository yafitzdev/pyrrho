from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_moe_release.py"
SPEC = importlib.util.spec_from_file_location("benchmark_moe_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_moe_release = importlib.util.module_from_spec(SPEC)
sys.modules["benchmark_moe_release"] = benchmark_moe_release
SPEC.loader.exec_module(benchmark_moe_release)


def test_metric_summary_reports_basic_stats() -> None:
    summary = benchmark_moe_release.metric_summary([1.0, 2.0, 3.0])

    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_prediction_summary_counts_seed_rejections() -> None:
    summary = benchmark_moe_release.prediction_summary(
        [
            {
                "classification": "TRUSTWORTHY",
                "seed_rejected": {"42": False, "7": True, "1337": False},
            },
            {
                "classification": "ABSTAIN",
                "seed_rejected": {"42": True, "7": False, "1337": True},
            },
        ]
    )

    assert summary["rows"] == 2
    assert summary["counts"] == {"TRUSTWORTHY": 1, "ABSTAIN": 1}
    assert summary["seed_level_rejections"] == 3


def test_normalize_batch_sizes_deduplicates_in_order() -> None:
    assert benchmark_moe_release.normalize_batch_sizes(16, [1, 4, 4, 16, 1]) == [1, 4, 16]
    assert benchmark_moe_release.normalize_batch_sizes(8, None) == [8]


def test_fastest_profile_prefers_lowest_ms_per_row_then_smaller_batch() -> None:
    profiles = [
        {
            "batch_size": 16,
            "aggregate": {
                "ms_per_row": {"mean": 12.0},
                "rows_per_second": {"mean": 80.0},
            },
        },
        {
            "batch_size": 4,
            "aggregate": {
                "ms_per_row": {"mean": 10.0},
                "rows_per_second": {"mean": 100.0},
            },
        },
        {
            "batch_size": 8,
            "aggregate": {
                "ms_per_row": {"mean": 10.0},
                "rows_per_second": {"mean": 100.0},
            },
        },
    ]

    assert benchmark_moe_release.fastest_profile(profiles)["batch_size"] == 4
