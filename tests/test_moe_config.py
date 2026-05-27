from __future__ import annotations

import pytest
import yaml

from pyrrho.moe import PyrrhoMoEConfig


def test_default_moe_config_matches_budget_window() -> None:
    cfg = PyrrhoMoEConfig()
    report = cfg.budget_report()
    counts = report["parameters"]

    assert 3_900_000_000 <= counts["total"] <= 4_100_000_000
    assert 380_000_000 <= counts["active_inclusive"] <= 430_000_000
    assert counts["active_excluding_embedding"] < counts["active_inclusive"]
    assert report["derived"]["physical_shards_per_group"] == 2


def test_qwen_alpha_config_matches_budget_window() -> None:
    raw = yaml.safe_load(
        open("configs/moe/pyrrho_moe_g3_alpha_qwen.yaml", encoding="utf-8")
    )
    cfg = PyrrhoMoEConfig.from_mapping(raw["architecture"])
    report = cfg.budget_report()
    counts = report["parameters"]

    assert report["derived"]["head_dim"] == 128
    assert report["derived"]["q_dim"] == 2048
    assert report["derived"]["kv_dim"] == 1024
    assert counts["total"] == 4_083_139_633
    assert counts["active_inclusive"] == 423_871_537
    assert report["derived"]["physical_shards_per_group"] == 6
    assert all(report["budget_checks"].values())


def test_moe_config_rejects_top2_cpu_path() -> None:
    cfg = PyrrhoMoEConfig(top_k=2)
    with pytest.raises(ValueError, match="top_k=1"):
        cfg.parameter_counts()


def test_moe_config_requires_even_semantic_shards() -> None:
    cfg = PyrrhoMoEConfig(experts_per_moe_layer=10)
    with pytest.raises(ValueError, match="divide evenly"):
        cfg.parameter_counts()
