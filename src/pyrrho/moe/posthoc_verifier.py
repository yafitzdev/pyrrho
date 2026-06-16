"""Runtime helpers for packaged MoE post-hoc verifier rerankers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pyrrho.metrics import ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID, gated_predictions

PACKAGE_SCHEMA_VERSION = "pyrrho_moe_posthoc_verifier_package_v1"
FEATURE_SCHEMA_VERSION = "pyrrho_moe_posthoc_features_v1"


@dataclass(frozen=True)
class PosthocVerifierSeed:
    """Loaded verifier and thresholds for one packaged seed."""

    seed: int
    base_threshold: float
    verifier_threshold: float
    verifier: Any
    manifest_entry: dict[str, Any]


@dataclass(frozen=True)
class PosthocVerifierResult:
    """Prediction arrays after applying a packaged verifier policy."""

    base_predictions: np.ndarray
    guarded_predictions: np.ndarray
    accept_scores: np.ndarray
    rejected_mask: np.ndarray

    @property
    def rejected_count(self) -> int:
        return int(self.rejected_mask.sum())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_package_path(package_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (package_dir / path).resolve()


def feature_schema_from_config(model_config: dict[str, Any]) -> dict[str, Any]:
    num_labels = int(model_config.get("num_labels", 3))
    num_routes = int(model_config["num_routes"])
    num_taxonomy = int(model_config["num_taxonomy_patterns"])
    num_scalars = int(model_config["num_scalar_targets"])
    blocks = [
        {"name": "governance_logits", "width": num_labels},
        {"name": "governance_probs", "width": num_labels},
        {"name": "route_logits", "width": num_routes},
        {"name": "route_probs", "width": num_routes},
        {"name": "taxonomy_logits", "width": num_taxonomy},
        {"name": "taxonomy_probs", "width": num_taxonomy},
        {"name": "scalar_preds", "width": num_scalars},
        {"name": "trustworthy_probability", "width": 1},
        {"name": "trust_margin_vs_best_non_trustworthy", "width": 1},
        {"name": "disputed_minus_abstain_logit", "width": 1},
        {"name": "governance_entropy", "width": 1},
        {"name": "route_entropy", "width": 1},
        {"name": "taxonomy_entropy", "width": 1},
        {"name": "route_pred_one_hot", "width": num_routes},
        {"name": "taxonomy_pred_one_hot", "width": num_taxonomy},
    ]
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "total_width": int(sum(int(block["width"]) for block in blocks)),
        "blocks": blocks,
        "candidate_policy": "score every row; demote only base TRUSTWORTHY predictions below threshold",
        "demotion_policy": "replace rejected TRUSTWORTHY with the higher ABSTAIN/DISPUTED base logit",
    }


def _as_2d_float(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def softmax(values: np.ndarray) -> np.ndarray:
    arr = _as_2d_float("values", values)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    arr = _as_2d_float("probs", probs)
    clipped = np.clip(arr, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1, keepdims=True)


def one_hot(ids: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"ids must be 1D, got shape {arr.shape}")
    if int(width) <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if arr.size and (arr.min() < 0 or arr.max() >= int(width)):
        raise ValueError(f"ids are outside one-hot width {width}")
    out = np.zeros((arr.shape[0], int(width)), dtype=np.float32)
    if arr.size:
        out[np.arange(arr.shape[0]), arr] = 1.0
    return out


def _validate_row_count(arrays: dict[str, np.ndarray]) -> int:
    sizes = {name: int(arr.shape[0]) for name, arr in arrays.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"feature inputs have mismatched row counts: {sizes}")
    return next(iter(sizes.values()), 0)


def build_posthoc_features(
    *,
    governance_logits: np.ndarray,
    route_logits: np.ndarray,
    taxonomy_logits: np.ndarray,
    scalar_preds: np.ndarray,
) -> np.ndarray:
    """Build verifier features from frozen MoE outputs.

    This must stay byte-for-byte compatible in column order with
    ``scripts/train_moe_posthoc_verifier.py``.
    """

    governance = _as_2d_float("governance_logits", governance_logits)
    route = _as_2d_float("route_logits", route_logits)
    taxonomy = _as_2d_float("taxonomy_logits", taxonomy_logits)
    scalars = _as_2d_float("scalar_preds", scalar_preds)
    _validate_row_count(
        {
            "governance_logits": governance,
            "route_logits": route,
            "taxonomy_logits": taxonomy,
            "scalar_preds": scalars,
        }
    )
    if governance.shape[1] <= TRUSTWORTHY_ID:
        raise ValueError("governance_logits must include ABSTAIN/DISPUTED/TRUSTWORTHY columns")

    governance_probs = softmax(governance)
    route_probs = softmax(route)
    taxonomy_probs = softmax(taxonomy)
    route_pred = route.argmax(axis=-1)
    taxonomy_pred = taxonomy.argmax(axis=-1)
    p_t = governance_probs[:, [TRUSTWORTHY_ID]]
    non_t = governance_probs[:, [ABSTAIN_ID, DISPUTED_ID]]
    trust_margin = p_t - non_t.max(axis=1, keepdims=True)
    disputed_minus_abstain = governance[:, [DISPUTED_ID]] - governance[:, [ABSTAIN_ID]]
    return np.concatenate(
        [
            governance,
            governance_probs,
            route,
            route_probs,
            taxonomy,
            taxonomy_probs,
            scalars,
            p_t,
            trust_margin,
            disputed_minus_abstain,
            entropy(governance_probs),
            entropy(route_probs),
            entropy(taxonomy_probs),
            one_hot(route_pred, route.shape[1]),
            one_hot(taxonomy_pred, taxonomy.shape[1]),
        ],
        axis=1,
    ).astype(np.float32)


def runner_up_non_trustworthy(governance_logits: np.ndarray) -> np.ndarray:
    governance = _as_2d_float("governance_logits", governance_logits)
    if governance.shape[1] <= TRUSTWORTHY_ID:
        raise ValueError("governance_logits must include ABSTAIN/DISPUTED/TRUSTWORTHY columns")
    return np.where(
        governance[:, ABSTAIN_ID] >= governance[:, DISPUTED_ID],
        ABSTAIN_ID,
        DISPUTED_ID,
    ).astype(np.int64)


def apply_verifier_policy(
    *,
    base_predictions: np.ndarray,
    runner_up_predictions: np.ndarray,
    accept_scores: np.ndarray,
    verifier_threshold: float,
) -> PosthocVerifierResult:
    base = np.asarray(base_predictions, dtype=np.int64)
    runner_up = np.asarray(runner_up_predictions, dtype=np.int64)
    scores = np.asarray(accept_scores, dtype=np.float32)
    if base.ndim != 1 or runner_up.ndim != 1 or scores.ndim != 1:
        raise ValueError("base_predictions, runner_up_predictions, and accept_scores must be 1D")
    if not (base.shape[0] == runner_up.shape[0] == scores.shape[0]):
        raise ValueError(
            "policy arrays have mismatched lengths: "
            f"base={base.shape[0]} runner_up={runner_up.shape[0]} scores={scores.shape[0]}"
        )
    guarded = base.copy()
    rejected = (base == TRUSTWORTHY_ID) & (scores < float(verifier_threshold))
    guarded[rejected] = runner_up[rejected]
    return PosthocVerifierResult(
        base_predictions=base,
        guarded_predictions=guarded,
        accept_scores=scores,
        rejected_mask=rejected,
    )


class PosthocVerifierPackage:
    """Loaded Stage 0 MoE post-hoc verifier package."""

    def __init__(
        self,
        *,
        package_dir: Path,
        manifest: dict[str, Any],
        seeds: dict[int, PosthocVerifierSeed],
    ) -> None:
        self.package_dir = package_dir
        self.manifest = manifest
        self._seeds = seeds

    @classmethod
    def load(cls, package_dir: str | Path, *, verify_hashes: bool = True) -> "PosthocVerifierPackage":
        root = Path(package_dir).resolve()
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path)
        cls._validate_manifest(manifest)
        seeds: dict[int, PosthocVerifierSeed] = {}
        for entry in manifest["seeds"]:
            cls._validate_seed_entry(root, entry, verify_hashes=verify_hashes)
            verifier = joblib.load(resolve_package_path(root, entry["verifier_path"]))
            seed_id = int(entry["seed"])
            seeds[seed_id] = PosthocVerifierSeed(
                seed=seed_id,
                base_threshold=float(entry["base_threshold"]),
                verifier_threshold=float(entry["selected_threshold"]),
                verifier=verifier,
                manifest_entry=dict(entry),
            )
        return cls(package_dir=root, manifest=manifest, seeds=seeds)

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any]) -> None:
        if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported verifier package schema: "
                f"{manifest.get('schema_version')!r}"
            )
        feature_schema = manifest.get("feature_schema")
        if not isinstance(feature_schema, dict):
            raise ValueError("manifest is missing feature_schema")
        if feature_schema.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported verifier feature schema: "
                f"{feature_schema.get('schema_version')!r}"
            )
        if int(feature_schema.get("total_width", 0)) <= 0:
            raise ValueError("feature_schema.total_width must be positive")
        seeds = manifest.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("manifest must contain at least one seed")

    @staticmethod
    def _validate_seed_entry(package_dir: Path, entry: dict[str, Any], *, verify_hashes: bool) -> None:
        for key in ["seed", "base_threshold", "selected_threshold", "verifier_path"]:
            if key not in entry:
                raise ValueError(f"seed entry is missing {key!r}")
        artifacts = entry.get("artifacts", {})
        for artifact_name, path_key in [
            ("verifier", "verifier_path"),
            ("report", "report_path"),
            ("test_predictions", "test_predictions_path"),
        ]:
            if path_key not in entry:
                continue
            path = resolve_package_path(package_dir, entry[path_key])
            if not path.exists():
                raise FileNotFoundError(path)
            expected_hash = artifacts.get(artifact_name, {}).get("sha256")
            if verify_hashes and expected_hash and sha256_file(path) != expected_hash:
                raise ValueError(
                    f"sha256 mismatch for seed {entry.get('seed')} {artifact_name}: {path}"
                )

    @property
    def seed_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._seeds))

    @property
    def feature_width(self) -> int:
        return int(self.manifest["feature_schema"]["total_width"])

    def seed(self, seed: int) -> PosthocVerifierSeed:
        try:
            return self._seeds[int(seed)]
        except KeyError:
            raise KeyError(f"package has no seed {seed}; available={self.seed_ids}") from None

    def accept_scores(self, seed: int, features: np.ndarray) -> np.ndarray:
        runtime = self.seed(seed)
        matrix = _as_2d_float("features", features)
        if matrix.shape[1] != self.feature_width:
            raise ValueError(
                f"feature width mismatch: got {matrix.shape[1]}, expected {self.feature_width}"
            )
        proba = np.asarray(runtime.verifier.predict_proba(matrix), dtype=np.float32)
        if proba.ndim != 2 or proba.shape[0] != matrix.shape[0] or proba.shape[1] < 2:
            raise ValueError(f"verifier predict_proba returned invalid shape {proba.shape}")
        return proba[:, 1]

    def apply_features(
        self,
        *,
        seed: int,
        features: np.ndarray,
        governance_logits: np.ndarray,
        base_threshold: float | None = None,
        verifier_threshold: float | None = None,
    ) -> PosthocVerifierResult:
        runtime = self.seed(seed)
        threshold = runtime.base_threshold if base_threshold is None else float(base_threshold)
        guard_threshold = (
            runtime.verifier_threshold if verifier_threshold is None else float(verifier_threshold)
        )
        governance = _as_2d_float("governance_logits", governance_logits)
        scores = self.accept_scores(seed, features)
        base = gated_predictions(governance, threshold, num_classes=governance.shape[1])
        runner_up = runner_up_non_trustworthy(governance)
        return apply_verifier_policy(
            base_predictions=base,
            runner_up_predictions=runner_up,
            accept_scores=scores,
            verifier_threshold=guard_threshold,
        )

    def apply_logits(
        self,
        *,
        seed: int,
        governance_logits: np.ndarray,
        route_logits: np.ndarray,
        taxonomy_logits: np.ndarray,
        scalar_preds: np.ndarray,
        base_threshold: float | None = None,
        verifier_threshold: float | None = None,
    ) -> PosthocVerifierResult:
        features = build_posthoc_features(
            governance_logits=governance_logits,
            route_logits=route_logits,
            taxonomy_logits=taxonomy_logits,
            scalar_preds=scalar_preds,
        )
        return self.apply_features(
            seed=seed,
            features=features,
            governance_logits=governance_logits,
            base_threshold=base_threshold,
            verifier_threshold=verifier_threshold,
        )
