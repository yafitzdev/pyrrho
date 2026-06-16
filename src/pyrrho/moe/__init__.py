"""pyrrho-MoE scaffolding.

The first MoE milestone is configuration, parameter accounting, and data
plumbing. Model code lives here once the Stage 0 route prototype starts.
"""

from pyrrho.moe.config import (
    DEFAULT_SEMANTIC_EXPERT_GROUPS,
    MoEParameterCounts,
    PyrrhoMoEConfig,
)
from pyrrho.moe.inference import (
    ENSEMBLE_POLICY_NAMES,
    MoEInferenceLengths,
    MoEInferenceRuntime,
    PreparedInferenceRow,
    collate_inference_rows,
    combine_seed_prediction_rows,
    prepare_inference_row,
)
from pyrrho.moe.modeling import (
    GuardedSupportAggregatingMoEConfig,
    GuardedSupportAggregatingMoEForGovernance,
    RouteCoupledMoEConfig,
    RouteCoupledMoEForGovernance,
    SupportAggregatingMoEConfig,
    SupportAggregatingMoEForGovernance,
    TinyMoEConfig,
    TinyMoEForGovernance,
    TokenRouteCoupledMoEConfig,
    TokenRouteCoupledMoEForGovernance,
    TrustGuardedSupportAggregatingMoEConfig,
    TrustGuardedSupportAggregatingMoEForGovernance,
)
from pyrrho.moe.posthoc_verifier import (
    FEATURE_SCHEMA_VERSION,
    PACKAGE_SCHEMA_VERSION,
    PosthocVerifierPackage,
    PosthocVerifierResult,
    PosthocVerifierSeed,
    apply_verifier_policy,
    build_posthoc_features,
    feature_schema_from_config,
)
from pyrrho.moe.posthoc_policies import (
    PosthocPolicyOutput,
    majority_vote,
    trustworthy_quorum,
)
from pyrrho.moe.posthoc_thresholds import (
    guarded_predictions_at_threshold,
    non_trustworthy_fallback_from_probs,
    select_threshold_row,
    sweep_verifier_thresholds,
)
from pyrrho.moe.qwen_governance import (
    QwenMoEForGovernance,
    QwenMoEGovernanceConfig,
    last_token_pool,
)

__all__ = [
    "DEFAULT_SEMANTIC_EXPERT_GROUPS",
    "ENSEMBLE_POLICY_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "GuardedSupportAggregatingMoEConfig",
    "GuardedSupportAggregatingMoEForGovernance",
    "MoEInferenceLengths",
    "MoEInferenceRuntime",
    "MoEParameterCounts",
    "PACKAGE_SCHEMA_VERSION",
    "PreparedInferenceRow",
    "PosthocPolicyOutput",
    "PosthocVerifierPackage",
    "PosthocVerifierResult",
    "PosthocVerifierSeed",
    "PyrrhoMoEConfig",
    "QwenMoEForGovernance",
    "QwenMoEGovernanceConfig",
    "RouteCoupledMoEConfig",
    "RouteCoupledMoEForGovernance",
    "SupportAggregatingMoEConfig",
    "SupportAggregatingMoEForGovernance",
    "TokenRouteCoupledMoEConfig",
    "TokenRouteCoupledMoEForGovernance",
    "TrustGuardedSupportAggregatingMoEConfig",
    "TrustGuardedSupportAggregatingMoEForGovernance",
    "TinyMoEConfig",
    "TinyMoEForGovernance",
    "apply_verifier_policy",
    "build_posthoc_features",
    "collate_inference_rows",
    "combine_seed_prediction_rows",
    "feature_schema_from_config",
    "guarded_predictions_at_threshold",
    "last_token_pool",
    "majority_vote",
    "non_trustworthy_fallback_from_probs",
    "prepare_inference_row",
    "select_threshold_row",
    "sweep_verifier_thresholds",
    "trustworthy_quorum",
]
