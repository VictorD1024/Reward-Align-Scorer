"""Linear semantic scoring, confidence routing, and trajectory selection."""

from .confidence import (
    ConfidenceReport,
    RoutingConfig,
    assess_confidence,
    assess_from_details,
    classify_step,
    classify_steps,
    should_fallback_to_judge,
)
from .scorer import (
    Alignment,
    RewardScore,
    ScorerConfig,
    SemanticRewardScorer,
    SimilarityMatrix,
    WindowConfig,
    format_step,
    monotonic_align,
    normalize_steps,
    sliding_windows,
)
from .trace_detector import TraceDetector
from .trajectories import TrajectoryResult, score_trajectories

__all__ = [
    "Alignment",
    "ConfidenceReport",
    "RewardScore",
    "RoutingConfig",
    "ScorerConfig",
    "SemanticRewardScorer",
    "SimilarityMatrix",
    "TraceDetector",
    "TrajectoryResult",
    "WindowConfig",
    "assess_confidence",
    "assess_from_details",
    "classify_step",
    "classify_steps",
    "format_step",
    "monotonic_align",
    "normalize_steps",
    "score_trajectories",
    "should_fallback_to_judge",
    "sliding_windows",
]
