"""Backward-compatible scorer imports.

New code may import from :mod:`raise_scorer.core`.
"""

from .core.scorer import (
    Alignment,
    RewardScore,
    ScorerConfig,
    SemanticRewardScorer,
    SimilarityMatrix,
    WindowConfig,
    _sliding_windows_with_stats,
    format_step,
    monotonic_align,
    normalize_steps,
    sliding_windows,
)

_normalize_steps = normalize_steps
_step_repr = format_step

__all__ = [
    "Alignment",
    "RewardScore",
    "ScorerConfig",
    "SemanticRewardScorer",
    "SimilarityMatrix",
    "WindowConfig",
    "_normalize_steps",
    "_sliding_windows_with_stats",
    "_step_repr",
    "format_step",
    "monotonic_align",
    "normalize_steps",
    "sliding_windows",
]
