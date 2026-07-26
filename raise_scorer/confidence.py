"""Backward-compatible confidence-routing imports.

New code may import from :mod:`raise_scorer.core`.
"""

from .core.confidence import (
    ConfidenceReport,
    RoutingConfig,
    assess_confidence,
    assess_from_details,
    classify_step,
    classify_steps,
    should_fallback_to_judge,
)

__all__ = [
    "ConfidenceReport",
    "RoutingConfig",
    "assess_confidence",
    "assess_from_details",
    "classify_step",
    "classify_steps",
    "should_fallback_to_judge",
]
