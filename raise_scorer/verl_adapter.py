"""Backward-compatible veRL integration imports.

New code may import from :mod:`raise_scorer.integrations`.
"""

from .integrations.ground_truth import (
    ReferenceStepsResolution,
    normalize_reference_steps,
    resolve_reference_steps,
)
from .integrations.verl import compute_score, parse_reference_steps

__all__ = [
    "ReferenceStepsResolution",
    "compute_score",
    "normalize_reference_steps",
    "parse_reference_steps",
    "resolve_reference_steps",
]
