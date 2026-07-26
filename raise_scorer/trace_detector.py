"""Backward-compatible trace detector imports.

New code may import from :mod:`raise_scorer.core`.
"""

from .core.trace_detector import (
    DEFAULT_REASONING_PATTERNS,
    DEFAULT_TOOL_PATTERNS,
    TraceDetector,
)

__all__ = [
    "DEFAULT_REASONING_PATTERNS",
    "DEFAULT_TOOL_PATTERNS",
    "TraceDetector",
]
