"""Backward-compatible trajectory imports.

New code may import from :mod:`raise_scorer.core`.
"""

from .core.trajectories import TrajectoryResult, score_trajectories

__all__ = ["TrajectoryResult", "score_trajectories"]
