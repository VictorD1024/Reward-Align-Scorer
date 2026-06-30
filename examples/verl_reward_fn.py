"""Example veRL reward function wrapper.

In a veRL checkout, this can live under:

    verl/utils/reward_score/semantic_align.py

Then configure your trainer to use that reward module, depending on your
project's reward loading convention.
"""

from reward_align_scorer.verl_adapter import compute_score


__all__ = ["compute_score"]
