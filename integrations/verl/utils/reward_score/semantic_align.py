"""veRL reward_score plugin wrapper.

Copy this file into a veRL checkout at:

    verl/utils/reward_score/semantic_align.py

Install this package in the same Python environment:

    pip install -e /path/to/reward-align-scorer

Then point your reward configuration at the `semantic_align` reward module
according to your veRL project's reward loading convention.
"""

from reward_align_scorer.verl_adapter import compute_score


__all__ = ["compute_score"]
