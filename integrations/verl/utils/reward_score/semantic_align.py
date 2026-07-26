"""veRL reward_score plugin wrapper.

Copy this file into a veRL checkout at:

    verl/utils/reward_score/semantic_align.py

Install this package in the same Python environment:

    pip install -e /path/to/RAISE

Then point veRL's custom reward configuration at this file and select
``compute_score``. The adapter accepts veRL's named ``data_source``,
``solution_str``, ``ground_truth``, and ``extra_info`` arguments. Structured
reference steps may live in either ``reward_model.ground_truth`` or
``extra_info`` in the source Parquet row.
"""

from raise_scorer.integrations import compute_score


__all__ = ["compute_score"]
