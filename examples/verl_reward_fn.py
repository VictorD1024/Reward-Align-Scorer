"""Example veRL reward function wrapper.

In a veRL checkout, this can live under:

    verl/utils/reward_score/semantic_align.py

Then configure your trainer to use that reward module. The exported function
accepts veRL's named ``data_source``, ``solution_str``, ``ground_truth``, and
``extra_info`` arguments. Reference steps can be stored in either structured
``reward_model.ground_truth`` or ``extra_info`` in the RLVR Parquet row.
"""

from raise_scorer.integrations import compute_score


__all__ = ["compute_score"]
