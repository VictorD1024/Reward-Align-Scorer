from __future__ import annotations

from dataclasses import dataclass

from .scorer import RewardScore, SemanticRewardScorer


@dataclass
class TrajectoryResult:
    """Best score over a set of valid reference trajectories.

    ``best_index`` says which trajectory won (useful to tell which branch the
    response actually took); ``result`` is the full :class:`RewardScore` for
    that trajectory; ``all_scores`` exposes every trajectory's score for
    debugging.
    """

    best_index: int
    result: RewardScore
    all_scores: list[float]

    @property
    def score(self) -> float:
        return self.result.score


def score_trajectories(
    scorer: SemanticRewardScorer,
    response: str,
    trajectories: list,
    coarse_to_fine: bool = True,
) -> TrajectoryResult:
    """Score ``response`` against several valid reference trajectories and keep
    the best.

    Each trajectory is a ``list[str | list[str]]`` passed straight to
    ``scorer.score``. Use this when a task admits multiple valid action
    sequences (branching plans) — e.g. a fix can go via
    ``["reproduce", "patch", "test"]`` or via
    ``["reproduce", "workaround", "test"]``; whichever branch the response
    took should score high.

    Cost: response windows are encoded once and reused across trajectories via
    the embedder's LRU cache, so runtime scales with the number of *distinct
    step strings*, not the number of trajectories. Step strings shared across
    trajectories (common prefixes/suffixes) are also cached.
    """
    if not trajectories:
        raise ValueError("trajectories must be a non-empty list")
    results = [scorer.score(response, t, coarse_to_fine=coarse_to_fine) for t in trajectories]
    all_scores = [r.score for r in results]
    best_index = max(
        range(len(results)),
        key=lambda i: (results[i].score, results[i].match_rate, results[i].order_rate),
    )
    return TrajectoryResult(best_index=best_index, result=results[best_index], all_scores=all_scores)
