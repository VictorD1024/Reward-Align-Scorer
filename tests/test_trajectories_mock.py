import pytest

torch = pytest.importorskip("torch")

from raise_scorer.scorer import ScorerConfig, SemanticRewardScorer, WindowConfig
from raise_scorer.trajectories import score_trajectories


class MockEmbedder:
    cache = type("Cache", (), {"stats": lambda self: {}})()

    def encode(self, texts, use_cache=True):
        rows = []
        for text in texts:
            if "read" in text:
                rows.append([1.0, 0.0, 0.0])
            elif "edit" in text:
                rows.append([0.0, 1.0, 0.0])
            elif "test" in text:
                rows.append([0.0, 0.0, 1.0])
            else:
                rows.append([0.0, 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32)


def _scorer():
    s = SemanticRewardScorer.__new__(SemanticRewardScorer)
    s.config = ScorerConfig(threshold=0.65, windows=WindowConfig(fine_window=16, fine_stride=8))
    s.embedder = MockEmbedder()
    return s


def test_score_trajectories_picks_branch_the_response_took():
    # Two valid branches: A reads+edits, B reads+tests. The response reads and
    # tests, so branch B should win (full match) over A (half match).
    scorer = _scorer()
    trajectories = [
        ["read task", "edit code"],
        ["read task", "run tests"],
    ]
    res = score_trajectories(
        scorer,
        response="read the issue. run tests.",
        trajectories=trajectories,
        coarse_to_fine=False,
    )
    assert res.best_index == 1
    assert res.score == pytest.approx(1.0)
    assert res.all_scores == [pytest.approx(0.5), pytest.approx(1.0)]


def test_score_trajectories_single_trajectory_passthrough():
    scorer = _scorer()
    res = score_trajectories(
        scorer,
        response="read the issue. run tests.",
        trajectories=[["read task", "run tests"]],
        coarse_to_fine=False,
    )
    assert res.best_index == 0
    assert res.score == pytest.approx(1.0)


def test_score_trajectories_rejects_empty():
    scorer = _scorer()
    with pytest.raises(ValueError):
        score_trajectories(scorer, response="x", trajectories=[], coarse_to_fine=False)
