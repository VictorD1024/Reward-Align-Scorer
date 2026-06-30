import pytest

torch = pytest.importorskip("torch")

from reward_align_scorer.scorer import monotonic_align


def test_monotonic_align_finds_ordered_path():
    sim = torch.tensor(
        [
            [0.90, 0.10, 0.10],
            [0.10, 0.85, 0.20],
            [0.20, 0.10, 0.88],
        ]
    )
    alignment = monotonic_align(sim, threshold=0.65)
    assert alignment.path == [(0, 0), (1, 1), (2, 2)]
    assert alignment.match_rate == 1.0


def test_monotonic_align_ignores_low_similarity():
    sim = torch.tensor(
        [
            [0.20, 0.10],
            [0.10, 0.30],
        ]
    )
    alignment = monotonic_align(sim, threshold=0.65)
    assert alignment.path == []
    assert alignment.match_rate == 0.0
