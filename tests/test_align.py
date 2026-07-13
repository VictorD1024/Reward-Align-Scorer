import pytest

torch = pytest.importorskip("torch")

from raise_scorer.scorer import monotonic_align


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
    assert alignment.order_rate == 0.0


def test_monotonic_align_detects_disorder_via_order_rate():
    # Both steps match (match_rate=1.0) but their best (argmax) windows are reversed:
    # step0's strongest match is window1, step1's strongest is window0.
    # DP still finds a monotonic path via (0,0)+(1,1), but order_rate must be 0.0.
    sim = torch.tensor(
        [
            [0.70, 0.95],
            [0.95, 0.70],
        ]
    )
    alignment = monotonic_align(sim, threshold=0.65)
    assert alignment.match_rate == 1.0
    assert alignment.order_rate == 0.0


def test_monotonic_align_partial_order_rate():
    # All three steps match via DP on the diagonal, but argmax windows are [0, 1, 0]:
    # adjacent pair (0,1) is ordered, (1,0) is not -> order_rate = 0.5.
    sim = torch.tensor(
        [
            [0.90, 0.10, 0.10],
            [0.10, 0.90, 0.10],
            [0.95, 0.10, 0.70],
        ]
    )
    alignment = monotonic_align(sim, threshold=0.65)
    assert alignment.match_rate == 1.0
    assert alignment.order_rate == 0.5
