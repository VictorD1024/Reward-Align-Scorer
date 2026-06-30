import pytest

torch = pytest.importorskip("torch")

from reward_align_scorer.scorer import ScorerConfig, SemanticRewardScorer
from reward_align_scorer.scorer import WindowConfig


class MockEmbedder:
    cache = type("Cache", (), {"stats": lambda self: {}})()

    def encode(self, texts, use_cache=True):
        mapping = {
            "read task": [1.0, 0.0, 0.0],
            "edit code": [0.0, 1.0, 0.0],
            "run tests": [0.0, 0.0, 1.0],
        }
        rows = []
        for text in texts:
            if "read" in text:
                rows.append(mapping["read task"])
            elif "edit" in text:
                rows.append(mapping["edit code"])
            elif "test" in text:
                rows.append(mapping["run tests"])
            else:
                rows.append([0.0, 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32)


def test_scorer_with_mock_embedder():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(threshold=0.65, windows=WindowConfig(fine_window=16, fine_stride=8))
    scorer.embedder = MockEmbedder()

    result = scorer.score(
        response="read the issue. edit the code. run tests.",
        reference_steps=["read task", "edit code", "run tests"],
        coarse_to_fine=False,
    )

    assert result.match_rate == 1.0
    assert result.order_rate == 1.0
    assert result.score == pytest.approx(1.0)
