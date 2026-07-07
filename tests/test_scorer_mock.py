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


def test_scorer_multi_candidate_step_credits_on_any_match():
    # Step 0 has two candidate actions; only "read task" is realizable in the
    # response. The step should still be credited, and matched_steps should
    # report the candidate that actually matched.
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(threshold=0.65, windows=WindowConfig(fine_window=16, fine_stride=8))
    scorer.embedder = MockEmbedder()

    result = scorer.score(
        response="read the issue. run tests.",
        reference_steps=[["read task", "edit code"], "run tests"],
        coarse_to_fine=False,
    )

    assert result.match_rate == 1.0
    assert "read task" in result.matched_steps
    # The non-realizable candidate must not be reported as the match.
    assert "edit code" not in result.matched_steps
    assert result.unmatched_steps == []


def test_scorer_multi_candidate_step_unmatched_reports_all_options():
    # No candidate of step 0 is realizable -> step is unmatched and reported
    # as the joined candidate list.
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(threshold=0.65, windows=WindowConfig(fine_window=16, fine_stride=8))
    scorer.embedder = MockEmbedder()

    result = scorer.score(
        response="read the issue. run tests.",
        reference_steps=[["edit code", "refactor module"], "run tests"],
        coarse_to_fine=False,
    )

    assert result.match_rate == pytest.approx(0.5)
    assert "edit code | refactor module" in result.unmatched_steps


def _make_scorer(require_trace: bool) -> SemanticRewardScorer:
    s = SemanticRewardScorer.__new__(SemanticRewardScorer)
    s.config = ScorerConfig(
        threshold=0.65,
        require_trace=require_trace,
        windows=WindowConfig(fine_window=16, fine_stride=8),
    )
    s.embedder = MockEmbedder()
    s._trace_detector = None
    return s


def test_scorer_require_trace_kills_padded_recitation():
    # Padded recitation: restates each step with intention verbs, no tool calls
    # and no causal analysis. The embedding CAN map these windows to the steps
    # (proved by the require_trace=False case matching), so it is the trace
    # gate — not the embedder — that denies the recitation.
    recitation = "I will read task. I will edit code. I will run tests."
    steps = ["read task", "edit code", "run tests"]

    gated = _make_scorer(require_trace=True).score(recitation, steps, coarse_to_fine=False)
    ungated = _make_scorer(require_trace=False).score(recitation, steps, coarse_to_fine=False)

    assert gated.match_rate == 0.0
    assert ungated.match_rate > 0.0


def test_scorer_require_trace_keeps_genuine_with_evidence():
    # Each step's window carries evidence: reasoning ("because") for read,
    # tool tag (<edit_file>) for edit, output marker (Output:) for test.
    genuine = "read because bug found. edit <edit_file> patch. test Output: passed."
    steps = ["read task", "edit code", "run tests"]

    gated = _make_scorer(require_trace=True).score(genuine, steps, coarse_to_fine=False)
    ungated = _make_scorer(require_trace=False).score(genuine, steps, coarse_to_fine=False)

    # Trace evidence keeps the gated score at least as high as ungated coverage
    # (no step is falsely denied), and strictly above the recitation's 0.0.
    assert gated.match_rate > 0.0
    assert gated.match_rate >= ungated.match_rate - 1e-6
