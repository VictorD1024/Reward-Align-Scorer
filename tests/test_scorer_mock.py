import pytest

torch = pytest.importorskip("torch")

from raise_scorer.scorer import ScorerConfig, SemanticRewardScorer, WindowConfig  # noqa: E402


class MockEmbedder:
    cache = type("Cache", (), {"stats": lambda self: {}})()

    def encode(self, texts, use_cache=True, text_types=None):
        del text_types
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


class CountingMockEmbedder(MockEmbedder):
    def __init__(self):
        self.calls = []

    def encode(self, texts, use_cache=True, text_types=None):
        texts = list(texts)
        self.calls.append({"texts": texts, "text_types": list(text_types) if text_types else None})
        return super().encode(texts, use_cache=use_cache, text_types=text_types)


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


def test_scorer_reports_capped_window_coverage_stats():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(
        threshold=0.65,
        windows=WindowConfig(fine_window=48, fine_stride=24, max_windows=8),
    )
    scorer.embedder = MockEmbedder()

    response = "read the issue. " + ("filler " * 1000) + "run tests."
    result = scorer.score(
        response=response,
        reference_steps=["read task", "run tests"],
        coarse_to_fine=False,
    )

    assert result.stats["candidate_windows"] > result.stats["num_windows"]
    assert result.stats["num_windows"] == 8
    assert result.stats["windows_downsampled"] is True
    assert 0.0 < result.stats["coverage_rate"] < 1.0
    assert result.stats["tail_covered"] is True
    assert result.windows[-1][2] == len(response)


def test_score_batch_uses_one_embedding_call_for_one_pass():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(threshold=0.65, windows=WindowConfig(fine_window=16, fine_stride=8))
    scorer.embedder = CountingMockEmbedder()
    scorer._trace_detector = None

    results = scorer.score_batch(
        responses=[
            "read the issue. run tests.",
            "edit the code. run tests.",
        ],
        reference_steps_batch=[
            ["read task", "run tests"],
            ["edit code", "run tests"],
        ],
        coarse_to_fine=False,
    )

    assert len(scorer.embedder.calls) == 1
    call = scorer.embedder.calls[0]
    assert call["text_types"].count("query") == 4
    assert call["text_types"].count("passage") == len(call["text_types"]) - 4
    assert len(results) == 2
    assert all(result.score == pytest.approx(1.0) for result in results)
    assert all(result.stats["batch_size"] == 2 for result in results)
    assert all(result.stats["batch_pass_size"] == 2 for result in results)


def test_score_batch_matches_individual_scores():
    batch_scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    batch_scorer.config = ScorerConfig(threshold=0.65, windows=WindowConfig(fine_window=16, fine_stride=8))
    batch_scorer.embedder = MockEmbedder()
    batch_scorer._trace_detector = None

    single_scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    single_scorer.config = batch_scorer.config
    single_scorer.embedder = MockEmbedder()
    single_scorer._trace_detector = None

    responses = ["read the issue. run tests.", "edit the code. run tests."]
    steps_batch = [["read task", "run tests"], ["edit code", "run tests"]]
    batch_results = batch_scorer.score_batch(responses, steps_batch, coarse_to_fine=False)
    single_results = [
        single_scorer.score(response, steps, coarse_to_fine=False)
        for response, steps in zip(responses, steps_batch)
    ]

    assert [result.score for result in batch_results] == [result.score for result in single_results]
    assert [result.alignment_path for result in batch_results] == [
        result.alignment_path for result in single_results
    ]


def test_score_batch_coarse_to_fine_batches_each_pass():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(
        threshold=0.65,
        windows=WindowConfig(
            coarse_window=128,
            coarse_stride=64,
            fine_window=16,
            fine_stride=8,
        ),
    )
    scorer.embedder = CountingMockEmbedder()
    scorer._trace_detector = None

    results = scorer.score_batch(
        responses=[
            "read the issue. run tests.",
            "edit the code. run tests.",
        ],
        reference_steps_batch=[
            ["read task", "run tests"],
            ["edit code", "run tests"],
        ],
    )

    assert len(scorer.embedder.calls) == 2
    assert all(result.stats["method"] == "coarse_to_fine" for result in results)
    assert all(result.stats["batch_pass_size"] == 2 for result in results)


def test_score_batch_fine_pass_only_contains_incomplete_coarse_samples():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(
        threshold=0.65,
        windows=WindowConfig(
            coarse_window=128,
            coarse_stride=64,
            fine_window=16,
            fine_stride=8,
        ),
    )
    scorer.embedder = CountingMockEmbedder()
    scorer._trace_detector = None

    results = scorer.score_batch(
        responses=[
            "read the issue.",
            "edit the code. run tests.",
        ],
        reference_steps_batch=[
            ["read task"],
            ["edit code", "run tests"],
        ],
    )

    assert len(scorer.embedder.calls) == 2
    assert results[0].stats["method"] == "coarse"
    assert results[0].stats["batch_pass_size"] == 2
    assert results[1].stats["method"] == "coarse_to_fine"
    assert results[1].stats["batch_pass_size"] == 1


def test_score_batch_validates_input_lengths():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig()
    scorer.embedder = MockEmbedder()
    scorer._trace_detector = None

    with pytest.raises(ValueError, match="same length"):
        scorer.score_batch(["one", "two"], [["step"]])
    with pytest.raises(TypeError, match="sequence of strings"):
        scorer.score_batch("one response", [["step"]])
    assert scorer.score_batch([], []) == []


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
