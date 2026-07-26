import pytest

from raise_scorer.confidence import (
    assess_confidence,
    classify_step,
    classify_steps,
    should_fallback_to_judge,
    RoutingConfig,
)
from raise_scorer.scorer import RewardScore


def _result(**kwargs):
    defaults = dict(
        score=0.8,
        match_rate=0.8,
        order_rate=0.75,
        matched_steps=["a", "b"],
        unmatched_steps=[],
        alignment_path=[(0, 1), (1, 3)],
        windows=[],
        stats={
            "num_steps": 4,
            "mean_matched_sim": 0.78,
            "min_matched_sim": 0.72,
            "mean_margin": 0.08,
        },
    )
    defaults.update(kwargs)
    return RewardScore(**defaults)


def test_classify_step_tiers():
    assert classify_step("read the linked issue") == "proxy"
    assert classify_step("run tests to verify") == "observable"
    assert classify_step("locate the root cause") == "reasoning"


def test_classify_steps_with_candidates():
    tiers = classify_steps([["read the issue", "review the bug report"], "run tests"])
    assert tiers[0]["tier"] == "proxy"
    assert tiers[1]["tier"] == "observable"


def test_high_confidence_no_fallback():
    steps = ["summarize bug symptoms", "patch module.py", "run pytest"]
    report = assess_confidence(_result(), steps)
    assert report.confidence > 0.6
    assert report.fallback_recommended is False


def test_low_match_rate_triggers_fallback():
    steps = ["read issue", "apply fix", "run tests"]
    report = assess_confidence(_result(match_rate=0.33, score=0.2), steps)
    assert report.fallback_recommended is True
    assert any("low_match_rate" in r for r in report.reasons)


def test_trace_gate_denied_always_fallback():
    steps = ["run tests"]
    report = assess_confidence(
        _result(score=0.0, match_rate=0.0, stats={"trace_gate": "denied", "num_steps": 1}),
        steps,
    )
    assert report.confidence == 0.0
    assert report.fallback_recommended is True
    assert should_fallback_to_judge(
        _result(score=0.0, match_rate=0.0, stats={"trace_gate": "denied", "num_steps": 1}),
        steps,
    )


def test_high_proxy_fraction_triggers_fallback():
    steps = ["read the issue", "understand the bug report", "review the task"]
    report = assess_confidence(_result(stats={"num_steps": 3, "mean_matched_sim": 0.8, "mean_margin": 0.1}), steps)
    assert report.fallback_recommended is True
    assert any("proxy" in r for r in report.reasons)


def test_low_margin_triggers_fallback():
    steps = ["apply the fix", "run tests"]
    report = assess_confidence(
        _result(stats={"num_steps": 2, "mean_matched_sim": 0.75, "min_matched_sim": 0.70, "mean_margin": 0.01}),
        steps,
        RoutingConfig(min_margin=0.04),
    )
    assert report.fallback_recommended is True
    assert any("low_margin" in r for r in report.reasons)


def test_high_match_without_trace_triggers_fallback():
    steps = ["reproduce bug", "modify code", "run tests"]
    padded = (
        "First I will reproduce bug carefully. Then I will modify code. Finally I will run tests."
    )
    report = assess_confidence(
        _result(match_rate=1.0, order_rate=1.0, stats={"num_steps": 3, "mean_matched_sim": 0.82, "mean_margin": 0.04}),
        steps,
        response=padded,
    )
    assert report.fallback_recommended is True
    assert any("high_match_without_execution_trace" in r for r in report.reasons)


def test_intention_step_recitation_triggers_fallback_even_with_one_real_trace():
    steps = ["reproduce bug", "locate root cause", "modify code", "run tests"]
    partial_fabrication = (
        "I ran pytest tests/test_a.py and Output: PASSED. "
        "I will reproduce bug, locate root cause, and modify code as requested."
    )

    report = assess_confidence(
        _result(
            match_rate=1.0,
            order_rate=1.0,
            stats={"num_steps": 4, "mean_matched_sim": 0.82, "mean_margin": 0.04},
        ),
        steps,
        response=partial_fabrication,
    )

    assert report.signals["intention_recitation_fraction"] == pytest.approx(0.75)
    assert report.fallback_recommended is True
    assert any("intention_step_recitation" in reason for reason in report.reasons)


def test_completed_step_mentions_do_not_trigger_intention_recitation():
    steps = ["reproduce bug", "modify code", "run tests"]
    completed = (
        "I did reproduce bug with pytest. I then modify code in src/module.py "
        "and run tests; Output: PASSED."
    )

    report = assess_confidence(
        _result(
            match_rate=1.0,
            order_rate=1.0,
            stats={"num_steps": 3, "mean_matched_sim": 0.82, "mean_margin": 0.04},
        ),
        steps,
        response=completed,
    )

    assert report.signals["intention_recitation_fraction"] == 0.0
    assert not any("intention_step_recitation" in reason for reason in report.reasons)
