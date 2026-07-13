from raise_scorer.trace_detector import TraceDetector


def test_tool_trace_passes():
    d = TraceDetector()
    assert d.has_trace("Running <run_test>pytest x</run_test>", "run tests")
    assert d.has_trace("Changed line: result = config.get(\"d\", 42)", "modify code")
    assert d.has_trace("Output: 245 passed, 3 skipped", "run tests")


def test_reasoning_trace_passes_with_step_name_stripped():
    # The step "locate root cause" itself contains "root cause"; stripping the
    # step name first must still leave genuine extra reasoning detectable.
    d = TraceDetector()
    # Genuine: window states the root cause as an additional claim.
    assert d.has_trace("The root cause is a hardcoded initial value.", "locate root cause")
    assert d.has_trace("the bug is a hardcoded fallback", "locate root cause")
    assert d.has_trace("I changed it because the value was wrong", "modify code")


def test_padded_recitation_fails():
    d = TraceDetector()
    # Recitation windows: only intention verbs + step name, no trace of either kind.
    assert not d.has_trace("I will reproduce bug. I need to reproduce bug to confirm.", "reproduce bug")
    assert not d.has_trace(
        "I will locate root cause. The next step is to locate root cause carefully.",
        "locate root cause",
    )
    assert not d.has_trace("I intend to modify code accordingly so the defect goes away.", "modify code")
    assert not d.has_trace("I will run tests to verify. After I run tests, done.", "run tests")


def test_multi_candidate_step_all_candidates_stripped():
    d = TraceDetector()
    # Neither candidate carries a trace; stripping both leaves no reasoning signal.
    assert not d.has_trace("I will 打开网页来源 then 打开知识库来源", ["打开网页来源", "打开知识库来源"])


def test_empty_window_fails():
    d = TraceDetector()
    assert not d.has_trace("", "reproduce bug")
