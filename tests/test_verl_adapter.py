from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

import raise_scorer.integrations.verl as verl_adapter
from raise_scorer.scorer import RewardScore
from raise_scorer.integrations import parse_reference_steps


def test_canonical_key():
    assert parse_reference_steps({"reference_steps": ["a", "b"]}) == ["a", "b"]


def test_actions_alias():
    assert parse_reference_steps({"actions": ["x", "y"]}) == ["x", "y"]


def test_priority_reference_over_actions():
    assert parse_reference_steps({"reference_steps": ["a"], "actions": ["b"]}) == ["a"]


def test_string_input_splits_on_punctuation():
    assert parse_reference_steps({"reference_steps": "a, b；c\nd"}) == ["a", "b", "c", "d"]


def test_nested_candidate_lists_preserved():
    # A step may be a list of candidate actions; any one matching credits the step.
    assert parse_reference_steps({"reference_steps": ["search", ["open web", "open kb"], "answer"]}) == [
        "search",
        ["open web", "open kb"],
        "answer",
    ]


def test_empty_and_missing():
    assert parse_reference_steps({}) == []
    assert parse_reference_steps({"reference_steps": []}) == []


def test_compute_score_accepts_current_verl_named_interface(monkeypatch):
    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: None)

    score = verl_adapter.compute_score(
        data_source="demo/agent-trajectories",
        solution_str="run tests",
        ground_truth=None,
        extra_info={"reference_steps": ["run tests"]},
        framework_option="accepted via reward_kwargs",
    )

    assert score == pytest.approx(1.0)


def test_compute_score_keeps_original_direct_call_api(monkeypatch):
    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: None)

    details = verl_adapter.compute_score(
        "run tests",
        extra_info={"reference_steps": ["run tests"]},
        return_details=True,
    )

    assert details["score"] == pytest.approx(1.0)
    assert details["data_source"] is None


def test_compute_score_rejects_non_dict_extra_info():
    with pytest.raises(TypeError, match="extra_info must be a dict"):
        verl_adapter.compute_score("answer", extra_info="not-a-dict")


def test_get_scorer_forwards_embedding_environment_options(monkeypatch):
    captured = {}

    def fake_load(model_path, **kwargs):
        captured["model_path"] = model_path
        captured["backend_kwargs"] = kwargs
        return object()

    def fake_scorer(backend, config):
        captured["backend"] = backend
        captured["config"] = config
        return "configured-scorer"

    monkeypatch.setattr(verl_adapter, "_SCORER", None)
    monkeypatch.setattr(verl_adapter, "load_embedding_backend", fake_load)
    monkeypatch.setattr(verl_adapter, "SemanticRewardScorer", fake_scorer)
    monkeypatch.setenv("RAISE_MODEL_PATH", "intfloat/multilingual-e5-small")
    monkeypatch.setenv("RAISE_POOLING", "mean")
    monkeypatch.setenv("RAISE_QUERY_INSTRUCTION", "")
    monkeypatch.setenv("RAISE_PASSAGE_INSTRUCTION", "passage: ")
    monkeypatch.setenv("RAISE_MODEL_REVISION", "pinned-commit")
    monkeypatch.setenv("RAISE_LOCAL_FILES_ONLY", "1")
    monkeypatch.setenv("RAISE_THRESHOLD", "0.8")

    scorer = verl_adapter._get_scorer()

    assert scorer == "configured-scorer"
    assert captured["model_path"] == "intfloat/multilingual-e5-small"
    assert captured["backend_kwargs"] == {
        "pooling": "mean",
        "query_instruction": "",
        "passage_instruction": "passage: ",
        "revision": "pinned-commit",
        "local_files_only": True,
    }
    assert captured["config"].threshold == pytest.approx(0.8)


def test_compute_score_accepts_workflow_ground_truth(monkeypatch):
    semantic = RewardScore(
        score=1.0,
        match_rate=1.0,
        order_rate=1.0,
        matched_steps=["run tests"],
        unmatched_steps=[],
        alignment_path=[(0, 0)],
        windows=[("run tests Output: passed", 0, 24)],
        stats={
            "num_steps": 1,
            "mean_matched_sim": 0.9,
            "min_matched_sim": 0.9,
            "mean_margin": 0.2,
        },
    )
    workflow_result = SimpleNamespace(
        score=0.8,
        required_coverage=0.8,
        order_rate=1.0,
        semantic_result=semantic,
        node_rewards=[
            SimpleNamespace(
                required=True,
                fields={"action": SimpleNamespace(reference="run tests")},
            )
        ],
        best_path=["verify"],
        candidate_paths=1,
        paths_truncated=False,
        state_score=1.0,
        state_gate_passed=True,
        trace_validation=None,
        to_dict=lambda: {"score": 0.8, "best_path": ["verify"]},
    )

    class FakeWorkflowScorer:
        def __init__(self, scorer, config):
            assert scorer == "semantic-scorer"
            assert config is None

        def score(self, response, workflow, state_observations=None):
            assert response == "run tests Output: passed"
            assert workflow.start == "verify"
            assert state_observations == {"verify": {"after": {"tests": "passed"}}}
            return workflow_result

    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: "semantic-scorer")
    monkeypatch.setattr(verl_adapter, "WorkflowScorer", FakeWorkflowScorer)

    details = verl_adapter.compute_score(
        solution_str="run tests Output: passed",
        extra_info={
            "workflow_ground_truth": {
                "start": "verify",
                "terminals": ["verify"],
                "nodes": [{"id": "verify", "action": "run tests"}],
                "edges": [],
            },
            "state_observations": {"verify": {"after": {"tests": "passed"}}},
        },
        return_details=True,
    )

    assert details["score"] == pytest.approx(0.8)
    assert details["workflow"]["best_path"] == ["verify"]
    assert details["debug"]["workflow_state_gate_passed"] is True


def test_compute_score_accepts_trusted_trace_events_without_solution_text(monkeypatch):
    semantic = RewardScore(
        score=1.0,
        match_rate=1.0,
        order_rate=1.0,
        matched_steps=["run tests"],
        unmatched_steps=[],
        alignment_path=[(0, 0)],
        windows=[("Action: run tests", 0, 17)],
        stats={
            "num_steps": 1,
            "mean_matched_sim": 0.9,
            "min_matched_sim": 0.9,
            "mean_margin": 0.2,
        },
    )
    trace_validation = SimpleNamespace(valid=True)
    workflow_result = SimpleNamespace(
        score=1.0,
        required_coverage=1.0,
        order_rate=1.0,
        semantic_result=semantic,
        node_rewards=[
            SimpleNamespace(
                required=True,
                fields={"action": SimpleNamespace(reference="run tests")},
            )
        ],
        best_path=["verify"],
        candidate_paths=1,
        paths_truncated=False,
        state_score=1.0,
        state_gate_passed=True,
        trace_validation=trace_validation,
        to_dict=lambda: {
            "score": 1.0,
            "best_path": ["verify"],
            "trace_validation": {"valid": True},
        },
    )

    class FakeWorkflowScorer:
        def __init__(self, scorer, config):
            assert scorer == "semantic-scorer"
            assert config is None

        def score_events(
            self,
            events,
            workflow,
            validation_config=None,
            state_observations=None,
        ):
            assert events[0]["event_id"] == "runtime-test"
            assert workflow.start == "verify"
            assert validation_config.require_trusted is True
            assert validation_config.allow_serialized_trust is True
            assert state_observations is None
            return workflow_result

    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: "semantic-scorer")
    monkeypatch.setattr(verl_adapter, "WorkflowScorer", FakeWorkflowScorer)

    details = verl_adapter.compute_score(
        solution_str="",
        extra_info={
            "workflow_ground_truth": {
                "start": "verify",
                "terminals": ["verify"],
                "nodes": [{"id": "verify", "action": "run tests"}],
                "edges": [],
            },
            "trace_events": [
                {
                    "event_id": "runtime-test",
                    "action": "run tests",
                    "node_id": "verify",
                    "source": "runtime",
                    "trusted": True,
                }
            ],
            "trace_validation_config": {"allow_serialized_trust": True},
        },
        return_details=True,
    )

    assert details["score"] == pytest.approx(1.0)
    assert details["debug"]["workflow_trace_valid"] is True


def test_compute_score_adapts_trusted_tool_calls_without_solution_text(monkeypatch):
    semantic = RewardScore(
        score=1.0,
        match_rate=1.0,
        order_rate=1.0,
        matched_steps=["run tests"],
        unmatched_steps=[],
        alignment_path=[(0, 0)],
        windows=[("Action: run tests", 0, 17)],
        stats={
            "num_steps": 1,
            "mean_matched_sim": 0.9,
            "min_matched_sim": 0.9,
            "mean_margin": 0.2,
        },
    )
    trace_validation = SimpleNamespace(valid=True)
    workflow_result = SimpleNamespace(
        score=1.0,
        required_coverage=1.0,
        order_rate=1.0,
        semantic_result=semantic,
        node_rewards=[
            SimpleNamespace(
                required=True,
                fields={"action": SimpleNamespace(reference="run tests")},
            )
        ],
        best_path=["verify"],
        candidate_paths=1,
        paths_truncated=False,
        state_score=1.0,
        state_gate_passed=True,
        trace_validation=trace_validation,
        to_dict=lambda: {
            "score": 1.0,
            "best_path": ["verify"],
            "trace_validation": {"valid": True},
        },
    )

    class FakeWorkflowScorer:
        def __init__(self, scorer, config):
            assert scorer == "semantic-scorer"
            assert config is None

        def score_events(
            self,
            events,
            workflow,
            validation_config=None,
            state_observations=None,
        ):
            assert events[0].event_id == "runtime-test"
            assert events[0].action == "run tests"
            assert events[0].trusted is True
            assert events[0].tool_input["token"] == "[REDACTED]"
            assert workflow.start == "verify"
            assert validation_config.require_trusted is True
            assert validation_config.allow_serialized_trust is False
            assert state_observations is None
            return workflow_result

    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: "semantic-scorer")
    monkeypatch.setattr(verl_adapter, "WorkflowScorer", FakeWorkflowScorer)

    details = verl_adapter.compute_score(
        solution_str="",
        extra_info={
            "workflow_ground_truth": {
                "start": "verify",
                "terminals": ["verify"],
                "nodes": [{"id": "verify", "action": "run tests"}],
                "edges": [],
            },
            "tool_calls": [
                {
                    "id": "runtime-test",
                    "tool": "pytest",
                    "arguments": {"cmd": "pytest -q", "token": "secret"},
                    "output": "3 passed",
                    "node_id": "verify",
                }
            ],
            "tool_adapter_config": {"trusted_runtime": True},
        },
        return_details=True,
    )

    assert details["score"] == pytest.approx(1.0)
    assert details["debug"]["workflow_trace_valid"] is True
    assert details["debug"]["workflow_tool_calls_adapted"] is True


def test_compute_score_rejects_non_dict_tool_adapter_config(monkeypatch):
    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: "semantic-scorer")

    with pytest.raises(TypeError, match="tool_adapter_config must be a dict"):
        verl_adapter.compute_score(
            "",
            extra_info={
                "workflow_ground_truth": {
                    "start": "verify",
                    "terminals": ["verify"],
                    "nodes": [{"id": "verify", "action": "run tests"}],
                    "edges": [],
                },
                "tool_calls": [],
                "tool_adapter_config": "trusted",
            },
        )
