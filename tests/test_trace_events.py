import pytest

torch = pytest.importorskip("torch")

from raise_scorer.scorer import ScorerConfig, SemanticRewardScorer, WindowConfig  # noqa: E402
from raise_scorer.trace_events import (  # noqa: E402
    TraceEvent,
    TraceValidationConfig,
    render_trace_events,
    validate_trace_events,
)
from raise_scorer.workflow import (  # noqa: E402
    WorkflowEdge,
    WorkflowGroundTruth,
    WorkflowNode,
    WorkflowPotentialScorer,
    WorkflowScorer,
    WorkflowScorerConfig,
)


_CONCEPTS = (
    "inspect logs",
    "patch code",
    "run tests",
    "tests failed",
    "finish task",
)


class EventMockEmbedder:
    cache = type("Cache", (), {"stats": lambda self: {}})()

    def encode(self, texts, use_cache=True, text_types=None):
        del use_cache, text_types
        rows = []
        for text in texts:
            normalized = str(text).casefold()
            rows.append(
                [1.0 if concept in normalized else 0.0 for concept in _CONCEPTS]
            )
        return torch.nn.functional.normalize(
            torch.tensor(rows, dtype=torch.float32),
            dim=1,
        )


def _semantic_scorer():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(
        threshold=0.45,
        windows=WindowConfig(
            coarse_window=96,
            coarse_stride=32,
            fine_window=64,
            fine_stride=16,
        ),
    )
    scorer.embedder = EventMockEmbedder()
    scorer._trace_detector = None
    return scorer


def _workflow():
    return WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs"),
            WorkflowNode(
                "patch",
                "patch code",
                max_visits=2,
                preconditions={"repo": {"tests": "failing"}},
                postconditions={"repo": {"modified": True}},
            ),
            WorkflowNode("test", "run tests", max_visits=2),
            WorkflowNode("failure", "tests failed"),
            WorkflowNode(
                "done",
                "finish task",
                postconditions={"repo": {"tests": "passed"}},
            ),
        ],
        edges=[
            WorkflowEdge("inspect", "patch"),
            WorkflowEdge("patch", "test", max_traversals=2),
            WorkflowEdge("test", "done"),
            WorkflowEdge("test", "failure"),
            WorkflowEdge("failure", "patch", kind="retry"),
        ],
        start="inspect",
        terminals=["done"],
    )


def _runtime_event(event_id, action, node_id, **kwargs):
    return TraceEvent(
        event_id=event_id,
        action=action,
        node_id=node_id,
        source="runtime",
        trusted=True,
        **kwargs,
    )


def _valid_retry_events():
    return [
        _runtime_event("e1", "inspect logs", "inspect"),
        _runtime_event(
            "e2",
            "patch code",
            "patch",
            state_before={"repo": {"tests": "failing"}},
            state_after={"repo": {"modified": True}},
        ),
        _runtime_event("e3", "run tests", "test", status="failure"),
        _runtime_event("e4", "tests failed", "failure", status="failure"),
        _runtime_event(
            "e5",
            "patch code",
            "patch",
            transition_kind="retry",
            state_before={"repo": {"tests": "failing"}},
            state_after={"repo": {"modified": True}},
        ),
        _runtime_event("e6", "run tests", "test"),
        _runtime_event(
            "e7",
            "finish task",
            "done",
            state_after={"repo": {"tests": "passed"}},
        ),
    ]


def test_trace_validation_accepts_legal_retry_and_builds_state_observations():
    validation = validate_trace_events(_valid_retry_events(), _workflow())

    assert validation.valid is True
    assert validation.validity_score == pytest.approx(1.0)
    assert validation.node_path == [
        "inspect",
        "patch",
        "test",
        "failure",
        "patch",
        "test",
        "done",
    ]
    assert validation.occurrence_path[4] == "patch#2"
    assert validation.transition_counts["retry"] == 1
    assert validation.state_observations["patch#1"]["after"]["repo"]["modified"] is True
    assert validation.state_observations["patch#2"]["before"]["repo"]["tests"] == "failing"


def test_event_prefix_potential_finishes_at_full_coverage():
    scorer = WorkflowPotentialScorer(WorkflowScorer(_semantic_scorer()))

    result = scorer.score_event_prefixes(
        _valid_retry_events(),
        _workflow(),
        coarse_to_fine=False,
    )

    assert len(result.rewards) == len(_valid_retry_events())
    assert result.final_potential == pytest.approx(1.0)
    assert result.total_shaped_reward == pytest.approx(1.0)


def test_trace_validation_rejects_untrusted_illegal_and_incomplete_trace():
    events = [
        TraceEvent(
            event_id="model-claim",
            action="inspect logs",
            node_id="inspect",
            source="model",
            trusted=True,
        ),
        _runtime_event("illegal", "finish task", "done"),
    ]

    validation = validate_trace_events(events, _workflow())

    assert validation.valid is False
    assert "untrusted_event:model-claim" in validation.violations
    assert "illegal_transition:inspect->done" in validation.violations

    incomplete = validate_trace_events(
        [_runtime_event("inspect-only", "inspect logs", "inspect")],
        _workflow(),
    )
    assert "terminal_not_reached:inspect" in incomplete.violations


def test_serialized_events_cannot_self_promote_without_explicit_gateway():
    workflow = WorkflowGroundTruth(
        nodes=[WorkflowNode("done", "finish task")],
        edges=[],
        start="done",
        terminals=["done"],
    )
    payload = [
        {
            "event_id": "claimed-runtime",
            "action": "finish task",
            "node_id": "done",
            "source": "runtime",
            "trusted": True,
        }
    ]

    denied = validate_trace_events(payload, workflow)
    accepted = validate_trace_events(
        payload,
        workflow,
        TraceValidationConfig(allow_serialized_trust=True),
    )

    assert denied.valid is False
    assert "untrusted_event:claimed-runtime" in denied.violations
    assert accepted.valid is True


def test_trace_validation_enforces_node_and_edge_loop_limits():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("work", "patch code", max_visits=2),
            WorkflowNode("done", "finish task"),
        ],
        edges=[
            WorkflowEdge("work", "work", kind="retry", max_traversals=1),
            WorkflowEdge("work", "done"),
        ],
        start="work",
        terminals=["done"],
    )
    events = [
        _runtime_event("work-1", "patch code", "work", occurrence=1),
        _runtime_event(
            "work-2",
            "patch code",
            "work",
            occurrence=2,
            transition_kind="retry",
        ),
        _runtime_event(
            "work-3",
            "patch code",
            "work",
            occurrence=3,
            transition_kind="retry",
        ),
        _runtime_event("done", "finish task", "done"),
    ]

    validation = validate_trace_events(events, workflow)

    assert validation.valid is False
    assert "node_visit_limit_exceeded:work:3>2" in validation.violations
    assert "edge_traversal_limit_exceeded:work->work:2>1" in validation.violations


def test_trace_validation_tracks_recovery_and_rollback_transitions():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs"),
            WorkflowNode("failure", "tests failed"),
            WorkflowNode("recover", "patch code"),
            WorkflowNode("rollback", "patch code"),
            WorkflowNode("done", "finish task"),
        ],
        edges=[
            WorkflowEdge("inspect", "failure"),
            WorkflowEdge("failure", "recover", kind="recovery"),
            WorkflowEdge("recover", "rollback", kind="rollback"),
            WorkflowEdge("rollback", "done"),
        ],
        start="inspect",
        terminals=["done"],
    )
    events = [
        _runtime_event("inspect", "inspect logs", "inspect"),
        _runtime_event("failure", "tests failed", "failure", status="failure"),
        _runtime_event(
            "recover",
            "patch code",
            "recover",
            status="recovered",
            transition_kind="recovery",
        ),
        _runtime_event(
            "rollback",
            "patch code",
            "rollback",
            status="rolled_back",
            transition_kind="rollback",
        ),
        _runtime_event("done", "finish task", "done"),
    ]

    validation = validate_trace_events(events, workflow)

    assert validation.valid is True
    assert validation.transition_counts["recovery"] == 1
    assert validation.transition_counts["rollback"] == 1


def test_trace_rendering_is_deterministic_and_clips_large_tool_output():
    event = _runtime_event(
        "tool-1",
        "run tests",
        "test",
        tool_name="pytest",
        tool_input={"path": "tests/test_api.py"},
        tool_output="x" * 100,
        evidence="command executed",
        outcome="tests passed",
    )

    rendered = render_trace_events([event], max_field_chars=32)

    assert "source=runtime" in rendered
    assert "Action: run tests" in rendered
    assert 'Tool input: {"path": "tests/test_api.py"}' in rendered
    assert "Tool output: " + ("x" * 32) + "…" in rendered


def test_score_events_uses_runtime_path_and_automatic_state_observations():
    result = WorkflowScorer(_semantic_scorer()).score_events(
        _valid_retry_events(),
        _workflow(),
        coarse_to_fine=False,
    )

    assert result.score == pytest.approx(1.0)
    assert result.best_path == [
        "inspect",
        "patch",
        "test",
        "failure",
        "patch",
        "test",
        "done",
    ]
    assert result.state_gate_passed is True
    assert result.trace_validation.valid is True
    assert result.to_dict()["trace_validation"]["validity_score"] == pytest.approx(1.0)


def test_score_events_hard_gates_or_softly_penalizes_untrusted_events():
    events = _valid_retry_events()
    for event in events:
        event.source = "model"

    hard = WorkflowScorer(_semantic_scorer()).score_events(
        events,
        _workflow(),
        coarse_to_fine=False,
    )
    soft = WorkflowScorer(
        _semantic_scorer(),
        WorkflowScorerConfig(hard_trace_gate=False, trace_weight=0.5),
    ).score_events(
        events,
        _workflow(),
        validation_config=TraceValidationConfig(require_terminal=True),
        coarse_to_fine=False,
    )

    assert hard.semantic_score == pytest.approx(1.0)
    assert hard.score == pytest.approx(0.0)
    assert soft.trace_validation.valid is False
    assert 0.5 <= soft.score < soft.semantic_score
