import os

from raise_scorer.backends import load_embedding_backend
from raise_scorer.core import ScorerConfig, SemanticRewardScorer
from raise_scorer.runtime import TraceEvent
from raise_scorer.workflows import (
    WorkflowGroundTruth,
    WorkflowScorer,
    WorkflowScorerConfig,
)


def main():
    model = os.environ.get("RAISE_MODEL_PATH", "intfloat/multilingual-e5-small")
    backend = load_embedding_backend(
        model,
        local_files_only=os.environ.get("RAISE_LOCAL_FILES_ONLY") == "1",
    )
    if backend is None:
        raise RuntimeError("Set RAISE_MODEL_PATH to a downloaded embedding model or Hub model ID.")

    semantic_scorer = SemanticRewardScorer(backend, ScorerConfig(threshold=0.80))
    workflow = WorkflowGroundTruth.from_dict(
        {
            "name": "bounded-repo-repair",
            "start": "inspect",
            "terminals": ["done"],
            "stage_weights": {
                "diagnosis": 1.0,
                "repair": 2.0,
                "verification": 2.0,
            },
            "nodes": [
                {
                    "id": "inspect",
                    "stage": "diagnosis",
                    "action": ["inspect the failing code", "inspect the relevant files"],
                    "evidence": "identify the affected file and failing behavior",
                },
                {
                    "id": "patch",
                    "stage": "repair",
                    "action": ["patch the implementation", "apply a workaround"],
                    "evidence": "show the concrete code change",
                    "max_visits": 2,
                    "preconditions": {"repo": {"tests": "failing"}},
                    "postconditions": {"repo": {"modified": True}},
                },
                {
                    "id": "test",
                    "stage": "verification",
                    "action": "run the affected tests",
                    "max_visits": 2,
                },
                {
                    "id": "failure",
                    "stage": "verification",
                    "action": "tests still fail",
                },
                {
                    "id": "done",
                    "stage": "verification",
                    "action": ["tests pass", "test suite passes"],
                    "outcome": "summarize the verified fix",
                    "postconditions": {"repo": {"tests": "passed"}},
                },
            ],
            "edges": [
                {"from": "inspect", "to": "patch"},
                {"from": "patch", "to": "test", "max_traversals": 2},
                {"from": "test", "to": "done"},
                {"from": "test", "to": "failure"},
                {"from": "failure", "to": "patch", "kind": "retry"},
            ],
        }
    )

    events = [
        TraceEvent(
            event_id="inspect-1",
            node_id="inspect",
            action="inspect the failing code",
            evidence="identify the affected file and failing behavior",
            source="runtime",
            trusted=True,
            tool_name="read_file",
            tool_output={"path": "src/parser.py", "status": "read"},
        ),
        TraceEvent(
            event_id="patch-1",
            node_id="patch",
            action="patch the implementation",
            evidence="show the concrete code change",
            source="runtime",
            trusted=True,
            tool_name="apply_patch",
            state_before={"repo": {"tests": "failing"}},
            state_after={"repo": {"modified": True}},
        ),
        TraceEvent(
            event_id="test-1",
            node_id="test",
            action="run the affected tests",
            status="failure",
            source="runtime",
            trusted=True,
            tool_name="pytest",
            tool_output="1 failed",
        ),
        TraceEvent(
            event_id="failure-1",
            node_id="failure",
            action="tests still fail",
            status="failure",
            source="runtime",
            trusted=True,
        ),
        TraceEvent(
            event_id="patch-2",
            node_id="patch",
            action="patch the implementation",
            evidence="show the concrete code change",
            transition_kind="retry",
            source="runtime",
            trusted=True,
            tool_name="apply_patch",
            state_before={"repo": {"tests": "failing"}},
            state_after={"repo": {"modified": True}},
        ),
        TraceEvent(
            event_id="test-2",
            node_id="test",
            action="run the affected tests",
            source="runtime",
            trusted=True,
            tool_name="pytest",
            tool_output="245 passed",
        ),
        TraceEvent(
            event_id="done-1",
            node_id="done",
            action="test suite passes",
            outcome="summarize the verified fix",
            source="runtime",
            trusted=True,
            state_after={"repo": {"tests": "passed"}},
        ),
    ]
    result = WorkflowScorer(
        semantic_scorer,
        WorkflowScorerConfig(state_weight=0.25),
    ).score_events(
        events,
        workflow,
    )

    print("score:", result.score)
    print("best path:", result.best_path)
    print("transition counts:", result.transition_counts)
    print("stage rewards:", result.stage_rewards)
    print("state gate passed:", result.state_gate_passed)
    print("trace valid:", result.trace_validation.valid)


if __name__ == "__main__":
    main()
