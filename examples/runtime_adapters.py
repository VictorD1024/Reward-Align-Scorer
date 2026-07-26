import os

from raise_scorer.backends import load_embedding_backend
from raise_scorer.core import ScorerConfig, SemanticRewardScorer
from raise_scorer.runtime import RuntimeEventAdapter
from raise_scorer.workflows import (
    WorkflowGroundTruth,
    WorkflowScorer,
)


def main():
    model = os.environ.get("RAISE_MODEL_PATH", "intfloat/multilingual-e5-small")
    backend = load_embedding_backend(
        model,
        local_files_only=os.environ.get("RAISE_LOCAL_FILES_ONLY") == "1",
    )
    if backend is None:
        raise RuntimeError(
            "Set RAISE_MODEL_PATH to a downloaded embedding model or Hub model ID."
        )

    workflow = WorkflowGroundTruth.from_dict(
        {
            "name": "runtime-adapter-demo",
            "start": "inspect",
            "terminals": ["verify"],
            "nodes": [
                {
                    "id": "inspect",
                    "action": "read file src/parser.py",
                    "evidence": "file path: src/parser.py",
                    "outcome": "file contents observed",
                },
                {
                    "id": "patch",
                    "action": "modify file src/parser.py",
                    "evidence": "file path: src/parser.py",
                    "outcome": "file modified",
                    "postconditions": {"repo": {"modified": True}},
                },
                {
                    "id": "verify",
                    "action": "run tests",
                    "evidence": "pytest result: 12 passed",
                    "outcome": "tests pass",
                    "postconditions": {"repo": {"tests": "passed"}},
                },
            ],
            "edges": [
                {"from": "inspect", "to": "patch"},
                {"from": "patch", "to": "verify"},
            ],
        }
    )

    tool_calls = [
        {
            "id": "inspect-1",
            "tool": "read_file",
            "arguments": {"path": "src/parser.py"},
            "node_id": "inspect",
        },
        {
            "id": "patch-1",
            "tool": "apply_patch",
            "arguments": {
                "path": "src/parser.py",
                "patch": "fix parser boundary",
                "api_key": "redacted before scoring",
            },
            "node_id": "patch",
            "state_after": {"repo": {"modified": True}},
        },
        {
            "id": "verify-1",
            "tool": "exec_command",
            "arguments": {"cmd": "python -m pytest tests/test_parser.py -q"},
            "output": {"stdout": "12 passed in 0.42s"},
            "exit_code": 0,
            "node_id": "verify",
            "state_after": {"repo": {"tests": "passed"}},
        },
    ]

    # This trust decision belongs to rollout infrastructure, never model data.
    events = RuntimeEventAdapter(trusted_runtime=True).adapt_many(tool_calls)
    semantic_scorer = SemanticRewardScorer(
        backend,
        ScorerConfig(threshold=0.80),
    )
    result = WorkflowScorer(semantic_scorer).score_events(events, workflow)

    print("score:", result.score)
    print("path:", result.trace_validation.node_path)
    print("trace valid:", result.trace_validation.valid)
    print("redacted api_key:", events[1].tool_input["api_key"])


if __name__ == "__main__":
    main()
