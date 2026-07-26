import pytest

torch = pytest.importorskip("torch")

from raise_scorer.scorer import ScorerConfig, SemanticRewardScorer, WindowConfig  # noqa: E402
from raise_scorer.workflow import (  # noqa: E402
    PotentialRewardConfig,
    WorkflowEdge,
    WorkflowGroundTruth,
    WorkflowNode,
    WorkflowPotentialScorer,
    WorkflowScorer,
    WorkflowScorerConfig,
    build_bounded_workflow_graph,
    enumerate_workflow_paths,
)


_CONCEPTS = (
    "inspect logs",
    "patch code",
    "add workaround",
    "run tests",
    "write report",
    "diff emitted",
    "bug removed",
    "tests failed",
    "recover workspace",
    "rollback change",
    "finish task",
)


class WorkflowMockEmbedder:
    cache = type("Cache", (), {"stats": lambda self: {}})()

    def encode(self, texts, use_cache=True, text_types=None):
        del use_cache, text_types
        rows = []
        for text in texts:
            normalized = str(text).casefold()
            rows.append(
                [1.0 if concept in normalized else 0.0 for concept in _CONCEPTS]
            )
        tensor = torch.tensor(rows, dtype=torch.float32)
        return torch.nn.functional.normalize(tensor, dim=1)


def _semantic_scorer():
    scorer = SemanticRewardScorer.__new__(SemanticRewardScorer)
    scorer.config = ScorerConfig(
        threshold=0.45,
        windows=WindowConfig(
            coarse_window=48,
            coarse_stride=16,
            fine_window=40,
            fine_stride=12,
        ),
    )
    scorer.embedder = WorkflowMockEmbedder()
    scorer._trace_detector = None
    return scorer


def _trace(*events):
    separator = " neutral " * 6
    return separator.join(f"{event}." for event in events)


def test_workflow_from_dict_and_validation():
    workflow = WorkflowGroundTruth.from_dict(
        {
            "name": "branching-fix",
            "start": "inspect",
            "terminals": ["verify"],
            "nodes": [
                {"id": "inspect", "action": "inspect logs", "stage": "diagnosis"},
                {"id": "patch", "action": "patch code", "stage": "repair"},
                {"id": "verify", "action": "run tests", "stage": "verification"},
            ],
            "edges": [
                {"from": "inspect", "to": "patch"},
                {"from": "patch", "to": "verify"},
            ],
        }
    )

    assert workflow.name == "branching-fix"
    assert workflow.edges[0].source == "inspect"
    assert workflow.edges[0].target == "patch"

    with pytest.raises(ValueError, match="unknown node"):
        WorkflowGroundTruth(
            nodes=[WorkflowNode("start", "inspect logs")],
            edges=[WorkflowEdge("start", "missing")],
            start="start",
            terminals=["start"],
        )


def test_path_enumeration_bounds_retry_loop():
    workflow = _retry_workflow()

    enumeration = enumerate_workflow_paths(workflow)
    capped = enumerate_workflow_paths(workflow, max_paths=1)

    assert [path.node_ids for path in enumeration.paths] == [
        ("inspect", "patch", "test", "done"),
        ("inspect", "patch", "test", "failure", "patch", "test", "done"),
    ]
    assert enumeration.truncated is False
    assert len(capped.paths) == 1
    assert capped.truncated is True


def test_workflow_scorer_picks_the_matching_alternative_branch():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs", stage="diagnosis"),
            WorkflowNode("patch", "patch code", stage="repair"),
            WorkflowNode("workaround", "add workaround", stage="repair"),
            WorkflowNode("verify", "run tests", stage="verification"),
        ],
        edges=[
            WorkflowEdge("inspect", "patch", kind="alternative"),
            WorkflowEdge("inspect", "workaround", kind="alternative"),
            WorkflowEdge("patch", "verify"),
            WorkflowEdge("workaround", "verify"),
        ],
        start="inspect",
        terminals=["verify"],
    )

    result = WorkflowScorer(_semantic_scorer()).score(
        _trace("inspect logs", "add workaround", "run tests"),
        workflow,
        coarse_to_fine=False,
    )

    assert result.best_path == ["inspect", "workaround", "verify"]
    assert result.score == pytest.approx(1.0)
    assert result.transition_counts == {"alternative": 1, "forward": 1}
    assert result.alignment_mode == "graph"
    assert result.graph_states == 4
    assert result.graph_terminal_states == 1


def test_graph_and_enumeration_modes_are_comparable_baselines():
    workflow = _retry_workflow()
    response = _trace("inspect logs", "patch code", "run tests", "finish task")

    graph = WorkflowScorer(
        _semantic_scorer(),
        WorkflowScorerConfig(alignment_mode="graph"),
    ).score(response, workflow, coarse_to_fine=False)
    enumeration = WorkflowScorer(
        _semantic_scorer(),
        WorkflowScorerConfig(alignment_mode="enumerate"),
    ).score(response, workflow, coarse_to_fine=False)

    assert graph.best_path == enumeration.best_path
    assert graph.score == pytest.approx(enumeration.score)
    assert graph.all_path_scores == [pytest.approx(1.0)]
    assert len(enumeration.all_path_scores) == 2


def test_graph_route_uses_evidence_when_branch_actions_are_identical():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs"),
            WorkflowNode("patch_with_diff", "patch code", evidence="diff emitted"),
            WorkflowNode("patch_with_report", "patch code", evidence="write report"),
            WorkflowNode("done", "finish task"),
        ],
        edges=[
            WorkflowEdge("inspect", "patch_with_diff", kind="alternative"),
            WorkflowEdge("inspect", "patch_with_report", kind="alternative"),
            WorkflowEdge("patch_with_diff", "done"),
            WorkflowEdge("patch_with_report", "done"),
        ],
        start="inspect",
        terminals=["done"],
    )

    result = WorkflowScorer(_semantic_scorer()).score(
        _trace("inspect logs", "patch code", "diff emitted", "finish task"),
        workflow,
        coarse_to_fine=False,
    )

    assert result.best_path == ["inspect", "patch_with_diff", "done"]
    assert result.score == pytest.approx(1.0)


def test_bounded_graph_respects_retry_limits_without_path_enumeration():
    graph = build_bounded_workflow_graph(_retry_workflow())

    assert len(graph.terminal_states) == 2
    assert graph.truncated is False
    assert max(state.node_visits[1] for state in graph.states) == 2


def test_optional_node_does_not_reduce_required_reward():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs", stage="diagnosis"),
            WorkflowNode("document", "write report", stage="report", required=False),
            WorkflowNode("verify", "run tests", stage="verification"),
        ],
        edges=[
            WorkflowEdge("inspect", "document"),
            WorkflowEdge("document", "verify"),
        ],
        start="inspect",
        terminals=["verify"],
    )

    result = WorkflowScorer(_semantic_scorer()).score(
        _trace("inspect logs", "run tests"),
        workflow,
        coarse_to_fine=False,
    )

    assert result.score == pytest.approx(1.0)
    assert result.required_coverage == pytest.approx(1.0)
    assert result.optional_coverage == pytest.approx(0.0)
    assert result.stage_rewards["report"].required_coverage is None
    assert result.stage_rewards["report"].optional_coverage == pytest.approx(0.0)


def test_action_evidence_outcome_and_stage_rewards_are_reported():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode(
                "patch",
                action=["patch code", "add workaround"],
                evidence="diff emitted",
                outcome="bug removed",
                stage="repair",
            )
        ],
        edges=[],
        start="patch",
        terminals=["patch"],
    )

    result = WorkflowScorer(_semantic_scorer()).score(
        _trace("add workaround", "diff emitted"),
        workflow,
        coarse_to_fine=False,
    )

    node = result.node_rewards[0]
    assert node.fields["action"].matched is True
    assert node.fields["evidence"].matched is True
    assert node.fields["outcome"].matched is False
    assert node.score == pytest.approx(0.75)
    assert result.stage_rewards["repair"].score == pytest.approx(0.75)
    assert result.score == pytest.approx(0.75)


def test_stage_weights_control_hierarchical_required_coverage():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs", stage="diagnosis"),
            WorkflowNode("verify", "run tests", stage="verification"),
        ],
        edges=[WorkflowEdge("inspect", "verify")],
        start="inspect",
        terminals=["verify"],
        stage_weights={"diagnosis": 1.0, "verification": 3.0},
    )

    result = WorkflowScorer(_semantic_scorer()).score(
        "inspect logs.",
        workflow,
        coarse_to_fine=False,
    )

    assert result.stage_rewards["diagnosis"].score == pytest.approx(1.0)
    assert result.stage_rewards["verification"].score == pytest.approx(0.0)
    assert result.required_coverage == pytest.approx(0.25)
    assert result.score == pytest.approx(0.25)


def test_retry_path_wins_when_failure_and_repeated_actions_are_observed():
    workflow = _retry_workflow()
    response = _trace(
        "inspect logs",
        "patch code",
        "run tests",
        "tests failed",
        "patch code",
        "run tests",
        "finish task",
    )

    result = WorkflowScorer(_semantic_scorer()).score(
        response,
        workflow,
        coarse_to_fine=False,
    )

    assert result.best_path == [
        "inspect",
        "patch",
        "test",
        "failure",
        "patch",
        "test",
        "done",
    ]
    assert result.transition_counts["retry"] == 1
    assert [reward.occurrence_key for reward in result.node_rewards].count("patch#2") == 1
    assert result.score == pytest.approx(1.0)


def test_recovery_and_rollback_edges_are_exposed():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs"),
            WorkflowNode("failure", "tests failed"),
            WorkflowNode("recover", "recover workspace"),
            WorkflowNode("rollback", "rollback change"),
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

    result = WorkflowScorer(_semantic_scorer()).score(
        _trace(
            "inspect logs",
            "tests failed",
            "recover workspace",
            "rollback change",
            "finish task",
        ),
        workflow,
        coarse_to_fine=False,
    )

    assert result.score == pytest.approx(1.0)
    assert result.transition_counts["recovery"] == 1
    assert result.transition_counts["rollback"] == 1


def test_state_transition_can_blend_or_hard_gate_reward():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode(
                "patch",
                "patch code",
                preconditions={"repo": {"dirty": False}},
                postconditions={"repo": {"tests": "passed"}},
            )
        ],
        edges=[],
        start="patch",
        terminals=["patch"],
    )
    invalid_state = {
        "patch": {
            "before": {"repo": {"dirty": False, "branch": "main"}},
            "after": {"repo": {"tests": "failed"}},
        }
    }

    blended = WorkflowScorer(
        _semantic_scorer(),
        WorkflowScorerConfig(state_weight=0.25),
    ).score(
        "patch code.",
        workflow,
        state_observations=invalid_state,
        coarse_to_fine=False,
    )
    gated = WorkflowScorer(
        _semantic_scorer(),
        WorkflowScorerConfig(hard_state_gate=True),
    ).score(
        "patch code.",
        workflow,
        state_observations=invalid_state,
        coarse_to_fine=False,
    )

    assert blended.semantic_score == pytest.approx(1.0)
    assert blended.state_score == pytest.approx(0.5)
    assert blended.score == pytest.approx(0.875)
    assert blended.state_gate_passed is False
    assert blended.node_rewards[0].state_reasons == ["postcondition_mismatch"]
    assert gated.score == pytest.approx(0.0)


def test_required_state_observation_penalizes_missing_observation():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode(
                "patch",
                "patch code",
                postconditions={"repo": {"tests": "passed"}},
            )
        ],
        edges=[],
        start="patch",
        terminals=["patch"],
    )

    result = WorkflowScorer(
        _semantic_scorer(),
        WorkflowScorerConfig(require_state_observations=True, state_weight=0.4),
    ).score(
        "patch code.",
        workflow,
        coarse_to_fine=False,
    )

    assert result.state_score == pytest.approx(0.0)
    assert result.score == pytest.approx(0.6)
    assert result.node_rewards[0].state_reasons == ["missing_state_observation"]


def test_turn_level_potential_rewards_telescope_at_gamma_one():
    workflow = WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs"),
            WorkflowNode("patch", "patch code"),
            WorkflowNode("test", "run tests"),
            WorkflowNode("done", "finish task"),
        ],
        edges=[
            WorkflowEdge("inspect", "patch"),
            WorkflowEdge("patch", "test"),
            WorkflowEdge("test", "done"),
        ],
        start="inspect",
        terminals=["done"],
    )
    scorer = WorkflowPotentialScorer(
        WorkflowScorer(_semantic_scorer()),
        PotentialRewardConfig(gamma=1.0),
    )

    result = scorer.score_turns(
        ["inspect logs.", "patch code.", "run tests.", "finish task."],
        workflow,
        coarse_to_fine=False,
    )

    assert result.potentials == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert result.rewards == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert result.total_shaped_reward == pytest.approx(result.final_potential)


def _retry_workflow():
    return WorkflowGroundTruth(
        nodes=[
            WorkflowNode("inspect", "inspect logs"),
            WorkflowNode("patch", "patch code", max_visits=2),
            WorkflowNode("test", "run tests", max_visits=2),
            WorkflowNode("failure", "tests failed"),
            WorkflowNode("done", "finish task"),
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
