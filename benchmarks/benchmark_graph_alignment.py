"""Compare graph-constrained workflow alignment with path enumeration."""

from __future__ import annotations

import argparse
import time
from statistics import median

from raise_scorer.backends import load_embedding_backend
from raise_scorer.core import ScorerConfig, SemanticRewardScorer
from raise_scorer.workflows import (
    WorkflowEdge,
    WorkflowGroundTruth,
    WorkflowNode,
    WorkflowScorer,
    WorkflowScorerConfig,
)


def layered_workflow(branches: int, layers: int) -> tuple[WorkflowGroundTruth, str]:
    if branches <= 1 or layers <= 0:
        raise ValueError("branches must exceed one and layers must be positive")
    nodes = [WorkflowNode("start", "inspect the task")]
    edges = []
    previous = ["start"]
    selected_actions = ["inspect the task"]
    selected_templates = (
        "diagnose the parser failure",
        "edit the source implementation",
        "execute targeted unit tests",
        "inspect the generated code diff",
        "summarize the verified repair",
    )
    alternative_templates = (
        "browse product documentation",
        "change deployment configuration",
        "update an unrelated dependency",
        "write release notes",
        "request manual approval",
    )
    for layer in range(layers):
        current = []
        for branch in range(branches):
            node_id = f"layer_{layer}_branch_{branch}"
            if branch == 0:
                action = (
                    f"{selected_templates[layer % len(selected_templates)]} "
                    f"for phase {layer}"
                )
            else:
                action = (
                    f"{alternative_templates[(layer + branch - 1) % len(alternative_templates)]} "
                    f"for phase {layer}"
                )
            nodes.append(WorkflowNode(node_id, action))
            current.append(node_id)
            if branch == 0:
                selected_actions.append(action)
        for source in previous:
            for target in current:
                edges.append(WorkflowEdge(source, target, kind="alternative"))
        previous = current
    nodes.append(WorkflowNode("done", "verify and finish the task"))
    selected_actions.append("verify and finish the task")
    edges.extend(WorkflowEdge(source, "done") for source in previous)
    workflow = WorkflowGroundTruth(
        nodes=nodes,
        edges=edges,
        start="start",
        terminals=["done"],
    )
    response = (" neutral context " * 8).join(
        f"{action}." for action in selected_actions
    )
    return workflow, response


def _timed_score(scorer, response, workflow, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = scorer.score(response, workflow, coarse_to_fine=False)
        samples.append((time.perf_counter() - started) * 1000)
    return result, median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    backend = load_embedding_backend(
        args.model_path,
        local_files_only=args.local_files_only,
    )
    if backend is None:
        raise RuntimeError(f"could not load embedding model: {args.model_path}")
    semantic = SemanticRewardScorer(
        backend,
        ScorerConfig(threshold=args.threshold),
    )
    workflow, response = layered_workflow(args.branches, args.layers)
    expected_paths = args.branches ** args.layers
    graph_scorer = WorkflowScorer(
        semantic,
        WorkflowScorerConfig(alignment_mode="graph"),
    )
    enumerate_scorer = WorkflowScorer(
        semantic,
        WorkflowScorerConfig(
            alignment_mode="enumerate",
            max_paths=expected_paths,
        ),
    )

    # Warm model and shared reference cache before measuring both modes.
    graph_scorer.score(response, workflow, coarse_to_fine=False)
    graph_result, graph_ms = _timed_score(
        graph_scorer,
        response,
        workflow,
        args.repeats,
    )
    enum_result, enum_ms = _timed_score(
        enumerate_scorer,
        response,
        workflow,
        args.repeats,
    )

    print(f"terminal_paths={expected_paths}")
    print(
        "graph="
        f"{graph_ms:.3f}ms states={graph_result.graph_states} "
        f"transitions={graph_result.graph_transitions} score={graph_result.score:.4f}"
    )
    print(
        "enumerate="
        f"{enum_ms:.3f}ms scored_paths={enum_result.candidate_paths} "
        f"score={enum_result.score:.4f}"
    )
    print(f"speedup={enum_ms / graph_ms:.2f}x")
    print(f"path_agreement={graph_result.best_path == enum_result.best_path}")


if __name__ == "__main__":
    main()
