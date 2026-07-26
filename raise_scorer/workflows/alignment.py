"""Graph-constrained monotonic alignment for bounded workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.scorer import SemanticRewardScorer, SimilarityMatrix
from .engine import (
    _CONTRACT_FIELDS,
    WorkflowGroundTruth,
    WorkflowPath,
    _reference_for_scorer,
)


@dataclass(frozen=True)
class BoundedWorkflowState:
    """One state in the finite unrolling of a bounded workflow."""

    index: int
    node_id: str
    depth: int
    node_visits: tuple[int, ...]
    edge_visits: tuple[int, ...]
    incoming: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BoundedWorkflowGraph:
    states: tuple[BoundedWorkflowState, ...]
    start_state: int
    terminal_states: tuple[int, ...]
    truncated: bool
    expansions: int


@dataclass
class GraphAlignment:
    """Best legal workflow path and its direct graph-alignment diagnostics."""

    path: WorkflowPath
    matched_nodes: list[tuple[str, int]]
    matched_contracts: list[tuple[str, str, int]]
    estimated_coverage: float
    similarity: SimilarityMatrix
    graph_states: int
    graph_transitions: int
    terminal_states: int
    graph_truncated: bool


def build_bounded_workflow_graph(
    workflow: WorkflowGroundTruth,
    *,
    max_path_nodes: int = 64,
    max_states: int = 10000,
    max_expansions: int = 10000,
) -> BoundedWorkflowGraph:
    """Unroll a bounded cyclic workflow while sharing equivalent prefixes.

    A state contains the current node, depth, and only the counters that can
    affect future legality (nodes/edges in directed cycles). Acyclic branch
    histories can therefore reconverge instead of being duplicated.
    """
    if max_path_nodes <= 0 or max_states <= 0 or max_expansions <= 0:
        raise ValueError("graph limits must be positive")
    workflow.validate()

    node_ids = tuple(node.id for node in workflow.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    outgoing: dict[str, list[tuple[int, object]]] = {
        node_id: [] for node_id in node_ids
    }
    for edge_index, edge in enumerate(workflow.edges):
        outgoing[edge.source].append((edge_index, edge))

    adjacency = {
        node_id: [edge.target for _, edge in outgoing[node_id]]
        for node_id in node_ids
    }
    cyclic_edges = {
        edge_index
        for edge_index, edge in enumerate(workflow.edges)
        if edge.source in _reachable(edge.target, adjacency)
    }
    cyclic_nodes = {
        node_id
        for edge_index in cyclic_edges
        for node_id in (
            workflow.edges[edge_index].source,
            workflow.edges[edge_index].target,
        )
    }

    start_visits = [0] * len(node_ids)
    if workflow.start in cyclic_nodes:
        start_visits[node_index[workflow.start]] = 1
    start_key = (
        workflow.start,
        1,
        tuple(start_visits),
        tuple(0 for _ in workflow.edges),
    )
    keys = [start_key]
    key_to_index = {start_key: 0}
    incoming: list[list[tuple[int, int]]] = [[]]
    depths = [1]
    pending = [0]
    cursor = 0
    expansions = 0
    truncated = False

    while cursor < len(pending):
        state_index = pending[cursor]
        cursor += 1
        node_id, depth, node_visits, edge_visits = keys[state_index]
        if node_id in workflow.terminals:
            continue
        if depth >= max_path_nodes:
            truncated = True
            continue

        for edge_index, edge in outgoing[node_id]:
            if expansions >= max_expansions:
                truncated = True
                break
            expansions += 1
            target_index = node_index[edge.target]
            if (
                edge_index in cyclic_edges
                and edge_visits[edge_index] >= edge.max_traversals
            ):
                continue
            if (
                edge.target in cyclic_nodes
                and node_visits[target_index]
                >= workflow.nodes[target_index].max_visits
            ):
                continue

            next_node_visits = list(node_visits)
            if edge.target in cyclic_nodes:
                next_node_visits[target_index] += 1
            next_edge_visits = list(edge_visits)
            if edge_index in cyclic_edges:
                next_edge_visits[edge_index] += 1
            next_key = (
                edge.target,
                depth + 1,
                tuple(next_node_visits),
                tuple(next_edge_visits),
            )
            next_index = key_to_index.get(next_key)
            if next_index is None:
                if len(keys) >= max_states:
                    truncated = True
                    continue
                next_index = len(keys)
                key_to_index[next_key] = next_index
                keys.append(next_key)
                incoming.append([])
                depths.append(depth + 1)
                pending.append(next_index)
            incoming[next_index].append((state_index, edge_index))

    states = tuple(
        BoundedWorkflowState(
            index=index,
            node_id=key[0],
            depth=depths[index],
            node_visits=key[2],
            edge_visits=key[3],
            incoming=tuple(incoming[index]),
        )
        for index, key in enumerate(keys)
    )
    terminals = tuple(
        state.index for state in states if state.node_id in workflow.terminals
    )
    if not terminals:
        raise ValueError(
            "workflow has no terminal state within graph, traversal, and expansion constraints"
        )
    return BoundedWorkflowGraph(
        states=states,
        start_state=0,
        terminal_states=terminals,
        truncated=truncated,
        expansions=expansions,
    )


def align_workflow_response(
    response: str,
    workflow: WorkflowGroundTruth,
    semantic_scorer: SemanticRewardScorer,
    *,
    threshold: Optional[float] = None,
    window: Optional[int] = None,
    stride: Optional[int] = None,
    max_path_nodes: int = 64,
    max_states: int = 10000,
    max_expansions: int = 10000,
    passages: Optional[Sequence[str]] = None,
    action_weight: float = 0.50,
    evidence_weight: float = 0.25,
    outcome_weight: float = 0.25,
) -> GraphAlignment:
    """Find the best legal path in workflow-state × response-window space."""
    graph = build_bounded_workflow_graph(
        workflow,
        max_path_nodes=max_path_nodes,
        max_states=max_states,
        max_expansions=max_expansions,
    )
    threshold = semantic_scorer.config.threshold if threshold is None else threshold
    raw_weights = {
        "action": action_weight,
        "evidence": evidence_weight,
        "outcome": outcome_weight,
    }
    contract_specs = {}
    references = []
    for node in workflow.nodes:
        present = [
            field_name
            for field_name in _CONTRACT_FIELDS
            if getattr(node, field_name) is not None and raw_weights[field_name] > 0
        ]
        if not present:
            present = ["action"]
        total_weight = sum(raw_weights[field_name] for field_name in present)
        if total_weight <= 0:
            total_weight = 1.0
        specs = []
        for field_name in present:
            row_index = len(references)
            references.append(
                _reference_for_scorer(getattr(node, field_name))
            )
            specs.append(
                (
                    field_name,
                    row_index,
                    raw_weights[field_name] / total_weight,
                )
            )
        contract_specs[node.id] = specs
    if passages is None:
        similarity = semantic_scorer.similarity_matrix(
            response,
            references,
            window=window,
            stride=stride,
        )
    else:
        similarity = semantic_scorer.similarity_matrix_for_passages(
            passages,
            references,
        )
    values = similarity.values.detach().float().cpu().numpy()
    num_windows = len(similarity.windows)
    # Expand every workflow state into its ordered A/E/O contract phases. This
    # retains graph sharing across branches while allowing evidence and outcome
    # to occur in windows after the action.
    phases = []
    state_first_phase = {}
    state_last_phase = {}
    for state in graph.states:
        state_first_phase[state.index] = len(phases)
        specs = contract_specs[state.node_id]
        for field_name, row_index, normalized_weight in specs:
            phases.append(
                (
                    state.index,
                    state.node_id,
                    field_name,
                    row_index,
                    normalized_weight,
                )
            )
        state_last_phase[state.index] = len(phases) - 1

    num_phases = len(phases)
    dp = np.full((num_phases, num_windows + 1), -np.inf, dtype=np.float64)
    bp_kind = np.zeros((num_phases, num_windows + 1), dtype=np.int8)
    bp_parent = np.full((num_phases, num_windows + 1), -1, dtype=np.int32)
    bp_edge = np.full((num_phases, num_windows + 1), -1, dtype=np.int32)

    for phase_index, phase in enumerate(phases):
        state_index, node_id, _field_name, row_index, normalized_weight = phase
        state = graph.states[state_index]
        first_phase = state_first_phase[state_index]
        if phase_index > first_phase:
            incoming = ((phase_index - 1, -1),)
        else:
            incoming = tuple(
                (state_last_phase[parent], edge_index)
                for parent, edge_index in state.incoming
            )

        if phase_index == state_first_phase[graph.start_state]:
            base = np.zeros(num_windows + 1, dtype=np.float64)
            base_parent = np.full(num_windows + 1, -1, dtype=np.int32)
            base_edge = np.full(num_windows + 1, -1, dtype=np.int32)
        else:
            base = np.full(num_windows + 1, -np.inf, dtype=np.float64)
            base_parent = np.full(num_windows + 1, -1, dtype=np.int32)
            base_edge = np.full(num_windows + 1, -1, dtype=np.int32)
            for parent, edge_index in incoming:
                better = dp[parent] > base
                base[better] = dp[parent, better]
                base_parent[better] = parent
                base_edge[better] = edge_index

        node = workflow.node_map[node_id]
        row = values[row_index] if num_windows else np.empty(0)
        priority = node.weight * normalized_weight
        if not node.required:
            priority *= 1e-6

        dp[phase_index, 0] = base[0]
        bp_kind[phase_index, 0] = 2
        bp_parent[phase_index, 0] = base_parent[0]
        bp_edge[phase_index, 0] = base_edge[0]
        for j in range(1, num_windows + 1):
            # Skip a response window.
            best = dp[phase_index, j - 1]
            kind = 1
            parent = phase_index
            edge_index = -1

            # Skip this workflow action.
            if base[j] > best:
                best = base[j]
                kind = 2
                parent = base_parent[j]
                edge_index = base_edge[j]

            # Match the action to this response window. The tiny similarity
            # term gives deterministic preference to the cleaner match without
            # changing action-coverage priority.
            sim = float(row[j - 1]) if num_windows else 0.0
            if sim >= threshold and np.isfinite(base[j - 1]):
                gain = priority * (1.0 + sim * 1e-4)
                candidate = base[j - 1] + gain
                if candidate >= best:
                    best = candidate
                    kind = 3
                    parent = base_parent[j - 1]
                    edge_index = base_edge[j - 1]

            dp[phase_index, j] = best
            bp_kind[phase_index, j] = kind
            bp_parent[phase_index, j] = parent
            bp_edge[phase_index, j] = edge_index

    def backtrack(terminal_state_index: int):
        phase_path = []
        edge_path = []
        matched_phases = []
        phase_index = state_last_phase[terminal_state_index]
        j = num_windows
        while phase_index >= 0:
            kind = int(bp_kind[phase_index, j])
            if kind == 1:
                j -= 1
                continue
            phase_path.append(phase_index)
            if kind == 3:
                matched_phases.append((phase_index, j - 1))
                next_j = j - 1
            else:
                next_j = j
            edge_index = int(bp_edge[phase_index, j])
            parent = int(bp_parent[phase_index, j])
            if edge_index >= 0:
                edge_path.append(edge_index)
            phase_index = parent
            j = next_j
        phase_path.reverse()
        edge_path.reverse()
        matched_phases.reverse()

        workflow_state_path = []
        for index in phase_path:
            workflow_state_index = phases[index][0]
            if (
                not workflow_state_path
                or workflow_state_path[-1] != workflow_state_index
            ):
                workflow_state_path.append(workflow_state_index)
        node_path = tuple(
            graph.states[index].node_id for index in workflow_state_path
        )
        matched_contracts = [
            (phases[index][1], phases[index][2], window_index)
            for index, window_index in matched_phases
        ]
        matched_nodes = [
            (node_id, window_index)
            for node_id, field_name, window_index in matched_contracts
            if field_name == "action"
        ]
        matched_weight = sum(
            workflow.node_map[phases[index][1]].weight * phases[index][4]
            for index, _ in matched_phases
            if workflow.node_map[phases[index][1]].required
        )
        total_weight = sum(
            workflow.node_map[node_id].weight
            for node_id in node_path
            if workflow.node_map[node_id].required
        )
        coverage = matched_weight / total_weight if total_weight else 0.0
        return (
            node_path,
            tuple(edge_path),
            matched_nodes,
            matched_contracts,
            coverage,
        )

    candidates = [
        (terminal_index, *backtrack(terminal_index))
        for terminal_index in graph.terminal_states
    ]
    (
        _terminal,
        node_path,
        edge_path,
        matched_nodes,
        matched_contracts,
        estimated_coverage,
    ) = max(
        candidates,
        key=lambda item: (
            item[5],
            dp[state_last_phase[item[0]], num_windows],
            -graph.states[item[0]].depth,
        ),
    )
    return GraphAlignment(
        path=WorkflowPath(node_ids=node_path, edge_indexes=edge_path),
        matched_nodes=matched_nodes,
        matched_contracts=matched_contracts,
        estimated_coverage=estimated_coverage,
        similarity=similarity,
        graph_states=len(graph.states),
        graph_transitions=sum(len(state.incoming) for state in graph.states),
        terminal_states=len(graph.terminal_states),
        graph_truncated=graph.truncated,
    )


def _reachable(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    visited = set()
    pending = [start]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(adjacency.get(node_id, ()))
    return visited


__all__ = [
    "BoundedWorkflowGraph",
    "BoundedWorkflowState",
    "GraphAlignment",
    "align_workflow_response",
    "build_bounded_workflow_graph",
]
