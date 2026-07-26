"""Bounded workflow/state-machine GroundTruth scoring.

This module keeps the existing linear ``reference_steps`` API intact and adds
an explicit graph layer for long-horizon agent tasks. A workflow is expanded
into bounded legal paths; all candidate paths are then scored in one
``score_batch`` call so response-window embeddings remain shared.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from ..core.scorer import RewardScore, SemanticRewardScorer


StepReference = Union[str, tuple[str, ...]]
StateValidator = Callable[["WorkflowNode", "StateObservation"], Union[bool, float]]

_EDGE_KINDS = {"forward", "alternative", "retry", "recovery", "rollback"}
_CONTRACT_FIELDS = ("action", "evidence", "outcome")


def _normalize_reference(value: Any, *, field_name: str, required: bool) -> Optional[StepReference]:
    if value is None:
        if required:
            raise ValueError(f"{field_name} must not be empty")
        return None
    if isinstance(value, (list, tuple)):
        candidates = tuple(str(candidate).strip() for candidate in value if str(candidate).strip())
        if not candidates:
            if required:
                raise ValueError(f"{field_name} must not be empty")
            return None
        return candidates
    normalized = str(value).strip()
    if not normalized:
        if required:
            raise ValueError(f"{field_name} must not be empty")
        return None
    return normalized


def _reference_for_scorer(reference: StepReference):
    return list(reference) if isinstance(reference, tuple) else reference


@dataclass
class WorkflowNode:
    """One state-machine node with an Action/Evidence/Outcome contract."""

    id: str
    action: StepReference
    stage: str = "default"
    required: bool = True
    evidence: Optional[StepReference] = None
    outcome: Optional[StepReference] = None
    weight: float = 1.0
    max_visits: int = 1
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    postconditions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.stage = str(self.stage).strip() or "default"
        self.action = _normalize_reference(
            self.action,
            field_name=f"node {self.id or '<unknown>'} action",
            required=True,
        )
        self.evidence = _normalize_reference(
            self.evidence,
            field_name=f"node {self.id or '<unknown>'} evidence",
            required=False,
        )
        self.outcome = _normalize_reference(
            self.outcome,
            field_name=f"node {self.id or '<unknown>'} outcome",
            required=False,
        )
        self.preconditions = dict(self.preconditions or {})
        self.postconditions = dict(self.postconditions or {})
        if not self.id:
            raise ValueError("workflow node id must not be empty")
        if self.weight <= 0:
            raise ValueError(f"node {self.id} weight must be positive")
        if self.max_visits <= 0:
            raise ValueError(f"node {self.id} max_visits must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowNode":
        if not isinstance(data, Mapping):
            raise TypeError("workflow node must be a mapping")
        return cls(
            id=data.get("id", ""),
            action=data.get("action"),
            stage=data.get("stage", "default"),
            required=bool(data.get("required", True)),
            evidence=data.get("evidence"),
            outcome=data.get("outcome"),
            weight=float(data.get("weight", 1.0)),
            max_visits=int(data.get("max_visits", 1)),
            preconditions=data.get("preconditions", {}),
            postconditions=data.get("postconditions", {}),
        )


@dataclass
class WorkflowEdge:
    """A legal transition. Cyclic edge kinds must still carry finite bounds."""

    source: str
    target: str
    kind: str = "forward"
    max_traversals: int = 1

    def __post_init__(self) -> None:
        self.source = str(self.source).strip()
        self.target = str(self.target).strip()
        self.kind = str(self.kind).strip().lower()
        if not self.source or not self.target:
            raise ValueError("workflow edge source and target must not be empty")
        if self.kind not in _EDGE_KINDS:
            supported = ", ".join(sorted(_EDGE_KINDS))
            raise ValueError(f"unsupported edge kind {self.kind!r}; expected one of: {supported}")
        if self.max_traversals <= 0:
            raise ValueError("edge max_traversals must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowEdge":
        if not isinstance(data, Mapping):
            raise TypeError("workflow edge must be a mapping")
        return cls(
            source=data.get("source", data.get("from", "")),
            target=data.get("target", data.get("to", "")),
            kind=data.get("kind", "forward"),
            max_traversals=int(data.get("max_traversals", 1)),
        )


@dataclass
class WorkflowGroundTruth:
    """Validated DAG or bounded cyclic state-machine specification."""

    nodes: Sequence[WorkflowNode]
    edges: Sequence[WorkflowEdge]
    start: str
    terminals: Sequence[str]
    stage_weights: Mapping[str, float] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        self.nodes = tuple(
            node if isinstance(node, WorkflowNode) else WorkflowNode.from_dict(node)
            for node in self.nodes
        )
        self.edges = tuple(
            edge if isinstance(edge, WorkflowEdge) else WorkflowEdge.from_dict(edge)
            for edge in self.edges
        )
        self.start = str(self.start).strip()
        self.terminals = tuple(str(node_id).strip() for node_id in self.terminals)
        self.stage_weights = {
            str(stage).strip(): float(weight)
            for stage, weight in dict(self.stage_weights or {}).items()
        }
        self.name = str(self.name).strip()
        self.validate()

    @property
    def node_map(self) -> dict[str, WorkflowNode]:
        return {node.id: node for node in self.nodes}

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("workflow must contain at least one node")
        node_ids = [node.id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("workflow node ids must be unique")
        node_set = set(node_ids)
        if self.start not in node_set:
            raise ValueError(f"workflow start node {self.start!r} does not exist")
        if not self.terminals:
            raise ValueError("workflow must contain at least one terminal")
        if any(not terminal for terminal in self.terminals):
            raise ValueError("workflow terminal ids must not be empty")
        unknown_terminals = set(self.terminals) - node_set
        if unknown_terminals:
            raise ValueError(f"unknown workflow terminals: {sorted(unknown_terminals)}")
        if not any(node.required for node in self.nodes):
            raise ValueError("workflow must contain at least one required node")
        for stage, weight in self.stage_weights.items():
            if not stage or weight <= 0:
                raise ValueError("workflow stage weights must use non-empty stages and positive values")

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        reverse: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError(
                    f"workflow edge {edge.source!r}->{edge.target!r} references an unknown node"
                )
            adjacency[edge.source].append(edge.target)
            reverse[edge.target].append(edge.source)

        reachable = _reachable_from(self.start, adjacency)
        unreachable = node_set - reachable
        if unreachable:
            raise ValueError(f"workflow contains unreachable nodes: {sorted(unreachable)}")

        can_finish = set()
        for terminal in self.terminals:
            can_finish.update(_reachable_from(terminal, reverse))
        dead_end_nodes = node_set - can_finish
        if dead_end_nodes:
            raise ValueError(
                "workflow nodes cannot reach a terminal: "
                f"{sorted(dead_end_nodes)}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowGroundTruth":
        if not isinstance(data, Mapping):
            raise TypeError("workflow ground truth must be a mapping")
        return cls(
            nodes=data.get("nodes", ()),
            edges=data.get("edges", ()),
            start=data.get("start", ""),
            terminals=data.get("terminals", ()),
            stage_weights=data.get("stage_weights", {}),
            name=data.get("name", ""),
        )


def _reachable_from(start: str, adjacency: Mapping[str, Sequence[str]]) -> set[str]:
    visited = set()
    pending = [start]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(adjacency.get(node_id, ()))
    return visited


@dataclass(frozen=True)
class WorkflowPath:
    node_ids: tuple[str, ...]
    edge_indexes: tuple[int, ...]


@dataclass(frozen=True)
class PathEnumeration:
    paths: tuple[WorkflowPath, ...]
    truncated: bool


def enumerate_workflow_paths(
    workflow: WorkflowGroundTruth,
    *,
    max_paths: int = 128,
    max_path_nodes: int = 64,
    max_expansions: int = 10000,
) -> PathEnumeration:
    """Enumerate legal terminal paths while enforcing edge/node loop bounds."""
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")
    if max_path_nodes <= 0 or max_expansions <= 0:
        raise ValueError("max_path_nodes and max_expansions must be positive")

    workflow.validate()
    node_map = workflow.node_map
    outgoing: dict[str, list[tuple[int, WorkflowEdge]]] = {
        node_id: [] for node_id in node_map
    }
    for edge_index, edge in enumerate(workflow.edges):
        outgoing[edge.source].append((edge_index, edge))

    collection_limit = max_paths + 1
    paths: list[WorkflowPath] = []
    expansions = 0
    expansion_limit_reached = False

    def visit(
        node_id: str,
        node_path: list[str],
        edge_path: list[int],
        node_visits: dict[str, int],
        edge_visits: dict[int, int],
    ) -> None:
        nonlocal expansions, expansion_limit_reached
        if len(paths) >= collection_limit:
            return
        if expansions >= max_expansions:
            expansion_limit_reached = True
            return
        expansions += 1
        if node_id in workflow.terminals:
            paths.append(WorkflowPath(tuple(node_path), tuple(edge_path)))
            return
        if len(node_path) >= max_path_nodes:
            return

        for edge_index, edge in outgoing[node_id]:
            if edge_visits.get(edge_index, 0) >= edge.max_traversals:
                continue
            target_visits = node_visits.get(edge.target, 0)
            if target_visits >= node_map[edge.target].max_visits:
                continue

            next_node_visits = dict(node_visits)
            next_node_visits[edge.target] = target_visits + 1
            next_edge_visits = dict(edge_visits)
            next_edge_visits[edge_index] = next_edge_visits.get(edge_index, 0) + 1
            visit(
                edge.target,
                [*node_path, edge.target],
                [*edge_path, edge_index],
                next_node_visits,
                next_edge_visits,
            )

    visit(
        workflow.start,
        [workflow.start],
        [],
        {workflow.start: 1},
        {},
    )
    if not paths:
        raise ValueError(
            "workflow has no terminal path within path, traversal, and expansion constraints"
        )
    return PathEnumeration(
        paths=tuple(paths[:max_paths]),
        truncated=len(paths) > max_paths or expansion_limit_reached,
    )


@dataclass
class StateObservation:
    before: Mapping[str, Any] = field(default_factory=dict)
    after: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.before = dict(self.before or {})
        self.after = dict(self.after or {})

    @classmethod
    def from_value(cls, value: Any) -> "StateObservation":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("state observation must be a mapping or StateObservation")
        return cls(before=value.get("before", {}), after=value.get("after", {}))


@dataclass(frozen=True)
class WorkflowScorerConfig:
    action_weight: float = 0.50
    evidence_weight: float = 0.25
    outcome_weight: float = 0.25
    state_weight: float = 0.25
    require_state_observations: bool = False
    hard_state_gate: bool = False
    trace_weight: float = 0.25
    hard_trace_gate: bool = True
    alignment_mode: str = "graph"
    max_paths: int = 128
    max_path_nodes: int = 64
    max_expansions: int = 10000
    max_graph_states: int = 10000

    def __post_init__(self) -> None:
        field_weights = (self.action_weight, self.evidence_weight, self.outcome_weight)
        if any(weight < 0 for weight in field_weights) or sum(field_weights) <= 0:
            raise ValueError("A/E/O weights must be non-negative and contain a positive weight")
        if not 0.0 <= self.state_weight <= 1.0:
            raise ValueError("state_weight must be between 0 and 1")
        if not 0.0 <= self.trace_weight <= 1.0:
            raise ValueError("trace_weight must be between 0 and 1")
        if self.alignment_mode not in {"graph", "enumerate"}:
            raise ValueError("alignment_mode must be 'graph' or 'enumerate'")
        if (
            self.max_paths <= 0
            or self.max_path_nodes <= 0
            or self.max_expansions <= 0
            or self.max_graph_states <= 0
        ):
            raise ValueError(
                "workflow search limits must be positive"
            )


@dataclass
class FieldReward:
    reference: StepReference
    matched: bool
    weight: float


@dataclass
class NodeReward:
    node_id: str
    occurrence: int
    occurrence_key: str
    stage: str
    required: bool
    score: float
    fields: dict[str, FieldReward]
    state_score: Optional[float]
    state_reasons: list[str]


@dataclass
class StageReward:
    score: float
    required_coverage: Optional[float]
    optional_coverage: Optional[float]
    required_nodes: int
    optional_nodes: int


@dataclass
class WorkflowReward:
    score: float
    semantic_score: float
    required_coverage: float
    optional_coverage: Optional[float]
    order_rate: float
    state_score: Optional[float]
    state_gate_passed: Optional[bool]
    best_path: list[str]
    edge_kinds: list[str]
    transition_counts: dict[str, int]
    node_rewards: list[NodeReward]
    stage_rewards: dict[str, StageReward]
    semantic_result: RewardScore
    optional_result: Optional[RewardScore]
    candidate_paths: int
    paths_truncated: bool
    all_path_scores: list[float]
    trace_validation: Optional[Any] = None
    alignment_mode: str = "enumerate"
    graph_states: Optional[int] = None
    graph_transitions: Optional[int] = None
    graph_terminal_states: Optional[int] = None
    graph_estimated_coverage: Optional[float] = None

    def to_dict(self) -> dict:
        output = asdict(self)
        if self.trace_validation is not None:
            output["trace_validation"] = self.trace_validation.to_dict()
        return output


@dataclass
class _ContractItem:
    node: WorkflowNode
    occurrence: int
    occurrence_key: str
    field_name: str
    reference: StepReference
    normalized_weight: float


@dataclass
class _PreparedPath:
    path: WorkflowPath
    required_items: list[_ContractItem]
    optional_items: list[_ContractItem]
    occurrence_nodes: list[tuple[WorkflowNode, int, str]]


@dataclass
class _PathScore:
    reward: WorkflowReward
    selection_key: tuple


class WorkflowScorer:
    """Score a response with graph alignment or the enumeration baseline."""

    def __init__(
        self,
        semantic_scorer: SemanticRewardScorer,
        config: Optional[WorkflowScorerConfig] = None,
    ):
        self.semantic_scorer = semantic_scorer
        self.config = config or WorkflowScorerConfig()

    def score(
        self,
        response: str,
        workflow: WorkflowGroundTruth,
        *,
        state_observations: Optional[Mapping[str, Any]] = None,
        state_validator: Optional[StateValidator] = None,
        coarse_to_fine: bool = True,
    ) -> WorkflowReward:
        graph_alignment = None
        if self.config.alignment_mode == "graph":
            from .alignment import align_workflow_response

            graph_alignment = align_workflow_response(
                response,
                workflow,
                self.semantic_scorer,
                max_path_nodes=self.config.max_path_nodes,
                max_states=self.config.max_graph_states,
                max_expansions=self.config.max_expansions,
                action_weight=self.config.action_weight,
                evidence_weight=self.config.evidence_weight,
                outcome_weight=self.config.outcome_weight,
            )
            paths = (graph_alignment.path,)
            candidate_paths = graph_alignment.terminal_states
            paths_truncated = graph_alignment.graph_truncated
        else:
            enumeration = enumerate_workflow_paths(
                workflow,
                max_paths=self.config.max_paths,
                max_path_nodes=self.config.max_path_nodes,
                max_expansions=self.config.max_expansions,
            )
            paths = enumeration.paths
            candidate_paths = len(paths)
            paths_truncated = enumeration.truncated

        prepared = [self._prepare_path(workflow, path) for path in paths]
        for item in prepared:
            if not item.required_items:
                raise ValueError(
                    f"legal path {item.path.node_ids!r} contains no required contract items"
                )

        jobs: list[tuple[int, str, list[_ContractItem]]] = []
        for path_index, item in enumerate(prepared):
            jobs.append((path_index, "required", item.required_items))
            if item.optional_items:
                jobs.append((path_index, "optional", item.optional_items))

        semantic_results = self.semantic_scorer.score_batch(
            responses=[response] * len(jobs),
            reference_steps_batch=[
                [_reference_for_scorer(contract.reference) for contract in contracts]
                for _, _, contracts in jobs
            ],
            coarse_to_fine=coarse_to_fine,
        )
        result_map: dict[tuple[int, str], RewardScore] = {
            (path_index, role): result
            for (path_index, role, _), result in zip(jobs, semantic_results)
        }

        observations = {
            str(key): StateObservation.from_value(value)
            for key, value in dict(state_observations or {}).items()
        }
        path_scores = [
            self._build_path_reward(
                workflow=workflow,
                prepared=item,
                required_result=result_map[(path_index, "required")],
                optional_result=result_map.get((path_index, "optional")),
                observations=observations,
                state_validator=state_validator,
                candidate_paths=candidate_paths,
                paths_truncated=paths_truncated,
            )
            for path_index, item in enumerate(prepared)
        ]
        best = max(path_scores, key=lambda item: item.selection_key)
        best.reward.all_path_scores = [item.reward.score for item in path_scores]
        best.reward.alignment_mode = self.config.alignment_mode
        if graph_alignment is not None:
            best.reward.graph_states = graph_alignment.graph_states
            best.reward.graph_transitions = graph_alignment.graph_transitions
            best.reward.graph_terminal_states = graph_alignment.terminal_states
            best.reward.graph_estimated_coverage = graph_alignment.estimated_coverage
        return best.reward

    def score_events(
        self,
        events: Sequence[Any],
        workflow: WorkflowGroundTruth,
        *,
        validation_config=None,
        state_observations: Optional[Mapping[str, Any]] = None,
        state_validator: Optional[StateValidator] = None,
        coarse_to_fine: bool = True,
    ) -> WorkflowReward:
        """Score trusted runtime events and validate their state-machine path."""
        # Lazy import avoids a module cycle: trace validation uses the workflow
        # schema, while this convenience method consumes trace validation.
        from ..runtime.events import (
            TraceValidationConfig,
            render_trace_events,
            validate_trace_events,
        )

        if validation_config is None:
            validation_config = TraceValidationConfig()
        elif isinstance(validation_config, Mapping):
            validation_config = TraceValidationConfig(**validation_config)

        validation = validate_trace_events(events, workflow, validation_config)
        merged_observations = dict(validation.state_observations)
        merged_observations.update(dict(state_observations or {}))
        trace_text = render_trace_events(
            events,
            max_field_chars=validation_config.max_field_chars,
        )
        reward = self.score(
            trace_text,
            workflow,
            state_observations=merged_observations,
            state_validator=state_validator,
            coarse_to_fine=coarse_to_fine,
        )

        expected_path = validation.node_path
        path_matches = (
            reward.best_path == expected_path
            if validation.terminal_reached
            else reward.best_path[:len(expected_path)] == expected_path
        )
        if expected_path and not path_matches:
            validation.add_violation(
                "semantic_path_mismatch:"
                f"{'->'.join(expected_path)}!={'->'.join(reward.best_path)}"
            )

        reward.trace_validation = validation
        if not validation.valid:
            if self.config.hard_trace_gate:
                reward.score = 0.0
            else:
                reward.score *= (
                    1.0
                    - self.config.trace_weight
                    + self.config.trace_weight * validation.validity_score
                )
        return reward

    def _prepare_path(
        self,
        workflow: WorkflowGroundTruth,
        path: WorkflowPath,
    ) -> _PreparedPath:
        node_map = workflow.node_map
        visit_counts: dict[str, int] = {}
        required_items: list[_ContractItem] = []
        optional_items: list[_ContractItem] = []
        occurrence_nodes = []
        raw_field_weights = {
            "action": self.config.action_weight,
            "evidence": self.config.evidence_weight,
            "outcome": self.config.outcome_weight,
        }

        for node_id in path.node_ids:
            node = node_map[node_id]
            occurrence = visit_counts.get(node_id, 0) + 1
            visit_counts[node_id] = occurrence
            occurrence_key = f"{node_id}#{occurrence}"
            occurrence_nodes.append((node, occurrence, occurrence_key))

            present_fields = [
                field_name
                for field_name in _CONTRACT_FIELDS
                if getattr(node, field_name) is not None and raw_field_weights[field_name] > 0
            ]
            total_field_weight = sum(raw_field_weights[field_name] for field_name in present_fields)
            contracts = required_items if node.required else optional_items
            for field_name in present_fields:
                contracts.append(
                    _ContractItem(
                        node=node,
                        occurrence=occurrence,
                        occurrence_key=occurrence_key,
                        field_name=field_name,
                        reference=getattr(node, field_name),
                        normalized_weight=raw_field_weights[field_name] / total_field_weight,
                    )
                )
        return _PreparedPath(
            path=path,
            required_items=required_items,
            optional_items=optional_items,
            occurrence_nodes=occurrence_nodes,
        )

    def _build_path_reward(
        self,
        *,
        workflow: WorkflowGroundTruth,
        prepared: _PreparedPath,
        required_result: RewardScore,
        optional_result: Optional[RewardScore],
        observations: Mapping[str, StateObservation],
        state_validator: Optional[StateValidator],
        candidate_paths: int,
        paths_truncated: bool,
    ) -> _PathScore:
        required_matches = {step_index for step_index, _ in required_result.alignment_path}
        optional_matches = (
            {step_index for step_index, _ in optional_result.alignment_path}
            if optional_result is not None
            else set()
        )
        item_match: dict[tuple[str, str], bool] = {}
        for index, item in enumerate(prepared.required_items):
            item_match[(item.occurrence_key, item.field_name)] = index in required_matches
        for index, item in enumerate(prepared.optional_items):
            item_match[(item.occurrence_key, item.field_name)] = index in optional_matches

        node_rewards = []
        for node, occurrence, occurrence_key in prepared.occurrence_nodes:
            fields: dict[str, FieldReward] = {}
            raw_weights = {
                "action": self.config.action_weight,
                "evidence": self.config.evidence_weight,
                "outcome": self.config.outcome_weight,
            }
            present = [
                field_name
                for field_name in _CONTRACT_FIELDS
                if getattr(node, field_name) is not None and raw_weights[field_name] > 0
            ]
            denominator = sum(raw_weights[field_name] for field_name in present)
            score = 0.0
            for field_name in present:
                normalized_weight = raw_weights[field_name] / denominator
                matched = item_match.get((occurrence_key, field_name), False)
                fields[field_name] = FieldReward(
                    reference=getattr(node, field_name),
                    matched=matched,
                    weight=normalized_weight,
                )
                if matched:
                    score += normalized_weight

            state_score, state_reasons = self._score_node_state(
                node,
                occurrence,
                occurrence_key,
                observations,
                state_validator,
            )
            node_rewards.append(
                NodeReward(
                    node_id=node.id,
                    occurrence=occurrence,
                    occurrence_key=occurrence_key,
                    stage=node.stage,
                    required=node.required,
                    score=score,
                    fields=fields,
                    state_score=state_score,
                    state_reasons=state_reasons,
                )
            )

        stage_rewards = self._stage_rewards(node_rewards, prepared.occurrence_nodes)
        required_coverage = self._aggregate_required_stages(
            workflow,
            stage_rewards,
            prepared.occurrence_nodes,
        )
        optional_nodes = [reward for reward in node_rewards if not reward.required]
        optional_coverage = (
            sum(
                reward.score
                * next(
                    node.weight
                    for node, occurrence, _ in prepared.occurrence_nodes
                    if node.id == reward.node_id and occurrence == reward.occurrence
                )
                for reward in optional_nodes
            )
            / sum(
                next(
                    node.weight
                    for node, occurrence, _ in prepared.occurrence_nodes
                    if node.id == reward.node_id and occurrence == reward.occurrence
                )
                for reward in optional_nodes
            )
            if optional_nodes
            else None
        )

        order_rate = _workflow_order_rate(required_result, prepared.required_items)
        semantic_score = required_coverage * order_rate
        state_nodes = [
            reward
            for reward in node_rewards
            if reward.required
            and (
                workflow.node_map[reward.node_id].preconditions
                or workflow.node_map[reward.node_id].postconditions
            )
        ]
        state_active = bool(observations) or self.config.require_state_observations
        state_score = None
        state_gate_passed = None
        final_score = semantic_score
        if state_nodes and state_active:
            state_gate_passed = all(reward.state_score == 1.0 for reward in state_nodes)
            state_score = sum(reward.state_score or 0.0 for reward in state_nodes) / len(
                state_nodes
            )
            if self.config.hard_state_gate and not state_gate_passed:
                final_score = 0.0
            else:
                final_score *= (
                    1.0 - self.config.state_weight
                    + self.config.state_weight * state_score
                )

        edge_kinds = [workflow.edges[index].kind for index in prepared.path.edge_indexes]
        transition_counts = {
            kind: edge_kinds.count(kind)
            for kind in sorted(_EDGE_KINDS)
            if edge_kinds.count(kind)
        }
        reward = WorkflowReward(
            score=final_score,
            semantic_score=semantic_score,
            required_coverage=required_coverage,
            optional_coverage=optional_coverage,
            order_rate=order_rate,
            state_score=state_score,
            state_gate_passed=state_gate_passed,
            best_path=list(prepared.path.node_ids),
            edge_kinds=edge_kinds,
            transition_counts=transition_counts,
            node_rewards=node_rewards,
            stage_rewards=stage_rewards,
            semantic_result=required_result,
            optional_result=optional_result,
            candidate_paths=candidate_paths,
            paths_truncated=paths_truncated,
            all_path_scores=[],
        )
        return _PathScore(
            reward=reward,
            selection_key=(
                reward.score,
                reward.required_coverage,
                reward.order_rate,
                len(required_matches),
                reward.optional_coverage or 0.0,
                -len(reward.best_path),
            ),
        )

    def _stage_rewards(
        self,
        node_rewards: Sequence[NodeReward],
        occurrence_nodes: Sequence[tuple[WorkflowNode, int, str]],
    ) -> dict[str, StageReward]:
        node_lookup = {
            (node.id, occurrence): node
            for node, occurrence, _ in occurrence_nodes
        }
        stages = sorted({reward.stage for reward in node_rewards})
        output = {}
        for stage in stages:
            required = [
                reward for reward in node_rewards if reward.stage == stage and reward.required
            ]
            optional = [
                reward for reward in node_rewards if reward.stage == stage and not reward.required
            ]

            def weighted_coverage(rewards: Sequence[NodeReward]) -> Optional[float]:
                if not rewards:
                    return None
                total = sum(
                    node_lookup[(reward.node_id, reward.occurrence)].weight
                    for reward in rewards
                )
                return sum(
                    reward.score * node_lookup[(reward.node_id, reward.occurrence)].weight
                    for reward in rewards
                ) / total

            required_coverage = weighted_coverage(required)
            optional_coverage = weighted_coverage(optional)
            output[stage] = StageReward(
                score=(
                    required_coverage
                    if required_coverage is not None
                    else optional_coverage or 0.0
                ),
                required_coverage=required_coverage,
                optional_coverage=optional_coverage,
                required_nodes=len(required),
                optional_nodes=len(optional),
            )
        return output

    def _aggregate_required_stages(
        self,
        workflow: WorkflowGroundTruth,
        stage_rewards: Mapping[str, StageReward],
        occurrence_nodes: Sequence[tuple[WorkflowNode, int, str]],
    ) -> float:
        required_stages = {
            stage: reward
            for stage, reward in stage_rewards.items()
            if reward.required_coverage is not None
        }
        weights = {}
        for stage in required_stages:
            if stage in workflow.stage_weights:
                weights[stage] = workflow.stage_weights[stage]
            else:
                weights[stage] = sum(
                    node.weight
                    for node, _, _ in occurrence_nodes
                    if node.required and node.stage == stage
                )
        total_weight = sum(weights.values())
        return sum(
            required_stages[stage].required_coverage * weight
            for stage, weight in weights.items()
        ) / total_weight

    def _score_node_state(
        self,
        node: WorkflowNode,
        occurrence: int,
        occurrence_key: str,
        observations: Mapping[str, StateObservation],
        state_validator: Optional[StateValidator],
    ) -> tuple[Optional[float], list[str]]:
        if not node.preconditions and not node.postconditions:
            return None, []

        observation = observations.get(occurrence_key)
        if observation is None and occurrence == 1:
            observation = observations.get(node.id)
        if observation is None:
            return None, ["missing_state_observation"]

        checks = []
        reasons = []
        if node.preconditions:
            passed = _mapping_contains(observation.before, node.preconditions)
            checks.append(float(passed))
            if not passed:
                reasons.append("precondition_mismatch")
        if node.postconditions:
            passed = _mapping_contains(observation.after, node.postconditions)
            checks.append(float(passed))
            if not passed:
                reasons.append("postcondition_mismatch")
        if state_validator is not None:
            custom = state_validator(node, observation)
            custom_score = float(custom)
            if not 0.0 <= custom_score <= 1.0:
                raise ValueError("state_validator must return bool or a float between 0 and 1")
            checks.append(custom_score)
            if custom_score < 1.0:
                reasons.append("custom_state_validator_failed")
        return sum(checks) / len(checks), reasons


def _mapping_contains(actual: Any, expected: Any) -> bool:
    """Return whether ``actual`` recursively contains every expected value."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _mapping_contains(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def _workflow_order_rate(
    result: RewardScore,
    items: Sequence[_ContractItem],
) -> float:
    """Correct repeated state-machine visits without hiding distinct-step disorder.

    The base scorer uses each step's strongest window for its all-pairs order
    metric. Repeated retry actions naturally share the same strongest window,
    which would penalize a perfectly ordered second visit. For repeated
    contracts only, use the monotonic path's occurrence-specific window;
    unique contracts retain the base strongest-window behavior.
    """
    best_sims = list(result.stats.get("best_step_sims", ()))
    best_windows = list(result.stats.get("best_window_indices", ()))
    if len(best_sims) != len(items) or len(best_windows) != len(items):
        return result.order_rate

    def reference_key(reference: StepReference) -> tuple[str, ...]:
        if isinstance(reference, tuple):
            return tuple(candidate.casefold() for candidate in reference)
        return (reference.casefold(),)

    keys = [reference_key(item.reference) for item in items]
    counts = Counter(keys)
    path_windows = {step_index: window_index for step_index, window_index in result.alignment_path}
    threshold = float(result.stats.get("threshold", 0.0))
    positions = []
    for index, (key, similarity, best_window) in enumerate(
        zip(keys, best_sims, best_windows)
    ):
        if float(similarity) < threshold:
            continue
        if counts[key] > 1 and index in path_windows:
            positions.append(path_windows[index])
        else:
            positions.append(int(best_window))

    if not positions:
        return 0.0
    if len(positions) == 1:
        return 1.0
    pair_credit = 0.0
    pair_count = 0
    for left_index, left_window in enumerate(positions[:-1]):
        for right_window in positions[left_index + 1:]:
            pair_count += 1
            if left_window < right_window:
                pair_credit += 1.0
            elif left_window == right_window:
                pair_credit += 0.5
    return pair_credit / pair_count


def score_workflow(
    semantic_scorer: SemanticRewardScorer,
    response: str,
    workflow: WorkflowGroundTruth,
    *,
    config: Optional[WorkflowScorerConfig] = None,
    state_observations: Optional[Mapping[str, Any]] = None,
    state_validator: Optional[StateValidator] = None,
    coarse_to_fine: bool = True,
) -> WorkflowReward:
    """Convenience wrapper for one workflow score."""
    return WorkflowScorer(semantic_scorer, config).score(
        response,
        workflow,
        state_observations=state_observations,
        state_validator=state_validator,
        coarse_to_fine=coarse_to_fine,
    )
