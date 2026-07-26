"""Trusted runtime event protocol for workflow reward scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..workflows.engine import WorkflowGroundTruth


_EVENT_STATUSES = {
    "pending",
    "running",
    "success",
    "failure",
    "retrying",
    "recovered",
    "rolled_back",
    "skipped",
}
_EVENT_SOURCES = {"runtime", "external", "model"}


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


@dataclass
class TraceEvent:
    """One ordered event emitted by a tool/runtime adapter.

    ``trusted`` must only be set by infrastructure outside the model output.
    ``node_id`` maps the event to a Workflow GroundTruth node. Consecutive
    events with the same node and occurrence belong to one node visit.
    """

    event_id: str
    action: str
    node_id: Optional[str] = None
    occurrence: Optional[int] = None
    evidence: Optional[str] = None
    outcome: Optional[str] = None
    status: str = "success"
    source: str = "model"
    trusted: bool = False
    transition_kind: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Any = None
    tool_output: Any = None
    state_before: Mapping[str, Any] = field(default_factory=dict)
    state_after: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id).strip()
        self.action = str(self.action).strip()
        self.node_id = _optional_text(self.node_id)
        self.evidence = _optional_text(self.evidence)
        self.outcome = _optional_text(self.outcome)
        self.status = str(self.status).strip().lower()
        self.source = str(self.source).strip().lower()
        self.transition_kind = _optional_text(self.transition_kind)
        if self.transition_kind is not None:
            self.transition_kind = self.transition_kind.lower()
        self.tool_name = _optional_text(self.tool_name)
        self.state_before = dict(self.state_before or {})
        self.state_after = dict(self.state_after or {})
        self.metadata = dict(self.metadata or {})
        if not self.event_id:
            raise ValueError("trace event_id must not be empty")
        if not self.action:
            raise ValueError(f"trace event {self.event_id!r} action must not be empty")
        if self.occurrence is not None and self.occurrence <= 0:
            raise ValueError("trace event occurrence must be positive")
        if self.status not in _EVENT_STATUSES:
            supported = ", ".join(sorted(_EVENT_STATUSES))
            raise ValueError(
                f"unsupported trace event status {self.status!r}; expected one of: {supported}"
            )
        if self.source not in _EVENT_SOURCES:
            supported = ", ".join(sorted(_EVENT_SOURCES))
            raise ValueError(
                f"unsupported trace event source {self.source!r}; expected one of: {supported}"
            )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        index: int = 0,
        *,
        allow_trusted: bool = False,
    ) -> "TraceEvent":
        if not isinstance(data, Mapping):
            raise TypeError("trace event must be a mapping")
        return cls(
            event_id=data.get("event_id", data.get("id", f"event-{index}")),
            action=data.get("action", ""),
            node_id=data.get("node_id"),
            occurrence=(
                int(data["occurrence"])
                if data.get("occurrence") is not None
                else None
            ),
            evidence=data.get("evidence"),
            outcome=data.get("outcome"),
            status=data.get("status", "success"),
            source=data.get("source", "model"),
            trusted=bool(data.get("trusted", False)) if allow_trusted else False,
            transition_kind=data.get("transition_kind"),
            tool_name=data.get("tool_name", data.get("tool")),
            tool_input=data.get("tool_input"),
            tool_output=data.get("tool_output"),
            state_before=data.get("state_before", {}),
            state_after=data.get("state_after", {}),
            metadata=data.get("metadata", {}),
        )


def parse_trace_events(
    events: Sequence[Any],
    *,
    allow_serialized_trust: bool = False,
) -> list[TraceEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("trace_events must be a sequence of mappings or TraceEvent objects")
    return [
        (
            event
            if isinstance(event, TraceEvent)
            else TraceEvent.from_dict(
                event,
                index,
                allow_trusted=allow_serialized_trust,
            )
        )
        for index, event in enumerate(events)
    ]


@dataclass(frozen=True)
class TraceValidationConfig:
    require_trusted: bool = True
    trusted_sources: tuple[str, ...] = ("runtime", "external")
    allow_serialized_trust: bool = False
    require_node_ids: bool = True
    require_start: bool = True
    require_terminal: bool = True
    max_field_chars: int = 2000

    def __post_init__(self) -> None:
        if not self.trusted_sources:
            raise ValueError("trusted_sources must not be empty")
        if any(source not in _EVENT_SOURCES for source in self.trusted_sources):
            raise ValueError("trusted_sources contains an unsupported source")
        if self.max_field_chars <= 0:
            raise ValueError("max_field_chars must be positive")


@dataclass
class TraceValidation:
    violations: list[str]
    warnings: list[str]
    node_path: list[str]
    occurrence_path: list[str]
    edge_kinds: list[str]
    transition_counts: dict[str, int]
    state_observations: dict[str, dict]
    mapped_events: int
    trusted_events: int
    total_events: int
    terminal_reached: bool
    checks_passed: int
    checks_total: int

    @property
    def valid(self) -> bool:
        return not self.violations

    @property
    def validity_score(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total else 0.0

    def add_violation(self, reason: str) -> None:
        if reason not in self.violations:
            self.violations.append(reason)
        self.checks_total += 1

    def to_dict(self) -> dict:
        output = asdict(self)
        output["valid"] = self.valid
        output["validity_score"] = self.validity_score
        return output


@dataclass
class _Visit:
    node_id: str
    occurrence: int
    occurrence_key: str
    events: list[TraceEvent]


def validate_trace_events(
    events: Sequence[Any],
    workflow: WorkflowGroundTruth,
    config: Optional[TraceValidationConfig] = None,
) -> TraceValidation:
    """Validate trusted events against graph topology and finite loop bounds."""
    config = config or TraceValidationConfig()
    parsed = parse_trace_events(
        events,
        allow_serialized_trust=config.allow_serialized_trust,
    )
    workflow.validate()
    node_map = workflow.node_map
    violations: list[str] = []
    warnings: list[str] = []
    checks_passed = 0
    checks_total = 0

    def check(condition: bool, reason: str) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
        elif reason not in violations:
            violations.append(reason)

    event_ids = [event.event_id for event in parsed]
    check(len(event_ids) == len(set(event_ids)), "duplicate_event_id")

    trusted_events = 0
    for event in parsed:
        trusted = event.trusted and event.source in config.trusted_sources
        if trusted:
            trusted_events += 1
        if config.require_trusted:
            check(trusted, f"untrusted_event:{event.event_id}")

    mapped_events = 0
    raw_mapped = []
    inferred_occurrences: dict[str, int] = {}
    previous_node = None
    previous_occurrence = None
    for event in parsed:
        if event.node_id is None:
            if config.require_node_ids:
                check(False, f"missing_node_id:{event.event_id}")
            continue
        mapped_events += 1
        if event.node_id not in node_map:
            check(False, f"unknown_node:{event.node_id}")
            continue

        occurrence = event.occurrence
        if occurrence is None:
            if previous_node == event.node_id and previous_occurrence is not None:
                occurrence = previous_occurrence
            else:
                occurrence = inferred_occurrences.get(event.node_id, 0) + 1
        inferred_occurrences[event.node_id] = max(
            inferred_occurrences.get(event.node_id, 0),
            occurrence,
        )
        raw_mapped.append((event, occurrence))
        previous_node = event.node_id
        previous_occurrence = occurrence
        checks_passed += 1
        checks_total += 1

    visits: list[_Visit] = []
    for event, occurrence in raw_mapped:
        occurrence_key = f"{event.node_id}#{occurrence}"
        if visits and visits[-1].occurrence_key == occurrence_key:
            visits[-1].events.append(event)
        else:
            visits.append(
                _Visit(
                    node_id=event.node_id,
                    occurrence=occurrence,
                    occurrence_key=occurrence_key,
                    events=[event],
                )
            )

    node_path = [visit.node_id for visit in visits]
    occurrence_path = [visit.occurrence_key for visit in visits]
    if not node_path:
        check(False, "no_mapped_workflow_path")
    elif config.require_start:
        check(node_path[0] == workflow.start, f"wrong_start:{node_path[0]}")

    node_visit_counts: dict[str, int] = {}
    for visit in visits:
        node_visit_counts[visit.node_id] = node_visit_counts.get(visit.node_id, 0) + 1
        check(
            visit.occurrence == node_visit_counts[visit.node_id],
            (
                f"non_sequential_occurrence:{visit.node_id}:"
                f"{visit.occurrence}!={node_visit_counts[visit.node_id]}"
            ),
        )
    for node_id, count in node_visit_counts.items():
        check(
            count <= node_map[node_id].max_visits,
            f"node_visit_limit_exceeded:{node_id}:{count}>{node_map[node_id].max_visits}",
        )

    pair_edges: dict[tuple[str, str], list[tuple[int, Any]]] = {}
    for edge_index, edge in enumerate(workflow.edges):
        pair_edges.setdefault((edge.source, edge.target), []).append((edge_index, edge))

    selected_edges = []
    edge_visits: dict[int, int] = {}
    for left, right in zip(visits, visits[1:]):
        candidates = pair_edges.get((left.node_id, right.node_id), [])
        requested_kinds = {
            event.transition_kind
            for event in right.events
            if event.transition_kind is not None
        }
        check(
            len(requested_kinds) <= 1,
            f"conflicting_transition_kinds:{right.occurrence_key}",
        )
        requested_kind = next(iter(requested_kinds), None)
        if requested_kind is not None:
            candidates = [
                candidate for candidate in candidates if candidate[1].kind == requested_kind
            ]
        if not candidates:
            suffix = f":{requested_kind}" if requested_kind else ""
            check(
                False,
                f"illegal_transition:{left.node_id}->{right.node_id}{suffix}",
            )
            continue
        if len(candidates) > 1 and requested_kind is None:
            warnings.append(f"ambiguous_transition:{left.node_id}->{right.node_id}")
        edge_index, edge = candidates[0]
        selected_edges.append(edge)
        edge_visits[edge_index] = edge_visits.get(edge_index, 0) + 1
        check(
            edge_visits[edge_index] <= edge.max_traversals,
            (
                f"edge_traversal_limit_exceeded:{edge.source}->{edge.target}:"
                f"{edge_visits[edge_index]}>{edge.max_traversals}"
            ),
        )

    terminal_reached = bool(node_path and node_path[-1] in workflow.terminals)
    if config.require_terminal:
        terminal_label = node_path[-1] if node_path else "<none>"
        check(terminal_reached, f"terminal_not_reached:{terminal_label}")

    state_observations = {}
    for visit in visits:
        before = {}
        after = {}
        for event in visit.events:
            if event.state_before and not before:
                before = dict(event.state_before)
            if event.state_after:
                after = dict(event.state_after)
        if before or after:
            state_observations[visit.occurrence_key] = {
                "before": before,
                "after": after,
            }

    edge_kinds = [edge.kind for edge in selected_edges]
    transition_counts = {
        kind: edge_kinds.count(kind)
        for kind in sorted(set(edge_kinds))
    }
    return TraceValidation(
        violations=violations,
        warnings=warnings,
        node_path=node_path,
        occurrence_path=occurrence_path,
        edge_kinds=edge_kinds,
        transition_counts=transition_counts,
        state_observations=state_observations,
        mapped_events=mapped_events,
        trusted_events=trusted_events,
        total_events=len(parsed),
        terminal_reached=terminal_reached,
        checks_passed=checks_passed,
        checks_total=checks_total,
    )


def render_trace_events(
    events: Sequence[Any],
    *,
    max_field_chars: int = 2000,
) -> str:
    """Render structured runtime events into deterministic scorer input."""
    if max_field_chars <= 0:
        raise ValueError("max_field_chars must be positive")
    parsed = parse_trace_events(events)
    rendered = []
    for event in parsed:
        attributes = [
            f"id={_clip(event.event_id, max_field_chars)}",
            f"source={event.source}",
            f"status={event.status}",
        ]
        if event.node_id:
            attributes.append(f"node={_clip(event.node_id, max_field_chars)}")
        if event.occurrence is not None:
            attributes.append(f"occurrence={event.occurrence}")
        if event.transition_kind:
            attributes.append(f"transition={event.transition_kind}")
        lines = [f"<trace_event {' '.join(attributes)}>"]
        lines.append(f"Action: {_clip(event.action, max_field_chars)}")
        if event.tool_name:
            lines.append(f"Tool: {_clip(event.tool_name, max_field_chars)}")
        if event.tool_input is not None:
            lines.append(
                f"Tool input: {_clip(_stable_text(event.tool_input), max_field_chars)}"
            )
        if event.evidence:
            lines.append(f"Evidence: {_clip(event.evidence, max_field_chars)}")
        if event.tool_output is not None:
            lines.append(
                f"Tool output: {_clip(_stable_text(event.tool_output), max_field_chars)}"
            )
        if event.outcome:
            lines.append(f"Outcome: {_clip(event.outcome, max_field_chars)}")
        lines.append("</trace_event>")
        rendered.append("\n".join(lines))
    return "\n".join(rendered)


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"
