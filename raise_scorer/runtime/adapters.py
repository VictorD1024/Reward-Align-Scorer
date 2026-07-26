"""Zero-dependency adapters from common Agent tool calls to TraceEvent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .events import TraceEvent


_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}
_PYTEST_SUMMARY_RE = re.compile(
    r"\b(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)\b",
    re.IGNORECASE,
)


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class ToolCallRecord:
    """Normalized runtime-owned tool call/result pair.

    Trust is intentionally absent from this schema. Only a
    :class:`RuntimeEventAdapter` created at a trusted infrastructure boundary
    can attach ``TraceEvent.trusted=True``.
    """

    call_id: str
    tool_name: str
    arguments: Any = None
    output: Any = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    node_id: Optional[str] = None
    occurrence: Optional[int] = None
    transition_kind: Optional[str] = None
    state_before: Mapping[str, Any] = field(default_factory=dict)
    state_after: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.call_id = str(self.call_id).strip()
        self.tool_name = str(self.tool_name).strip()
        self.error = _optional_text(self.error)
        self.node_id = _optional_text(self.node_id)
        self.transition_kind = _optional_text(self.transition_kind)
        if self.transition_kind is not None:
            self.transition_kind = self.transition_kind.lower()
        self.state_before = dict(self.state_before or {})
        self.state_after = dict(self.state_after or {})
        self.metadata = dict(self.metadata or {})
        if not self.call_id:
            raise ValueError("tool call_id must not be empty")
        if not self.tool_name:
            raise ValueError(f"tool call {self.call_id!r} tool_name must not be empty")
        if self.occurrence is not None and self.occurrence <= 0:
            raise ValueError("tool call occurrence must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int = 0) -> "ToolCallRecord":
        if not isinstance(data, Mapping):
            raise TypeError("tool call must be a mapping")
        output = data.get("output", data.get("result"))
        exit_code = data.get("exit_code")
        if exit_code is None and isinstance(output, Mapping):
            exit_code = output.get("exit_code", output.get("returncode"))
        return cls(
            call_id=data.get("call_id", data.get("id", f"tool-call-{index}")),
            tool_name=data.get("tool_name", data.get("name", data.get("tool", ""))),
            arguments=data.get("arguments", data.get("input")),
            output=output,
            error=data.get("error"),
            exit_code=int(exit_code) if exit_code is not None else None,
            node_id=data.get("node_id"),
            occurrence=(
                int(data["occurrence"])
                if data.get("occurrence") is not None
                else None
            ),
            transition_kind=data.get("transition_kind"),
            state_before=data.get("state_before", {}),
            state_after=data.get("state_after", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class EventSemantics:
    action: str
    evidence: Optional[str]
    outcome: Optional[str]
    status: str


AdapterHandler = Callable[[ToolCallRecord], EventSemantics]
NodeResolver = Callable[[ToolCallRecord, EventSemantics], Optional[str]]


class RuntimeEventAdapter:
    """Convert tool records to events while centralizing trust assignment."""

    def __init__(
        self,
        *,
        trusted_runtime: bool = False,
        node_resolver: Optional[NodeResolver] = None,
        redact_sensitive: bool = True,
    ):
        self.trusted_runtime = trusted_runtime
        self.node_resolver = node_resolver
        self.redact_sensitive = redact_sensitive
        self._handlers: dict[str, AdapterHandler] = {}

    def register(
        self,
        tool_names: str | Sequence[str],
        handler: AdapterHandler,
    ) -> None:
        names = [tool_names] if isinstance(tool_names, str) else list(tool_names)
        if not names:
            raise ValueError("tool_names must not be empty")
        for name in names:
            normalized = str(name).strip().casefold()
            if not normalized:
                raise ValueError("tool name must not be empty")
            self._handlers[normalized] = handler

    def adapt(self, record: ToolCallRecord | Mapping[str, Any], index: int = 0) -> TraceEvent:
        record = (
            record
            if isinstance(record, ToolCallRecord)
            else ToolCallRecord.from_dict(record, index)
        )
        handler = self._select_handler(record)
        semantics = handler(record)
        if not isinstance(semantics, EventSemantics):
            raise TypeError("runtime adapter handler must return EventSemantics")
        node_id = (
            self.node_resolver(record, semantics)
            if self.node_resolver is not None
            else record.node_id
        )
        tool_input = (
            _redact(record.arguments)
            if self.redact_sensitive
            else record.arguments
        )
        tool_output = (
            _redact(record.output)
            if self.redact_sensitive
            else record.output
        )
        return TraceEvent(
            event_id=record.call_id,
            action=semantics.action,
            node_id=node_id,
            occurrence=record.occurrence,
            evidence=semantics.evidence,
            outcome=semantics.outcome,
            status=semantics.status,
            source="runtime",
            trusted=self.trusted_runtime,
            transition_kind=record.transition_kind,
            tool_name=record.tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            state_before=record.state_before,
            state_after=record.state_after,
            metadata=(
                _redact(record.metadata)
                if self.redact_sensitive
                else record.metadata
            ),
        )

    def adapt_many(self, records: Sequence[Any]) -> list[TraceEvent]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("tool_calls must be a sequence")
        return [self.adapt(record, index) for index, record in enumerate(records)]

    def _select_handler(self, record: ToolCallRecord) -> AdapterHandler:
        name = record.tool_name.casefold()
        if name in self._handlers:
            return self._handlers[name]
        command = _command(record.arguments).casefold()
        if "pytest" in name or "pytest" in command:
            return _pytest_semantics
        if _matches_tool_name(name, {
            "shell",
            "bash",
            "exec",
            "exec_command",
            "terminal",
            "run_command",
        }):
            return _shell_semantics
        if _matches_tool_name(name, {"read_file", "open_file", "cat_file"}):
            return _file_read_semantics
        if _matches_tool_name(name, {
            "write_file",
            "edit_file",
            "apply_patch",
            "patch_file",
        }):
            return _file_write_semantics
        if _matches_tool_name(name, {
            "browser",
            "browser_search",
            "search",
            "open_url",
            "navigate",
            "click",
        }) or "browser" in _tool_name_parts(name):
            return _browser_semantics
        if _matches_tool_name(
            name,
            {"http", "http_request", "request", "fetch"},
        ):
            return _http_semantics
        return _generic_semantics


def adapt_tool_calls(
    records: Sequence[Any],
    *,
    trusted_runtime: bool = False,
    node_resolver: Optional[NodeResolver] = None,
    redact_sensitive: bool = True,
) -> list[TraceEvent]:
    """Convenience wrapper using the built-in adapter registry."""
    return RuntimeEventAdapter(
        trusted_runtime=trusted_runtime,
        node_resolver=node_resolver,
        redact_sensitive=redact_sensitive,
    ).adapt_many(records)


def _pytest_semantics(record: ToolCallRecord) -> EventSemantics:
    command = _command(record.arguments)
    output_text = _output_text(record.output)
    summary = ", ".join(
        f"{count} {label.lower()}"
        for count, label in _PYTEST_SUMMARY_RE.findall(output_text)
    )
    failed_summary = any(
        label.lower() in {"failed", "error", "errors"} and int(count) > 0
        for count, label in _PYTEST_SUMMARY_RE.findall(output_text)
    )
    failed = record.error is not None or (
        record.exit_code is not None and record.exit_code != 0
    ) or failed_summary
    evidence_parts = []
    if command:
        evidence_parts.append(f"pytest command: {command}")
    if summary:
        evidence_parts.append(f"pytest result: {summary}")
    if record.exit_code is not None:
        evidence_parts.append(f"exit code: {record.exit_code}")
    if record.error:
        evidence_parts.append(f"error: {record.error}")
    return EventSemantics(
        action="run tests",
        evidence="; ".join(evidence_parts) or "pytest executed",
        outcome="tests fail" if failed else "tests pass",
        status="failure" if failed else "success",
    )


def _shell_semantics(record: ToolCallRecord) -> EventSemantics:
    command = _command(record.arguments)
    failed = record.error is not None or (
        record.exit_code is not None and record.exit_code != 0
    )
    evidence = f"command: {command}" if command else "shell command executed"
    if record.exit_code is not None:
        evidence += f"; exit code: {record.exit_code}"
    if record.error:
        evidence += f"; error: {record.error}"
    return EventSemantics(
        action="run shell command",
        evidence=evidence,
        outcome="command failed" if failed else "command succeeded",
        status="failure" if failed else "success",
    )


def _file_read_semantics(record: ToolCallRecord) -> EventSemantics:
    path = _path(record.arguments)
    failed = record.error is not None or (
        record.exit_code is not None and record.exit_code != 0
    )
    return EventSemantics(
        action=f"read file {path}" if path else "read file",
        evidence=(
            f"file path: {path}"
            if path
            else record.error or "file read requested"
        ),
        outcome="file read failed" if failed else "file contents observed",
        status="failure" if failed else "success",
    )


def _file_write_semantics(record: ToolCallRecord) -> EventSemantics:
    path = _path(record.arguments)
    failed = record.error is not None or (
        record.exit_code is not None and record.exit_code != 0
    )
    action = f"modify file {path}" if path else "modify repository files"
    evidence = f"file path: {path}" if path else "patch or file update submitted"
    if record.error:
        evidence += f"; error: {record.error}"
    return EventSemantics(
        action=action,
        evidence=evidence,
        outcome="file modification failed" if failed else "file modified",
        status="failure" if failed else "success",
    )


def _browser_semantics(record: ToolCallRecord) -> EventSemantics:
    name = record.tool_name.casefold()
    arguments = record.arguments if isinstance(record.arguments, Mapping) else {}
    operation = str(arguments.get("operation", arguments.get("action", name))).casefold()
    if "search" in operation or "search" in name:
        action = "search the web"
    elif "click" in operation or "click" in name:
        action = "interact with web page"
    else:
        action = "open web page"
    target = (
        arguments.get("url")
        or arguments.get("query")
        or arguments.get("target")
    )
    failed = record.error is not None or (
        record.exit_code is not None and record.exit_code != 0
    )
    evidence = f"browser target: {target}" if target else "browser action recorded"
    if record.error:
        evidence += f"; error: {record.error}"
    return EventSemantics(
        action=action,
        evidence=evidence,
        outcome="browser action failed" if failed else "browser action completed",
        status="failure" if failed else "success",
    )


def _http_semantics(record: ToolCallRecord) -> EventSemantics:
    arguments = record.arguments if isinstance(record.arguments, Mapping) else {}
    method = str(arguments.get("method", "GET")).upper()
    url = arguments.get("url")
    status_code = record.metadata.get("status_code")
    if status_code is None and isinstance(record.output, Mapping):
        status_code = record.output.get("status_code", record.output.get("status"))
    try:
        numeric_status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        numeric_status = None
    failed = (
        record.error is not None
        or (record.exit_code is not None and record.exit_code != 0)
        or (numeric_status is not None and numeric_status >= 400)
    )
    evidence_parts = []
    if url:
        evidence_parts.append(f"url: {url}")
    if numeric_status is not None:
        evidence_parts.append(f"status code: {numeric_status}")
    if record.exit_code is not None:
        evidence_parts.append(f"exit code: {record.exit_code}")
    if record.error:
        evidence_parts.append(f"error: {record.error}")
    return EventSemantics(
        action=f"send {method} HTTP request",
        evidence="; ".join(evidence_parts) or "HTTP request recorded",
        outcome="HTTP request failed" if failed else "HTTP request succeeded",
        status="failure" if failed else "success",
    )


def _generic_semantics(record: ToolCallRecord) -> EventSemantics:
    failed = record.error is not None or (
        record.exit_code is not None and record.exit_code != 0
    )
    evidence_parts = [f"tool: {record.tool_name}"]
    if record.exit_code is not None:
        evidence_parts.append(f"exit code: {record.exit_code}")
    if record.error:
        evidence_parts.append(f"error: {record.error}")
    return EventSemantics(
        action=f"use tool {record.tool_name}",
        evidence="; ".join(evidence_parts),
        outcome="tool call failed" if failed else "tool call succeeded",
        status="failure" if failed else "success",
    )


def _command(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments.strip()
    if isinstance(arguments, Mapping):
        value = arguments.get("command", arguments.get("cmd", ""))
        return str(value).strip()
    return ""


def _path(arguments: Any) -> str:
    if not isinstance(arguments, Mapping):
        return ""
    value = (
        arguments.get("path")
        or arguments.get("file_path")
        or arguments.get("file")
    )
    return str(value).strip() if value is not None else ""


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        parts = [
            str(value[key])
            for key in ("stdout", "stderr", "text", "output", "content")
            if value.get(key) is not None
        ]
        if parts:
            return "\n".join(parts)
    return str(value)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                "[REDACTED]"
                if _is_sensitive_key(key)
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    return value


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).casefold()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_token", "_secret", "_password", "_key")
    )


def _tool_name_parts(name: str) -> set[str]:
    return {
        part
        for part in re.split(r"[^a-z0-9]+", name.casefold())
        if part
    }


def _matches_tool_name(name: str, aliases: set[str]) -> bool:
    normalized = name.casefold()
    return any(
        normalized == alias
        or normalized.endswith(
            (
                f".{alias}",
                f"/{alias}",
                f":{alias}",
                f"__{alias}",
                f"-{alias}",
            )
        )
        for alias in aliases
    )
