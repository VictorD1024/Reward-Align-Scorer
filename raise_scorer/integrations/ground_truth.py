"""GroundTruth extraction from RLVR/veRL row payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


_STEP_SPLIT_RE = re.compile(r"[，,；;\n]")
_CANONICAL_EXTRA_INFO_PATHS = (
    ("reference_steps",),
    ("actions",),
)
_NESTED_EXTRA_INFO_PATHS = (
    ("metadata", "reference_steps"),
    ("metadata", "actions"),
    ("rlvr", "reference_steps"),
    ("rlvr", "actions"),
    ("ground_truth", "reference_steps"),
    ("ground_truth", "actions"),
    ("reward_model", "ground_truth", "reference_steps"),
    ("reward_model", "ground_truth", "actions"),
)
_GROUND_TRUTH_KEYS = ("reference_steps", "actions", "steps")


@dataclass(frozen=True)
class ReferenceStepsResolution:
    """Normalized steps plus the RLVR field path that supplied them."""

    steps: list
    source: Optional[str]

    @property
    def found(self) -> bool:
        return bool(self.steps)


def resolve_reference_steps(
    extra_info: Optional[Mapping[str, Any]] = None,
    ground_truth: Any = None,
    *,
    field_paths: Optional[Sequence[str | Sequence[str]]] = None,
) -> ReferenceStepsResolution:
    """Resolve reference steps from values delivered to a reward callback.

    Priority is:

    1. canonical ``extra_info.reference_steps``;
    2. backward-compatible ``extra_info.actions``;
    3. caller-configured and known nested ``extra_info`` paths;
    4. structured ``reward_model.ground_truth`` passed as ``ground_truth``.

    Plain scalar ground-truth strings are intentionally ignored because they
    normally contain an RLVR final answer rather than process steps. JSON
    arrays/objects and native list/struct values are accepted.
    """
    if extra_info is None:
        extra_info = {}
    if not isinstance(extra_info, Mapping):
        raise TypeError("extra_info must be a mapping or None")

    extra_paths = [
        *_CANONICAL_EXTRA_INFO_PATHS,
        *_normalize_field_paths(field_paths),
        *_NESTED_EXTRA_INFO_PATHS,
    ]
    seen = set()
    for path in extra_paths:
        if path in seen:
            continue
        seen.add(path)
        value = _lookup_path(extra_info, path)
        if value is _MISSING:
            continue
        steps = normalize_reference_steps(value, allow_plain_text=True)
        if steps:
            return ReferenceStepsResolution(
                steps=steps,
                source=f"extra_info.{'.'.join(path)}",
            )

    structured_ground_truth = _decode_json_container(ground_truth)
    if isinstance(structured_ground_truth, Mapping):
        for key in _GROUND_TRUTH_KEYS:
            if key not in structured_ground_truth:
                continue
            steps = normalize_reference_steps(
                structured_ground_truth[key],
                allow_plain_text=True,
            )
            if steps:
                return ReferenceStepsResolution(
                    steps=steps,
                    source=f"ground_truth.{key}",
                )
    elif _is_non_string_sequence(structured_ground_truth):
        steps = normalize_reference_steps(
            structured_ground_truth,
            allow_plain_text=False,
        )
        if steps:
            return ReferenceStepsResolution(
                steps=steps,
                source="ground_truth",
            )

    return ReferenceStepsResolution(steps=[], source=None)


def normalize_reference_steps(value: Any, *, allow_plain_text: bool = True) -> list:
    """Normalize Arrow/Pandas/Python step containers for the core scorer."""
    value = _to_python_container(value)
    decoded = _decode_json_container(value)
    if decoded is not value:
        return normalize_reference_steps(
            decoded,
            allow_plain_text=allow_plain_text,
        )

    if isinstance(value, str):
        if not allow_plain_text:
            return []
        return [
            part.strip()
            for part in _STEP_SPLIT_RE.split(value)
            if part.strip()
        ]
    if not _is_non_string_sequence(value):
        return []

    normalized = []
    for item in value:
        item = _to_python_container(item)
        if isinstance(item, Mapping):
            if item.get("action") is not None:
                item = item["action"]
            elif item.get("actions") is not None:
                item = item["actions"]
            else:
                continue
        if _is_non_string_sequence(item):
            candidates = [
                str(candidate).strip()
                for candidate in item
                if str(candidate).strip()
            ]
            if candidates:
                normalized.append(candidates)
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_field_paths(
    field_paths: Optional[Sequence[str | Sequence[str]]],
) -> list[tuple[str, ...]]:
    if field_paths is None:
        return []
    if isinstance(field_paths, str):
        field_paths = [
            path.strip()
            for path in field_paths.split(",")
            if path.strip()
        ]
    if not isinstance(field_paths, Sequence):
        raise TypeError("reference_steps_paths must be a sequence or comma-separated string")

    normalized = []
    for path in field_paths:
        if isinstance(path, str):
            parts = tuple(part.strip() for part in path.split(".") if part.strip())
        elif _is_non_string_sequence(path):
            parts = tuple(str(part).strip() for part in path if str(part).strip())
        else:
            raise TypeError("each reference steps path must be a string or sequence")
        if not parts:
            raise ValueError("reference steps path must not be empty")
        normalized.append(parts)
    return normalized


class _Missing:
    pass


_MISSING = _Missing()


def _lookup_path(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _decode_json_container(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return value
    return decoded if isinstance(decoded, (list, dict)) else value


def _to_python_container(value: Any) -> Any:
    if isinstance(value, (str, bytes, Mapping, list, tuple)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
