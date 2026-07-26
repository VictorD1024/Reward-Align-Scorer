from __future__ import annotations

import os
import re
from collections import Counter
from typing import Optional

from ..backends.embedding import load_embedding_backend
from ..core.confidence import ConfidenceReport, assess_confidence
from ..core.scorer import ScorerConfig, SemanticRewardScorer
from ..runtime.adapters import RuntimeEventAdapter
from ..runtime.events import TraceValidationConfig, render_trace_events
from ..workflows.engine import WorkflowGroundTruth, WorkflowScorer, WorkflowScorerConfig
from .ground_truth import resolve_reference_steps


_SCORER: Optional[SemanticRewardScorer] = None


def _env(name: str, default: str = "", legacy: str | None = None) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    if legacy:
        return os.environ.get(legacy, default)
    return default


def _optional_env(name: str) -> str | None:
    return os.environ[name] if name in os.environ else None


def _get_scorer() -> Optional[SemanticRewardScorer]:
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    model_path = _env("RAISE_MODEL_PATH", legacy="REWARD_ALIGN_MODEL_PATH")
    if not model_path:
        return None
    backend = load_embedding_backend(
        model_path,
        pooling=_optional_env("RAISE_POOLING"),
        query_instruction=_optional_env("RAISE_QUERY_INSTRUCTION"),
        passage_instruction=_optional_env("RAISE_PASSAGE_INSTRUCTION"),
        revision=_optional_env("RAISE_MODEL_REVISION"),
        local_files_only=_env("RAISE_LOCAL_FILES_ONLY", "0") == "1",
    )
    if backend is None:
        return None
    _SCORER = SemanticRewardScorer(
        backend,
        ScorerConfig(
            threshold=float(_env("RAISE_THRESHOLD", "0.65", legacy="REWARD_ALIGN_THRESHOLD")),
            max_score=float(_env("RAISE_MAX_SCORE", "1.0", legacy="REWARD_ALIGN_MAX_SCORE")),
            max_length=int(_env("RAISE_MAX_LENGTH", "128", legacy="REWARD_ALIGN_MAX_LENGTH")),
            encode_batch_size=int(_env("RAISE_BATCH_SIZE", "128", legacy="REWARD_ALIGN_BATCH_SIZE")),
            require_trace=_env("RAISE_REQUIRE_TRACE", "0", legacy="REWARD_ALIGN_REQUIRE_TRACE") == "1",
        ),
    )
    return _SCORER


def parse_reference_steps(
    extra_info: dict,
    ground_truth=None,
    *,
    field_paths=None,
) -> list:
    """Read reference steps from veRL/RLVR reward inputs.

    This compatibility wrapper returns only the normalized steps. Use
    :func:`resolve_reference_steps` when source-path diagnostics are needed.
    """
    return resolve_reference_steps(
        extra_info,
        ground_truth,
        field_paths=field_paths,
    ).steps


def _extract_response(solution_str: str) -> tuple[str, str]:
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if not think_match:
        return "", solution_str.strip()
    think_text = think_match.group(1).strip()
    answer_text = solution_str[solution_str.find("</think>") + len("</think>"):].strip()
    return think_text, answer_text


def _repetition_penalty(text: str) -> tuple[float, float]:
    pieces = [s.strip() for s in re.split(r"[，,。；;！!？?\n\s]", text or "") if len(s.strip()) > 3]
    if len(pieces) <= 1:
        return 0.0, 0.0
    consecutive = sum(1 for i in range(len(pieces) - 1) if pieces[i] == pieces[i + 1])
    counts = Counter(pieces)
    duplicate = sum(count - 1 for count in counts.values() if count > 1)
    penalty = -((consecutive * 0.3) + (max(0, duplicate - consecutive) * 0.1))
    if len(set(pieces)) / len(pieces) < 0.3:
        penalty -= 0.5
    return duplicate / len(pieces), penalty


def compute_score(
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    return_details=False,
    *,
    data_source=None,
    **kwargs,
):
    """veRL reward_score-compatible compute_score implementation.

    This function is meant to be imported by a thin file under
    ``verl/utils/reward_score``. It does not replace veRL's own
    ``utils/reward_score/__init__.py`` dispatcher.

    Current veRL reward managers call custom functions with the named
    arguments ``data_source``, ``solution_str``, ``ground_truth``, and
    ``extra_info``. ``data_source`` and additional reward kwargs are accepted
    for framework compatibility but are not used by the semantic scorer.

    Keeping ``solution_str`` as the first positional argument preserves the
    package's original direct-call API::

        compute_score(response, extra_info={"reference_steps": steps})
    """
    reference_steps_paths = kwargs.pop("reference_steps_paths", None)
    del kwargs
    extra_info = extra_info or {}
    if not isinstance(extra_info, dict):
        raise TypeError("extra_info must be a dict or None")
    solution_str = "" if solution_str is None else str(solution_str)
    think_text, answer_text = _extract_response(solution_str)
    steps_resolution = resolve_reference_steps(
        extra_info,
        ground_truth,
        field_paths=reference_steps_paths,
    )
    steps = steps_resolution.steps
    workflow_data = extra_info.get("workflow_ground_truth")
    if workflow_data is None and isinstance(ground_truth, dict):
        workflow_data = ground_truth.get("workflow_ground_truth")
    trace_events = extra_info.get("trace_events")
    tool_calls = extra_info.get("tool_calls")
    tool_calls_adapted = False

    details = {
        "score": 0.0,
        "data_source": data_source,
        "reference_steps_source": steps_resolution.source,
        "semantic_score": 0.0,
        "step_match_rate": 0.0,
        "step_order_rate": 0.0,
        "matched_steps": [],
        "unmatched_steps": [_step_label(s) for s in steps],
        "repetition_rate": 0.0,
        "penalty": 0.0,
        "think_len": len(think_text),
        "confidence": 0.0,
        "fallback_recommended": True,
        "fallback_reasons": [],
        "step_tiers": [],
        "routing_signals": {},
        "workflow": None,
        "debug": {},
    }

    semantic = None
    workflow_result = None
    routing_steps = steps
    routing_response = answer_text
    scorer = _get_scorer()
    if (
        scorer is not None
        and workflow_data is not None
        and (answer_text or trace_events is not None or tool_calls is not None)
    ):
        workflow = (
            workflow_data
            if isinstance(workflow_data, WorkflowGroundTruth)
            else WorkflowGroundTruth.from_dict(workflow_data)
        )
        workflow_config_data = extra_info.get("workflow_config")
        workflow_config = (
            WorkflowScorerConfig(**workflow_config_data)
            if workflow_config_data is not None
            else None
        )
        workflow_scorer = WorkflowScorer(scorer, workflow_config)
        if trace_events is None and tool_calls is not None:
            adapter_config = extra_info.get("tool_adapter_config") or {}
            if not isinstance(adapter_config, dict):
                raise TypeError("tool_adapter_config must be a dict or None")
            trace_events = RuntimeEventAdapter(
                trusted_runtime=adapter_config.get("trusted_runtime") is True,
                redact_sensitive=(
                    adapter_config.get("redact_sensitive", True) is not False
                ),
            ).adapt_many(tool_calls)
            tool_calls_adapted = True
        if trace_events is not None:
            validation_config_data = extra_info.get("trace_validation_config")
            validation_config = (
                TraceValidationConfig(**validation_config_data)
                if validation_config_data is not None
                else TraceValidationConfig()
            )
            workflow_result = workflow_scorer.score_events(
                trace_events,
                workflow,
                validation_config=validation_config,
                state_observations=extra_info.get("state_observations"),
            )
            routing_response = render_trace_events(
                trace_events,
                max_field_chars=validation_config.max_field_chars,
            )
        else:
            workflow_result = workflow_scorer.score(
                answer_text,
                workflow,
                state_observations=extra_info.get("state_observations"),
            )
        semantic = workflow_result.semantic_result
        routing_steps = [
            _field_reference(field.reference)
            for node in workflow_result.node_rewards
            if node.required
            for field in node.fields.values()
        ]
        details["semantic_score"] = workflow_result.score
        details["step_match_rate"] = workflow_result.required_coverage
        details["step_order_rate"] = workflow_result.order_rate
        details["matched_steps"] = semantic.matched_steps
        details["unmatched_steps"] = semantic.unmatched_steps
        details["workflow"] = workflow_result.to_dict()
        details["debug"] = {
            **semantic.stats,
            "workflow_best_path": workflow_result.best_path,
            "workflow_candidate_paths": workflow_result.candidate_paths,
            "workflow_paths_truncated": workflow_result.paths_truncated,
            "workflow_state_score": workflow_result.state_score,
            "workflow_state_gate_passed": workflow_result.state_gate_passed,
            "workflow_trace_valid": (
                workflow_result.trace_validation.valid
                if workflow_result.trace_validation is not None
                else None
            ),
            "workflow_tool_calls_adapted": tool_calls_adapted,
        }
    elif scorer is not None and steps and answer_text:
        semantic = scorer.score(answer_text, steps)
        details["semantic_score"] = semantic.score
        details["step_match_rate"] = semantic.match_rate
        details["step_order_rate"] = semantic.order_rate
        details["matched_steps"] = semantic.matched_steps
        details["unmatched_steps"] = semantic.unmatched_steps
        details["debug"] = semantic.stats
        if semantic.stats.get("trace_gate") == "denied":
            details["trace_gate"] = "denied"
    elif steps and answer_text:
        matches = sum(1 for step in steps if _step_label(step) in answer_text)
        details["step_match_rate"] = matches / len(steps)
        details["step_order_rate"] = details["step_match_rate"]
        details["semantic_score"] = details["step_match_rate"]
        details["fallback_reasons"] = ["no_embedding_backend"]

    details["repetition_rate"], details["penalty"] = _repetition_penalty(answer_text)
    details["score"] = max(details["semantic_score"] + details["penalty"], 0.0)

    if semantic is not None:
        report = assess_confidence(semantic, routing_steps, response=routing_response)
        if workflow_result is not None and workflow_result.state_gate_passed is False:
            report.fallback_recommended = True
            report.reasons.append("workflow_state_gate_failed")
        if workflow_result is not None and workflow_result.paths_truncated:
            report.fallback_recommended = True
            report.reasons.append("workflow_paths_truncated")
        if (
            workflow_result is not None
            and workflow_result.trace_validation is not None
            and not workflow_result.trace_validation.valid
        ):
            report.fallback_recommended = True
            report.reasons.append("workflow_trace_invalid")
    elif (not steps and workflow_data is None) or not answer_text:
        report = ConfidenceReport(
            confidence=0.0,
            fallback_recommended=True,
            reasons=["missing_steps_or_response"],
        )
    else:
        report = ConfidenceReport(
            confidence=0.0,
            fallback_recommended=True,
            reasons=details["fallback_reasons"] or ["no_embedding_backend"],
        )

    details["confidence"] = report.confidence
    details["fallback_recommended"] = report.fallback_recommended
    details["fallback_reasons"] = report.reasons
    details["step_tiers"] = report.step_tiers
    details["routing_signals"] = report.signals

    if return_details:
        return details
    return details["score"]


def _step_label(step) -> str:
    if isinstance(step, list):
        return " | ".join(str(s).strip() for s in step if str(s).strip())
    return str(step).strip()


def _field_reference(reference) -> list[str] | str:
    return list(reference) if isinstance(reference, tuple) else str(reference)
