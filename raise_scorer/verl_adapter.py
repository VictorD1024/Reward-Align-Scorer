from __future__ import annotations

import os
import re
from collections import Counter
from typing import Optional

from .embedding import load_embedding_backend
from .confidence import assess_confidence, ConfidenceReport
from .scorer import ScorerConfig, SemanticRewardScorer


_SCORER: Optional[SemanticRewardScorer] = None


def _env(name: str, default: str = "", legacy: str | None = None) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    if legacy:
        return os.environ.get(legacy, default)
    return default


def _get_scorer() -> Optional[SemanticRewardScorer]:
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    model_path = _env("RAISE_MODEL_PATH", legacy="REWARD_ALIGN_MODEL_PATH")
    if not model_path:
        return None
    backend = load_embedding_backend(model_path)
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


def parse_reference_steps(extra_info: dict) -> list:
    """Read reference steps from ``extra_info``.

    Key priority: ``reference_steps`` (canonical) > ``actions`` (Agentic-RL alias).
    Each step may be a string (single required action) or a list of strings
    (candidate actions; any one matching credits the step).
    """
    source = extra_info.get("reference_steps")
    if source is None:
        source = extra_info.get("actions", [])
    if isinstance(source, str):
        return [s.strip() for s in re.split(r"[，,；;\n]", source) if s.strip()]
    if isinstance(source, list):
        normalized = []
        for item in source:
            if isinstance(item, list):
                cands = [str(c).strip() for c in item if str(c).strip()]
                if cands:
                    normalized.append(cands)
            else:
                s = str(item).strip()
                if s:
                    normalized.append(s)
        return normalized
    return []


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


def compute_score(solution_str, ground_truth=None, extra_info=None, return_details=False):
    """veRL reward_score-compatible compute_score implementation.

    This function is meant to be imported by a thin file under
    ``verl/utils/reward_score``. It does not replace veRL's own
    ``utils/reward_score/__init__.py`` dispatcher.
    """
    extra_info = extra_info or {}
    think_text, answer_text = _extract_response(solution_str)
    steps = parse_reference_steps(extra_info)

    details = {
        "score": 0.0,
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
        "debug": {},
    }

    semantic = None
    scorer = _get_scorer()
    if scorer is not None and steps and answer_text:
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
        report = assess_confidence(semantic, steps, response=answer_text)
    elif not steps or not answer_text:
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

