"""Confidence assessment and Judge-fallback routing for semantic reward scores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .scorer import RewardScore, format_step, normalize_steps
from .trace_detector import TraceDetector


# Heuristic step tiers for documentation and optional down-weighting hints.
_PREP_PATTERNS = (
    re.compile(r"^read\b", re.I),
    re.compile(r"^understand\b", re.I),
    re.compile(r"^review the (?:task|issue|bug report)\b", re.I),
)
_EXEC_PATTERNS = (
    re.compile(r"run test", re.I),
    re.compile(r"pytest", re.I),
    re.compile(r"apply (?:the )?(?:fix|patch|code fix)", re.I),
    re.compile(r"modify (?:the )?(?:code|implementation)", re.I),
    re.compile(r"add (?:or update )?(?:unit )?tests", re.I),
    re.compile(r"<\w+>", re.I),
    re.compile(r"verify the fix", re.I),
)
_REASONING_PATTERNS = (
    re.compile(r"locate (?:the )?root cause", re.I),
    re.compile(r"trace (?:the )?(?:call chain|failure)", re.I),
    re.compile(r"explain why", re.I),
    re.compile(r"identify (?:the )?(?:bug|issue|faulty)", re.I),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])|\n+")
_INTENTION_RE = re.compile(
    r"\b(?:i|we)?\s*(?:will|would|plan(?:ning)?\s+to|intend(?:ing)?\s+to|"
    r"(?:am|are)\s+going\s+to|need\s+to)\b|"
    r"(?:我|我们)?(?:将|会|计划|打算|准备|需要)(?:先|再|去|要)?",
    re.I,
)


def classify_step(step: str) -> str:
    """Classify a reference step by observability in the response text.

    Returns one of ``observable``, ``proxy``, ``reasoning``.
    Used for step-design review — not a hard gate.
    """
    text = (step or "").strip()
    if not text:
        return "proxy"
    if any(p.search(text) for p in _EXEC_PATTERNS):
        return "observable"
    if any(p.search(text) for p in _REASONING_PATTERNS):
        return "reasoning"
    if any(p.search(text) for p in _PREP_PATTERNS):
        return "proxy"
    return "reasoning"


def classify_steps(reference_steps) -> list[dict]:
    """Return per-step tier labels for a reference_steps list."""
    out = []
    for step in reference_steps:
        if isinstance(step, (list, tuple)):
            tiers = {classify_step(str(c)) for c in step if str(c).strip()}
            if "observable" in tiers:
                tier = "observable"
            elif tiers == {"proxy"}:
                tier = "proxy"
            else:
                tier = "reasoning"
            out.append({"step": format_step(step), "tier": tier, "candidates": list(step)})
        else:
            out.append({"step": str(step).strip(), "tier": classify_step(str(step))})
    return out


def _intention_recitation_fraction(response: str, reference_steps) -> float:
    """Fraction of steps mentioned only in intention/planning sentences.

    This is deliberately narrow: it requires an exact candidate phrase and an
    explicit future/intention marker in the same sentence. Paraphrased genuine
    evidence is therefore not penalized merely for lacking a known pattern.
    """
    normalized = normalize_steps(reference_steps)
    if not response or not normalized:
        return 0.0
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(response) if sentence.strip()]
    recited = 0
    for step in normalized:
        candidates = step if isinstance(step, list) else [step]
        intention_mention = False
        non_intention_mention = False
        for sentence in sentences:
            if not any(
                candidate and re.search(re.escape(candidate), sentence, flags=re.IGNORECASE)
                for candidate in candidates
            ):
                continue
            if _INTENTION_RE.search(sentence):
                intention_mention = True
            else:
                non_intention_mention = True
        if intention_mention and not non_intention_mention:
            recited += 1
    return recited / len(normalized)


@dataclass(frozen=True)
class RoutingConfig:
    """Thresholds for deciding whether semantic reward is trustworthy."""

    min_match_rate: float = 0.5
    min_order_rate: float = 0.25
    min_mean_matched_sim: float = 0.58
    min_margin: float = 0.0
    max_unmatched_steps: int = 1
    min_confidence: float = 0.50
    proxy_step_fraction: float = 0.6
    max_intention_recitation_fraction: float = 0.5
    high_match_rate: float = 0.95
    require_trace_when_high_match: bool = True


@dataclass
class ConfidenceReport:
    confidence: float
    fallback_recommended: bool
    reasons: list[str] = field(default_factory=list)
    step_tiers: list[dict] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def assess_confidence(
    result: RewardScore,
    reference_steps,
    config: Optional[RoutingConfig] = None,
    response: Optional[str] = None,
) -> ConfidenceReport:
    """Estimate how trustworthy a semantic reward score is."""
    config = config or RoutingConfig()
    stats = result.stats or {}
    reasons: list[str] = []
    step_tiers = classify_steps(reference_steps)

    if stats.get("trace_gate") == "denied":
        return ConfidenceReport(
            confidence=0.0,
            fallback_recommended=True,
            reasons=["trace_gate_denied"],
            step_tiers=step_tiers,
            signals={"trace_gate": "denied"},
        )

    mean_matched_sim = float(stats.get("mean_matched_sim", 0.0))
    mean_margin = float(stats.get("mean_margin", 0.0))
    min_matched_sim = float(stats.get("min_matched_sim", 0.0))
    num_steps = max(int(stats.get("num_steps", len(reference_steps) or 1)), 1)
    proxy_frac = sum(1 for s in step_tiers if s["tier"] == "proxy") / num_steps
    intention_recitation_fraction = _intention_recitation_fraction(response or "", reference_steps)

    has_trace = None
    if response and config.require_trace_when_high_match:
        flat = []
        for s in normalize_steps(reference_steps):
            flat += s if isinstance(s, list) else [s]
        has_trace = TraceDetector().has_trace(response, flat)

    signals = {
        "match_rate": result.match_rate,
        "order_rate": result.order_rate,
        "mean_matched_sim": mean_matched_sim,
        "min_matched_sim": min_matched_sim,
        "mean_margin": mean_margin,
        "unmatched_steps": len(result.unmatched_steps),
        "proxy_step_fraction": proxy_frac,
        "intention_recitation_fraction": intention_recitation_fraction,
        "has_execution_trace": has_trace,
    }

    if result.match_rate < config.min_match_rate:
        reasons.append(f"low_match_rate ({result.match_rate:.2f} < {config.min_match_rate})")
    if result.order_rate < config.min_order_rate:
        reasons.append(f"low_order_rate ({result.order_rate:.2f} < {config.min_order_rate})")
    if len(result.unmatched_steps) > config.max_unmatched_steps:
        reasons.append(
            f"too_many_unmatched ({len(result.unmatched_steps)} > {config.max_unmatched_steps})"
        )
    if mean_matched_sim and mean_matched_sim < config.min_mean_matched_sim:
        reasons.append(f"weak_mean_sim ({mean_matched_sim:.2f} < {config.min_mean_matched_sim})")
    if config.min_margin > 0 and mean_margin and mean_margin < config.min_margin:
        reasons.append(f"low_margin ({mean_margin:.3f} < {config.min_margin})")
    if proxy_frac >= config.proxy_step_fraction:
        reasons.append(f"high_proxy_step_fraction ({proxy_frac:.2f})")
    if intention_recitation_fraction >= config.max_intention_recitation_fraction:
        reasons.append(
            "intention_step_recitation "
            f"({intention_recitation_fraction:.2f} >= "
            f"{config.max_intention_recitation_fraction:.2f})"
        )
    if (
        config.require_trace_when_high_match
        and response
        and result.match_rate >= config.high_match_rate
        and has_trace is False
    ):
        reasons.append("high_match_without_execution_trace")

    trace_bonus = 0.08 if has_trace else (-0.15 if has_trace is False and result.match_rate >= 0.8 else 0.0)
    confidence = _clamp01(
        0.35 * result.match_rate
        + 0.20 * result.order_rate
        + 0.25 * _clamp01(mean_matched_sim)
        + 0.10 * _clamp01(mean_margin / 0.05)
        + 0.10 * (1.0 - proxy_frac)
        + trace_bonus
        - 0.25 * intention_recitation_fraction
    )
    if confidence < config.min_confidence:
        reasons.append(f"low_confidence ({confidence:.2f} < {config.min_confidence})")

    fallback = len(reasons) > 0
    return ConfidenceReport(
        confidence=confidence,
        fallback_recommended=fallback,
        reasons=reasons,
        step_tiers=step_tiers,
        signals=signals,
    )


def should_fallback_to_judge(
    result: RewardScore,
    reference_steps,
    config: Optional[RoutingConfig] = None,
    response: Optional[str] = None,
) -> bool:
    """Return True when LLM-as-Judge should arbitrate instead of semantic reward."""
    return assess_confidence(result, reference_steps, config, response=response).fallback_recommended


def assess_from_details(details: dict, reference_steps, config: Optional[RoutingConfig] = None, response: Optional[str] = None) -> ConfidenceReport:
    """Build a confidence report from ``compute_score(..., return_details=True)`` output."""
    pseudo = RewardScore(
        score=float(details.get("semantic_score", details.get("score", 0.0))),
        match_rate=float(details.get("step_match_rate", 0.0)),
        order_rate=float(details.get("step_order_rate", 0.0)),
        matched_steps=list(details.get("matched_steps", [])),
        unmatched_steps=list(details.get("unmatched_steps", [])),
        alignment_path=[],
        windows=[],
        stats=dict(details.get("debug", {})),
    )
    report = assess_confidence(pseudo, reference_steps, config, response=response)
    if details.get("trace_gate") == "denied":
        report.fallback_recommended = True
        if "trace_gate_denied" not in report.reasons:
            report.reasons.insert(0, "trace_gate_denied")
        report.confidence = 0.0
    return report
