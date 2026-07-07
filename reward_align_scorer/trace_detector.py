from __future__ import annotations

import re
from dataclasses import dataclass, field


# Demo-calibrated defaults. These were derived from `benchmarks/demo_samples.jsonl`
# and are a STARTING TEMPLATE ONLY. Recalibrate both lists on your real rollout
# data — your model's tool-call format and reasoning style may differ.
DEFAULT_TOOL_PATTERNS: list[str] = [
    r"<\w+>",              # tool tags like <run_test>, <edit_file>
    r"^diff --git", r"^@@ ", r"^--- a/", r"^\+\+\+ b/",  # git diff hunk markers
    r"```",                # fenced code blocks
    r"\bdef \w+",          # function definitions
    r"\w+\([^)]*\)",       # function calls like config.get(...)
    r"\b(?:src|tests?|test)/\S+",  # repo file paths
    r"\b\w+\.py\b",        # bare python module names
    r"Output:",
    r"\bPASSED\b", r"\bFAILED\b", r"\bpassed\b", r"\bfail(?:ed|ure)?\b",
]

DEFAULT_REASONING_PATTERNS: list[str] = [
    # Noun-phrase signals are verb-anchored so a step name that contains the
    # noun (e.g. step "locate root cause") does not self-match when a sliding
    # window splits the name and leaves "root cause" standing alone. Recitation
    # says "locate root cause carefully"; genuine reasoning says "the root
    # cause is a hardcoded value".
    r"\broot cause\s+(?:is|was|are|were)\b",
    r"\bthe (?:issue|problem|bug)\s+(?:is|was|lies)\b",
    # Connectors are rare in step names, kept as-is.
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bcaused by\b",
    r"\btherefore\b", r"\bthus\b", r"\bhence\b",
    r"\bleads? to\b", r"\bresults? (?:in|from)\b",
]


@dataclass
class TraceDetector:
    """UNION trace gate: a window passes if it contains EITHER a tool/execution
    trace OR a reasoning/analytical trace.

    Threat model: kill *padded recitation* — responses that restate each step
    name with filler verbs ("I will ...", "I plan to ...") but perform no real
    work. Such windows contain neither tool artifacts nor causal analysis.

    Step-name stripping: before checking *reasoning* patterns, the matched
    step's own text (and all its candidates) is removed from the window, so a
    recited step name like "locate root cause" does not self-match the
    "root cause" reasoning keyword. Tool patterns are step-independent and
    checked on the raw window.
    """

    tool_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_TOOL_PATTERNS))
    reasoning_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_REASONING_PATTERNS))

    def __post_init__(self) -> None:
        # MULTILINE so "^diff --git" etc. match at any line start, not just the
        # first line of the response/window.
        self._tool_res = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.tool_patterns]
        self._reasoning_res = [re.compile(p, re.IGNORECASE) for p in self.reasoning_patterns]

    def has_trace(self, window_text: str, strip_texts) -> bool:
        """Return True if ``window_text`` shows tool OR reasoning evidence.

        ``strip_texts`` is the set of step/candidate strings to remove from the
        window before the reasoning check — pass *all* steps' candidates, not
        just the current step's. Sliding windows can span several step
        paragraphs, so a neighbouring step's name (e.g. "locate root cause")
        would otherwise leak its "root cause" keyword into another step's
        window and falsely trigger the reasoning pattern. Tool patterns are
        step-independent and checked on the raw window.
        """
        if not window_text:
            return False
        if any(p.search(window_text) for p in self._tool_res):
            return True
        residual = window_text
        if isinstance(strip_texts, str):
            strip_texts = [strip_texts]
        for s in strip_texts:
            if s:
                residual = re.sub(re.escape(s), " ", residual, flags=re.IGNORECASE)
        return any(p.search(residual) for p in self._reasoning_res)
