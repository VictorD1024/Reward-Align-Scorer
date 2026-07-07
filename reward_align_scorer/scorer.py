from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .embedding import EmbeddingBackend, TextEmbedder
from .trace_detector import TraceDetector


@dataclass(frozen=True)
class WindowConfig:
    coarse_window: int = 128
    coarse_stride: int = 64
    fine_window: int = 48
    fine_stride: int = 24
    max_windows: int = 256


@dataclass(frozen=True)
class Alignment:
    path: list[tuple[int, int]]
    match_rate: float
    order_rate: float
    score: float


def sliding_windows(text: str, window: int, stride: int, max_windows: int = 256) -> list[tuple[str, int, int]]:
    """Split long responses into overlapping semantic windows."""
    text = (text or "").strip()
    if not text:
        return []
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if len(text) <= window:
        return [(text, 0, len(text))]

    windows = []
    start = 0
    while start < len(text) and len(windows) < max_windows:
        end = min(start + window, len(text))
        if end < len(text):
            cut = max(
                text.rfind("。", start, end),
                text.rfind("，", start, end),
                text.rfind("；", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if cut > start + window // 2:
                end = cut + 1
        if end <= start:
            end = min(start + window, len(text))

        windows.append((text[start:end], start, end))
        start = end - stride if end < len(text) else len(text)
        start = max(start, 0)

    return windows


def monotonic_align(sim_matrix: torch.Tensor, threshold: float = 0.65) -> Alignment:
    """Find a globally ordered step-window alignment path.

    The DP runs on CPU with numpy. The similarity matrix is typically tiny
    (num_steps x num_windows, e.g. 30x256), and doing the DP on the GPU would
    trigger a host-device sync per cell, which dominates runtime. Keeping it on
    CPU avoids that sync entirely while preserving the same recurrence.
    """
    if sim_matrix.ndim != 2:
        raise ValueError("sim_matrix must be a 2D tensor")

    num_steps, num_windows = sim_matrix.shape
    if num_steps == 0 or num_windows == 0:
        return Alignment(path=[], match_rate=0.0, order_rate=0.0, score=0.0)

    sim = sim_matrix.detach().to("cpu", dtype=torch.float32).numpy()
    gain = np.where(sim >= threshold, sim, 0.0)

    dp = np.zeros((num_steps + 1, num_windows + 1), dtype=np.float32)
    bp = np.zeros((num_steps + 1, num_windows + 1), dtype=np.int8)

    for i in range(1, num_steps + 1):
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        bp_i = bp[i]
        gain_i = gain[i - 1]
        for j in range(1, num_windows + 1):
            skip_window = dp_i[j - 1]
            skip_step = dp_prev[j]
            g = gain_i[j - 1]
            match = dp_prev[j - 1] + g
            if g > 0 and match >= skip_window and match >= skip_step:
                dp_i[j] = match
                bp_i[j] = 1
            elif skip_window >= skip_step:
                dp_i[j] = skip_window
                bp_i[j] = 2
            else:
                dp_i[j] = skip_step
                bp_i[j] = 3

    path: list[tuple[int, int]] = []
    i, j = num_steps, num_windows
    while i > 0 and j > 0:
        move = int(bp[i, j])
        if move == 1:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 2:
            j -= 1
        else:
            i -= 1

    path.reverse()
    match_rate = len(path) / num_steps

    matched_step_indices = sorted({idx for idx, _ in path})
    if not matched_step_indices:
        order_rate = 0.0
    elif len(matched_step_indices) == 1:
        order_rate = 1.0
    else:
        best_windows = sim[matched_step_indices].argmax(axis=1).tolist()
        ordered_pairs = sum(1 for a, b in zip(best_windows, best_windows[1:]) if a < b)
        order_rate = ordered_pairs / (len(matched_step_indices) - 1)

    return Alignment(
        path=path,
        match_rate=match_rate,
        order_rate=order_rate,
        score=float(dp[num_steps, num_windows]),
    )


@dataclass(frozen=True)
class ScorerConfig:
    threshold: float = 0.65
    max_score: float = 1.0
    max_length: int = 128
    encode_batch_size: int = 128
    text_cache_size: int = 8192
    require_trace: bool = False
    windows: WindowConfig = field(default_factory=WindowConfig)


@dataclass
class RewardScore:
    score: float
    match_rate: float
    order_rate: float
    matched_steps: list[str]
    unmatched_steps: list[str]
    alignment_path: list[tuple[int, int]]
    windows: list[tuple[str, int, int]]
    stats: dict


def _normalize_steps(steps) -> list:
    """Normalize reference steps to a list where each step is either a string
    (single required action) or a list of strings (candidate actions; any one
    matching credits the step). Empty entries are dropped."""
    normalized = []
    for step in steps:
        if isinstance(step, (list, tuple)):
            cands = [str(c).strip() for c in step if str(c).strip()]
            if cands:
                normalized.append(cands)
        else:
            s = str(step).strip()
            if s:
                normalized.append(s)
    return normalized


def _step_repr(step) -> str:
    """Render a step (str or candidate list) as a single string for diagnostics."""
    if isinstance(step, (list, tuple)):
        return " | ".join(str(c).strip() for c in step if str(c).strip())
    return str(step).strip()


class SemanticRewardScorer:
    """Embedding matrix + sliding windows + monotonic DP reward scorer."""

    def __init__(self, backend: EmbeddingBackend, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
        self.embedder = TextEmbedder(
            backend,
            batch_size=self.config.encode_batch_size,
            max_length=self.config.max_length,
            cache_size=self.config.text_cache_size,
        )
        self._trace_detector: Optional[TraceDetector] = None

    def _get_trace_detector(self) -> TraceDetector:
        if self._trace_detector is None:
            self._trace_detector = TraceDetector()
        return self._trace_detector

    def _score_once(self, reference_steps, response: str, window: int, stride: int, threshold: float):
        windows = sliding_windows(response, window, stride, max_windows=self.config.windows.max_windows)
        num_steps = len(reference_steps)
        if not reference_steps or not windows:
            return RewardScore(
                0.0, 0.0, 0.0, [], [_step_repr(s) for s in reference_steps], [], windows,
                {"num_windows": len(windows)},
            )

        # Each step is either a string or a list of candidate action strings.
        step_candidates = []
        for step in reference_steps:
            if isinstance(step, (list, tuple)):
                cands = [str(c).strip() for c in step if str(c).strip()]
            else:
                cands = [str(step).strip()] if str(step).strip() else []
            step_candidates.append(cands)

        if all(not c for c in step_candidates):
            return RewardScore(
                0.0, 0.0, 0.0, [], [_step_repr(s) for s in reference_steps], [], windows,
                {"num_windows": len(windows)},
            )

        flat = [c for cands in step_candidates for c in cands]
        flat_emb = self.embedder.encode(flat, use_cache=True)
        win_emb = self.embedder.encode([w[0] for w in windows], use_cache=True)
        if flat_emb is None or win_emb is None:
            return RewardScore(
                0.0, 0.0, 0.0, [], [_step_repr(s) for s in reference_steps], [], windows,
                {"num_windows": len(windows)},
            )

        # Per-candidate similarity, reduced to per-step by max over candidates.
        cand_sim = torch.mm(flat_emb, win_emb.T)
        offsets = []
        o = 0
        for cands in step_candidates:
            offsets.append((o, o + len(cands)))
            o += len(cands)
        sim_matrix = torch.stack(
            [cand_sim[s:e].max(dim=0).values if e > s else cand_sim.new_zeros(len(windows)) for s, e in offsets],
            dim=0,
        )

        alignment = monotonic_align(sim_matrix, threshold=threshold)
        matched_idx = {idx for idx, _ in alignment.path}
        path_by_step = {i: j for i, j in alignment.path}

        matched_sims: list[float] = []
        matched_margins: list[float] = []
        for step_idx, win_idx in alignment.path:
            sim_val = float(sim_matrix[step_idx, win_idx].item())
            matched_sims.append(sim_val)
            row = sim_matrix[step_idx].clone()
            row[win_idx] = -1.0
            second_best = float(row.max().item()) if len(windows) > 1 else 0.0
            matched_margins.append(max(sim_val - second_best, 0.0))

        matched_steps = []
        for idx in sorted(matched_idx):
            s, e = offsets[idx]
            win_j = path_by_step[idx]
            cand_idx = int(cand_sim[s:e, win_j].argmax().item())
            matched_steps.append(flat[s + cand_idx])

        unmatched_steps = [
            _step_repr(step) for idx, step in enumerate(reference_steps) if idx not in matched_idx
        ]

        return RewardScore(
            score=alignment.match_rate * alignment.order_rate * self.config.max_score,
            match_rate=alignment.match_rate,
            order_rate=alignment.order_rate,
            matched_steps=matched_steps,
            unmatched_steps=unmatched_steps,
            alignment_path=alignment.path,
            windows=windows,
            stats={
                "num_steps": num_steps,
                "num_windows": len(windows),
                "window": window,
                "stride": stride,
                "threshold": threshold,
                "mean_sim": float(sim_matrix.mean().detach().cpu()),
                "max_sim": float(sim_matrix.max().detach().cpu()),
                "matched_sims": matched_sims,
                "mean_matched_sim": float(sum(matched_sims) / len(matched_sims)) if matched_sims else 0.0,
                "min_matched_sim": float(min(matched_sims)) if matched_sims else 0.0,
                "mean_margin": float(sum(matched_margins) / len(matched_margins)) if matched_margins else 0.0,
                "alignment_score": alignment.score,
                "embedding_cache": self.embedder.cache.stats(),
            },
        )

    def score(self, response: str, reference_steps, coarse_to_fine: bool = True) -> RewardScore:
        reference_steps = _normalize_steps(reference_steps)

        # P0 response-level trace gate: a response that contains NEITHER
        # tool/execution evidence NOR reasoning/analytical evidence anywhere is
        # treated as recitation and scored 0. This is applied to the whole
        # response rather than per matched window because (a) preparation steps
        # like "read the issue" have no execution evidence by nature, and (b) a
        # step's evidence is frequently in a different window than its best
        # semantic match — both caused systematic false negatives in the earlier
        # per-step variant (65% of genuine PR-fix samples lost a step). Threat
        # model: lazy/padded recitation. Partial faking is out of scope.
        if self.config.require_trace:
            detector = self._get_trace_detector()
            flat_steps = []
            for s in reference_steps:
                if isinstance(s, (list, tuple)):
                    flat_steps += [str(c).strip() for c in s if str(c).strip()]
                else:
                    flat_steps.append(str(s).strip())
            if not detector.has_trace(response or "", flat_steps):
                return RewardScore(
                    score=0.0,
                    match_rate=0.0,
                    order_rate=0.0,
                    matched_steps=[],
                    unmatched_steps=[_step_repr(s) for s in reference_steps],
                    alignment_path=[],
                    windows=[],
                    stats={"trace_gate": "denied", "num_steps": len(reference_steps)},
                )

        if not coarse_to_fine:
            return self._score_once(
                reference_steps,
                response,
                self.config.windows.fine_window,
                self.config.windows.fine_stride,
                self.config.threshold,
            )

        coarse = self._score_once(
            reference_steps,
            response,
            self.config.windows.coarse_window,
            self.config.windows.coarse_stride,
            self.config.threshold * 0.9,
        )
        if coarse.match_rate >= 1.0:
            coarse.stats["method"] = "coarse"
            return coarse

        fine = self._score_once(
            reference_steps,
            response,
            self.config.windows.fine_window,
            self.config.windows.fine_stride,
            self.config.threshold,
        )
        result = fine if fine.match_rate >= coarse.match_rate else coarse
        result.stats["method"] = "coarse_to_fine"
        result.stats["coarse_match_rate"] = coarse.match_rate
        result.stats["fine_match_rate"] = fine.match_rate
        return result
