from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..backends.embedding import EmbeddingBackend, TextEmbedder
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


def _window_coverage(windows: list[tuple[str, int, int]]) -> int:
    """Return the number of unique character positions covered by windows."""
    if not windows:
        return 0
    covered = 0
    current_start, current_end = windows[0][1], windows[0][2]
    for _, start, end in windows[1:]:
        if start > current_end:
            covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return covered + current_end - current_start


def _sliding_windows_with_stats(
    text: str,
    window: int,
    stride: int,
    max_windows: int = 256,
) -> tuple[list[tuple[str, int, int]], dict]:
    """Split text and retain coverage diagnostics for capped long responses."""
    text = (text or "").strip()
    if not text:
        return [], {
            "input_chars": 0,
            "candidate_windows": 0,
            "num_windows": 0,
            "windows_downsampled": False,
            "covered_chars": 0,
            "coverage_rate": 0.0,
            "tail_covered": False,
        }
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if stride >= window:
        raise ValueError("stride must be smaller than window")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if len(text) <= window:
        windows = [(text, 0, len(text))]
        return windows, {
            "input_chars": len(text),
            "candidate_windows": 1,
            "num_windows": 1,
            "windows_downsampled": False,
            "covered_chars": len(text),
            "coverage_rate": 1.0,
            "tail_covered": True,
        }

    candidates = []
    start = 0
    while start < len(text):
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

        candidates.append((text[start:end], start, end))
        if end >= len(text):
            break
        # Boundary-aware cuts can make a window shorter than ``stride``.
        # Always advance at least one character to avoid a non-progress loop.
        start = max(end - stride, start + 1)

    if len(candidates) <= max_windows:
        windows = candidates
    elif max_windows == 1:
        # With one slot it is impossible to retain both ends. Prefer the tail:
        # earlier pipeline stages usually already preserve the response head,
        # while silent loss of final actions is the failure this cap prevents.
        windows = [candidates[-1]]
    else:
        # Evenly sample the complete candidate sequence, including index 0 and
        # the final tail window. This avoids the previous head-only truncation.
        last = len(candidates) - 1
        indices = [round(i * last / (max_windows - 1)) for i in range(max_windows)]
        windows = [candidates[i] for i in indices]

    covered_chars = _window_coverage(windows)
    stats = {
        "input_chars": len(text),
        "candidate_windows": len(candidates),
        "num_windows": len(windows),
        "windows_downsampled": len(candidates) > len(windows),
        "covered_chars": covered_chars,
        "coverage_rate": covered_chars / len(text),
        "tail_covered": bool(windows and windows[-1][2] == len(text)),
    }
    return windows, stats


def sliding_windows(text: str, window: int, stride: int, max_windows: int = 256) -> list[tuple[str, int, int]]:
    """Split long responses into overlapping, tail-preserving semantic windows."""
    windows, _ = _sliding_windows_with_stats(text, window, stride, max_windows)
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

    # Order is measured independently from the monotonic path. Include every
    # step whose best window clears the threshold, even if DP omitted that step
    # because its strongest position was out of order. Otherwise DP can hide
    # exactly the disorder this signal is intended to detect.
    best_similarities = sim.max(axis=1)
    order_step_indices = np.flatnonzero(best_similarities >= threshold).tolist()
    if not order_step_indices:
        order_rate = 0.0
    elif len(order_step_indices) == 1:
        order_rate = 1.0
    else:
        best_windows = sim[order_step_indices].argmax(axis=1).tolist()
        pair_credit = 0.0
        pair_count = 0
        for left_index, left_window in enumerate(best_windows[:-1]):
            for right_window in best_windows[left_index + 1:]:
                pair_count += 1
                if left_window < right_window:
                    pair_credit += 1.0
                elif left_window == right_window:
                    pair_credit += 0.5
        order_rate = pair_credit / pair_count

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


@dataclass
class SimilarityMatrix:
    """Reusable step-window similarities produced by one embedding pass."""

    reference_steps: list
    windows: list[tuple[str, int, int]]
    values: torch.Tensor
    stats: dict


def normalize_steps(steps) -> list:
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


def format_step(step) -> str:
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

    def similarity_matrix(
        self,
        response: str,
        reference_steps,
        *,
        window: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> SimilarityMatrix:
        """Embed references and response windows once and return their cosine matrix.

        Candidate references within one step are reduced with ``max``. This is
        the same matrix consumed by the linear monotonic aligner, exposed as a
        public primitive for graph-constrained alignment and diagnostics.
        """
        reference_steps = normalize_steps(reference_steps or [])
        window = self.config.windows.fine_window if window is None else window
        stride = self.config.windows.fine_stride if stride is None else stride
        windows, window_stats = _sliding_windows_with_stats(
            "" if response is None else str(response),
            window,
            stride,
            max_windows=self.config.windows.max_windows,
        )
        return self._similarity_matrix_from_windows(
            reference_steps,
            windows,
            {
                **window_stats,
                "window": window,
                "stride": stride,
            },
        )

    def similarity_matrix_for_passages(
        self,
        passages,
        reference_steps,
    ) -> SimilarityMatrix:
        """Return similarities while preserving caller-defined turn boundaries."""
        if isinstance(passages, (str, bytes)):
            raise TypeError("passages must be a sequence of strings")
        reference_steps = normalize_steps(reference_steps or [])
        windows = []
        offset = 0
        for passage in passages:
            text = "" if passage is None else str(passage).strip()
            if text:
                windows.append((text, offset, offset + len(text)))
            offset += len(text) + 1
        return self._similarity_matrix_from_windows(
            reference_steps,
            windows,
            {
                "input_chars": max(offset - 1, 0),
                "candidate_windows": len(windows),
                "num_windows": len(windows),
                "windows_downsampled": False,
                "covered_chars": sum(end - start for _, start, end in windows),
                "coverage_rate": 1.0 if windows else 0.0,
                "tail_covered": bool(windows),
                "window": "passage",
                "stride": "passage",
            },
        )

    def _similarity_matrix_from_windows(
        self,
        reference_steps,
        windows,
        stats,
    ) -> SimilarityMatrix:
        if not reference_steps or not windows:
            return SimilarityMatrix(
                reference_steps=reference_steps,
                windows=windows,
                values=torch.empty((len(reference_steps), len(windows))),
                stats={
                    **stats,
                    "num_steps": len(reference_steps),
                    "embedding_cache": self.embedder.cache.stats(),
                },
            )

        step_candidates = [
            (
                [str(candidate).strip() for candidate in step if str(candidate).strip()]
                if isinstance(step, (list, tuple))
                else [str(step).strip()]
            )
            for step in reference_steps
        ]
        flat = [candidate for candidates in step_candidates for candidate in candidates]
        window_texts = [item[0] for item in windows]
        encoded = self.embedder.encode(
            [*flat, *window_texts],
            use_cache=True,
            text_types=["query"] * len(flat) + ["passage"] * len(window_texts),
        )
        candidate_similarities = torch.mm(
            encoded[:len(flat)],
            encoded[len(flat):].T,
        )
        offsets = []
        offset = 0
        for candidates in step_candidates:
            offsets.append((offset, offset + len(candidates)))
            offset += len(candidates)
        values = torch.stack(
            [
                candidate_similarities[start:end].max(dim=0).values
                for start, end in offsets
            ],
            dim=0,
        )
        return SimilarityMatrix(
            reference_steps=reference_steps,
            windows=windows,
            values=values,
            stats={
                **stats,
                "num_steps": len(reference_steps),
                "mean_sim": float(values.mean().detach().cpu()),
                "max_sim": float(values.max().detach().cpu()),
                "embedding_cache": self.embedder.cache.stats(),
            },
        )

    def _empty_score(self, reference_steps, windows, stats) -> RewardScore:
        return RewardScore(
            score=0.0,
            match_rate=0.0,
            order_rate=0.0,
            matched_steps=[],
            unmatched_steps=[format_step(s) for s in reference_steps],
            alignment_path=[],
            windows=windows,
            stats=stats,
        )

    def _score_from_embeddings(
        self,
        reference_steps,
        windows,
        window_stats,
        step_candidates,
        flat,
        flat_emb,
        win_emb,
        threshold: float,
        window: int,
        stride: int,
        batch_pass_size: int,
        batch_input_texts: int,
    ) -> RewardScore:
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
            format_step(step) for idx, step in enumerate(reference_steps) if idx not in matched_idx
        ]
        best_step_sims, best_window_indices = sim_matrix.max(dim=1)

        return RewardScore(
            score=alignment.match_rate * alignment.order_rate * self.config.max_score,
            match_rate=alignment.match_rate,
            order_rate=alignment.order_rate,
            matched_steps=matched_steps,
            unmatched_steps=unmatched_steps,
            alignment_path=alignment.path,
            windows=windows,
            stats={
                **window_stats,
                "num_steps": len(reference_steps),
                "window": window,
                "stride": stride,
                "threshold": threshold,
                "batch_pass_size": batch_pass_size,
                "batch_input_texts": batch_input_texts,
                "mean_sim": float(sim_matrix.mean().detach().cpu()),
                "max_sim": float(sim_matrix.max().detach().cpu()),
                "best_step_sims": best_step_sims.detach().to("cpu").tolist(),
                "best_window_indices": best_window_indices.detach().to("cpu").tolist(),
                "matched_sims": matched_sims,
                "mean_matched_sim": float(sum(matched_sims) / len(matched_sims)) if matched_sims else 0.0,
                "min_matched_sim": float(min(matched_sims)) if matched_sims else 0.0,
                "mean_margin": float(sum(matched_margins) / len(matched_margins)) if matched_margins else 0.0,
                "alignment_score": alignment.score,
                "embedding_cache": self.embedder.cache.stats(),
            },
        )

    def _score_batch_once(
        self,
        jobs: list[tuple[list, str]],
        window: int,
        stride: int,
        threshold: float,
    ) -> list[RewardScore]:
        """Run one windowing/alignment pass for several samples.

        Every step candidate and response window in this pass is assembled into
        one ``TextEmbedder.encode`` call. The returned embeddings are then
        sliced back into per-sample similarity matrices and CPU DP jobs.
        """
        results: list[Optional[RewardScore]] = [None] * len(jobs)
        prepared = []
        all_texts: list[str] = []
        all_text_types: list[str] = []

        for job_index, (reference_steps, response) in enumerate(jobs):
            windows, window_stats = _sliding_windows_with_stats(
                response,
                window,
                stride,
                max_windows=self.config.windows.max_windows,
            )
            if not reference_steps or not windows:
                results[job_index] = self._empty_score(reference_steps, windows, window_stats)
                continue

            step_candidates = []
            for step in reference_steps:
                if isinstance(step, (list, tuple)):
                    cands = [str(c).strip() for c in step if str(c).strip()]
                else:
                    cands = [str(step).strip()] if str(step).strip() else []
                step_candidates.append(cands)

            flat = [candidate for candidates in step_candidates for candidate in candidates]
            if not flat:
                results[job_index] = self._empty_score(reference_steps, windows, window_stats)
                continue

            text_offset = len(all_texts)
            window_texts = [item[0] for item in windows]
            all_texts.extend(flat)
            all_texts.extend(window_texts)
            all_text_types.extend(["query"] * len(flat))
            all_text_types.extend(["passage"] * len(window_texts))
            prepared.append(
                {
                    "job_index": job_index,
                    "reference_steps": reference_steps,
                    "windows": windows,
                    "window_stats": window_stats,
                    "step_candidates": step_candidates,
                    "flat": flat,
                    "text_offset": text_offset,
                    "num_flat": len(flat),
                    "num_windows": len(window_texts),
                }
            )

        encoded = (
            self.embedder.encode(all_texts, use_cache=True, text_types=all_text_types)
            if all_texts
            else None
        )
        if prepared and encoded is None:
            raise RuntimeError("embedding backend returned no embeddings for a non-empty batch")

        for item in prepared:
            start = item["text_offset"]
            split = start + item["num_flat"]
            end = split + item["num_windows"]
            results[item["job_index"]] = self._score_from_embeddings(
                reference_steps=item["reference_steps"],
                windows=item["windows"],
                window_stats=item["window_stats"],
                step_candidates=item["step_candidates"],
                flat=item["flat"],
                flat_emb=encoded[start:split],
                win_emb=encoded[split:end],
                threshold=threshold,
                window=window,
                stride=stride,
                batch_pass_size=len(jobs),
                batch_input_texts=len(all_texts),
            )

        if any(result is None for result in results):
            raise RuntimeError("internal batch score assembly failed")
        return results

    def _score_once(self, reference_steps, response: str, window: int, stride: int, threshold: float):
        """Backward-compatible single-sample wrapper around the batch kernel."""
        return self._score_batch_once(
            [(reference_steps, response)],
            window=window,
            stride=stride,
            threshold=threshold,
        )[0]

    def _trace_gate_denial(self, response: str, reference_steps) -> Optional[RewardScore]:
        if not self.config.require_trace:
            return None
        # P0 response-level trace gate: a response that contains NEITHER
        # tool/execution evidence NOR reasoning/analytical evidence anywhere is
        # treated as recitation and scored 0. This is applied to the whole
        # response rather than per matched window because (a) preparation steps
        # like "read the issue" have no execution evidence by nature, and (b) a
        # step's evidence is frequently in a different window than its best
        # semantic match — both caused systematic false negatives in the earlier
        # per-step variant (65% of genuine PR-fix samples lost a step). Threat
        # model: lazy/padded recitation. Partial faking is out of scope.
        detector = self._get_trace_detector()
        flat_steps = []
        for step in reference_steps:
            if isinstance(step, (list, tuple)):
                flat_steps += [str(candidate).strip() for candidate in step if str(candidate).strip()]
            else:
                flat_steps.append(str(step).strip())
        if detector.has_trace(response or "", flat_steps):
            return None
        return RewardScore(
            score=0.0,
            match_rate=0.0,
            order_rate=0.0,
            matched_steps=[],
            unmatched_steps=[format_step(step) for step in reference_steps],
            alignment_path=[],
            windows=[],
            stats={"trace_gate": "denied", "num_steps": len(reference_steps)},
        )

    def score_batch(
        self,
        responses,
        reference_steps_batch,
        coarse_to_fine: bool = True,
    ) -> list[RewardScore]:
        """Score a rollout batch while sharing embedding forwards across samples."""
        if isinstance(responses, (str, bytes)):
            raise TypeError("responses must be a sequence of strings, not a single string")
        responses = list(responses)
        reference_steps_batch = list(reference_steps_batch)
        if len(responses) != len(reference_steps_batch):
            raise ValueError("responses and reference_steps_batch must have the same length")
        if not responses:
            return []

        normalized_jobs = [
            (
                normalize_steps(steps if steps is not None else []),
                "" if response is None else str(response),
            )
            for response, steps in zip(responses, reference_steps_batch)
        ]
        final_results: list[Optional[RewardScore]] = [None] * len(normalized_jobs)
        active_indices = []
        active_jobs = []
        for index, (reference_steps, response) in enumerate(normalized_jobs):
            denied = self._trace_gate_denial(response, reference_steps)
            if denied is not None:
                denied.stats["batch_size"] = len(normalized_jobs)
                final_results[index] = denied
            else:
                active_indices.append(index)
                active_jobs.append((reference_steps, response))

        if not coarse_to_fine:
            scored = self._score_batch_once(
                active_jobs,
                self.config.windows.fine_window,
                self.config.windows.fine_stride,
                self.config.threshold,
            )
            for index, result in zip(active_indices, scored):
                result.stats["batch_size"] = len(normalized_jobs)
                final_results[index] = result
            if any(result is None for result in final_results):
                raise RuntimeError("internal batch result assembly failed")
            return final_results

        coarse_results = self._score_batch_once(
            active_jobs,
            self.config.windows.coarse_window,
            self.config.windows.coarse_stride,
            self.config.threshold * 0.9,
        )
        fine_jobs = []
        fine_meta = []
        for active_position, (index, job, coarse) in enumerate(zip(active_indices, active_jobs, coarse_results)):
            if coarse.match_rate >= 1.0:
                coarse.stats["method"] = "coarse"
                coarse.stats["batch_size"] = len(normalized_jobs)
                final_results[index] = coarse
            else:
                fine_jobs.append(job)
                fine_meta.append((active_position, index))

        fine_results = self._score_batch_once(
            fine_jobs,
            self.config.windows.fine_window,
            self.config.windows.fine_stride,
            self.config.threshold,
        )
        for (active_position, index), fine in zip(fine_meta, fine_results):
            coarse = coarse_results[active_position]
            result = fine if fine.match_rate >= coarse.match_rate else coarse
            result.stats["method"] = "coarse_to_fine"
            result.stats["coarse_match_rate"] = coarse.match_rate
            result.stats["fine_match_rate"] = fine.match_rate
            result.stats["batch_size"] = len(normalized_jobs)
            final_results[index] = result

        if any(result is None for result in final_results):
            raise RuntimeError("internal batch result assembly failed")
        return final_results

    def score(self, response: str, reference_steps, coarse_to_fine: bool = True) -> RewardScore:
        return self.score_batch(
            responses=[response],
            reference_steps_batch=[reference_steps],
            coarse_to_fine=coarse_to_fine,
        )[0]
