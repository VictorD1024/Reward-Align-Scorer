from dataclasses import dataclass, field
from typing import Optional

import torch

from .embedding import EmbeddingBackend, TextEmbedder


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
    """Find a globally ordered step-window alignment path."""
    if sim_matrix.ndim != 2:
        raise ValueError("sim_matrix must be a 2D tensor")

    num_steps, num_windows = sim_matrix.shape
    if num_steps == 0 or num_windows == 0:
        return Alignment(path=[], match_rate=0.0, order_rate=0.0, score=0.0)

    device = sim_matrix.device
    dp = torch.zeros((num_steps + 1, num_windows + 1), device=device)
    bp = torch.zeros((num_steps + 1, num_windows + 1), dtype=torch.int8, device=device)

    for i in range(1, num_steps + 1):
        for j in range(1, num_windows + 1):
            skip_window = dp[i, j - 1]
            skip_step = dp[i - 1, j]
            sim = sim_matrix[i - 1, j - 1]
            match_gain = torch.where(sim >= threshold, sim, sim.new_tensor(0.0))
            match = dp[i - 1, j - 1] + match_gain

            if match >= skip_window and match >= skip_step and match_gain > 0:
                dp[i, j] = match
                bp[i, j] = 1
            elif skip_window >= skip_step:
                dp[i, j] = skip_window
                bp[i, j] = 2
            else:
                dp[i, j] = skip_step
                bp[i, j] = 3

    path: list[tuple[int, int]] = []
    i, j = num_steps, num_windows
    while i > 0 and j > 0:
        move = int(bp[i, j].item())
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
    return Alignment(
        path=path,
        match_rate=match_rate,
        order_rate=match_rate,
        score=float(dp[num_steps, num_windows].detach().cpu()),
    )


@dataclass(frozen=True)
class ScorerConfig:
    threshold: float = 0.65
    max_score: float = 1.0
    max_length: int = 128
    encode_batch_size: int = 128
    text_cache_size: int = 8192
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

    def _score_once(self, reference_steps: list[str], response: str, window: int, stride: int, threshold: float):
        windows = sliding_windows(response, window, stride, max_windows=self.config.windows.max_windows)
        if not reference_steps or not windows:
            return RewardScore(0.0, 0.0, 0.0, [], reference_steps, [], windows, {"num_windows": len(windows)})

        step_emb = self.embedder.encode(reference_steps, use_cache=True)
        win_emb = self.embedder.encode([w[0] for w in windows], use_cache=True)
        if step_emb is None or win_emb is None:
            return RewardScore(0.0, 0.0, 0.0, [], reference_steps, [], windows, {"num_windows": len(windows)})

        sim_matrix = torch.mm(step_emb, win_emb.T)
        alignment = monotonic_align(sim_matrix, threshold=threshold)
        matched_idx = {idx for idx, _ in alignment.path}

        return RewardScore(
            score=alignment.order_rate * self.config.max_score,
            match_rate=alignment.match_rate,
            order_rate=alignment.order_rate,
            matched_steps=[step for idx, step in enumerate(reference_steps) if idx in matched_idx],
            unmatched_steps=[step for idx, step in enumerate(reference_steps) if idx not in matched_idx],
            alignment_path=alignment.path,
            windows=windows,
            stats={
                "num_steps": len(reference_steps),
                "num_windows": len(windows),
                "window": window,
                "stride": stride,
                "mean_sim": float(sim_matrix.mean().detach().cpu()),
                "max_sim": float(sim_matrix.max().detach().cpu()),
                "alignment_score": alignment.score,
                "embedding_cache": self.embedder.cache.stats(),
            },
        )

    def score(self, response: str, reference_steps: list[str], coarse_to_fine: bool = True) -> RewardScore:
        reference_steps = [str(step).strip() for step in reference_steps if str(step).strip()]
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
