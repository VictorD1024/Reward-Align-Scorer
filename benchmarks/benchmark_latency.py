import argparse
import time

import torch

from reward_align_scorer import ScorerConfig, SemanticRewardScorer
from reward_align_scorer.embedding import load_embedding_backend
from reward_align_scorer.scorer import monotonic_align, sliding_windows


def _bench(fn, iters: int, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def _run_total(scorer, response, steps, iters):
    scorer.score(response, steps)  # warm + populate cache
    start = time.perf_counter()
    for _ in range(iters):
        scorer.score(response, steps)
    return (time.perf_counter() - start) / iters * 1000


def _run_breakdown(scorer, response, steps, iters, cfg):
    # Cold call: first score() includes the real encoder forward (empty cache).
    cold_start = time.perf_counter()
    scorer.score(response, steps)
    cold_ms = (time.perf_counter() - cold_start) * 1000

    # Warm total: cache populated.
    warm_ms = _bench(lambda: scorer.score(response, steps), iters=iters)

    # Stage breakdown on the warm path (mirror the coarse pass of _score_once).
    windows_fn = lambda: sliding_windows(
        response,
        cfg.windows.coarse_window,
        cfg.windows.coarse_stride,
        max_windows=cfg.windows.max_windows,
    )
    window_ms = _bench(windows_fn, iters=iters)
    windows = windows_fn()
    num_windows = len(windows)

    encode_ms = _bench(
        lambda: scorer.embedder.encode([w[0] for w in windows], use_cache=True),
        iters=iters,
    )
    step_emb = scorer.embedder.encode(steps, use_cache=True)
    win_emb = scorer.embedder.encode([w[0] for w in windows], use_cache=True)

    def _gemm():
        return torch.mm(step_emb, win_emb.T)

    gemm_ms = _bench(_gemm, iters=max(iters, 100))
    sim = _gemm()

    def _dp():
        return monotonic_align(sim, threshold=cfg.threshold)

    dp_ms = _bench(_dp, iters=max(iters, 100))

    cache_stats = scorer.embedder.cache.stats()

    print(f"steps={len(steps)} windows={num_windows} iters={iters}")
    print(f"{'stage':<22} {'ms/call':>10}")
    print("-" * 34)
    print(f"{'total (cold, 1st call)':<22} {cold_ms:>10.3f}")
    print(f"{'total (warm, cached)':<22} {warm_ms:>10.3f}")
    print(f"{'  sliding windows':<22} {window_ms:>10.3f}")
    print(f"{'  encode (warm cache)':<22} {encode_ms:>10.3f}")
    print(f"{'  GEMM (sim matrix)':<22} {gemm_ms:>10.3f}")
    print(f"{'  monotonic DP':<22} {dp_ms:>10.3f}")
    print("-" * 34)
    print(f"cache stats: {cache_stats}")
    print(
        "note: 'total (cold)' includes the real encoder forward; "
        "'encode (warm cache)' is near-zero when inputs repeat."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=128)
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="print per-stage timings (windows / encode / GEMM / DP) instead of a single total",
    )
    args = parser.parse_args()

    backend = load_embedding_backend(args.model_path)
    if backend is None:
        raise RuntimeError(f"cannot load embedding model: {args.model_path}")

    cfg = ScorerConfig()
    scorer = SemanticRewardScorer(backend, cfg)
    response = "read the issue, inspect files, edit code, run tests, summarize. " * args.repeat
    steps = ["read the issue", "inspect files", "edit code", "run tests", "summarize"]

    if args.breakdown:
        _run_breakdown(scorer, response, steps, args.iters, cfg)
    else:
        avg_ms = _run_total(scorer, response, steps, args.iters)
        print({"iters": args.iters, "avg_ms": avg_ms})


if __name__ == "__main__":
    main()
