import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raise_scorer.backends import load_embedding_backend  # noqa: E402
from raise_scorer.core import (  # noqa: E402
    ScorerConfig,
    SemanticRewardScorer,
    monotonic_align,
    sliding_windows,
)


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
    return _bench(
        lambda: scorer.score(response, steps),
        iters=iters,
        warm=1,
    )


def _run_batch_total(scorer, responses, steps_batch, iters):
    cold_start = time.perf_counter()
    scorer.score_batch(responses, steps_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1000
    warm_ms = _bench(
        lambda: scorer.score_batch(responses, steps_batch),
        iters=iters,
        warm=1,
    )
    batch_size = len(responses)
    return {
        "batch_size": batch_size,
        "iters": iters,
        "cold_ms_per_batch": cold_ms,
        "warm_ms_per_batch": warm_ms,
        "warm_ms_per_sample": warm_ms / batch_size,
        "warm_samples_per_second": batch_size * 1000 / warm_ms,
    }


def _run_breakdown(scorer, response, steps, iters, cfg):
    # Cold call: first score() includes the real encoder forward (empty cache).
    cold_start = time.perf_counter()
    scorer.score(response, steps)
    cold_ms = (time.perf_counter() - cold_start) * 1000

    # Warm total: cache populated.
    warm_ms = _bench(lambda: scorer.score(response, steps), iters=iters)

    # Stage breakdown on the warm path (mirror the coarse pass of _score_once).
    def windows_fn():
        return sliding_windows(
            response,
            cfg.windows.coarse_window,
            cfg.windows.coarse_stride,
            max_windows=cfg.windows.max_windows,
        )

    window_ms = _bench(windows_fn, iters=iters)
    windows = windows_fn()
    num_windows = len(windows)

    encode_ms = _bench(
        lambda: scorer.embedder.encode(
            [w[0] for w in windows],
            use_cache=True,
            text_types=["passage"] * len(windows),
        ),
        iters=iters,
    )
    step_emb = scorer.embedder.encode(steps, use_cache=True, text_types=["query"] * len(steps))
    win_emb = scorer.embedder.encode(
        [w[0] for w in windows],
        use_cache=True,
        text_types=["passage"] * len(windows),
    )

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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pooling", choices=("cls", "mean"))
    parser.add_argument("--query-instruction")
    parser.add_argument("--passage-instruction")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="print per-stage timings (windows / encode / GEMM / DP) instead of a single total",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.breakdown and args.batch_size != 1:
        parser.error("--breakdown currently requires --batch-size 1")

    backend = load_embedding_backend(
        args.model_path,
        pooling=args.pooling,
        query_instruction=args.query_instruction,
        passage_instruction=args.passage_instruction,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if backend is None:
        raise RuntimeError(f"cannot load embedding model: {args.model_path}")

    cfg = ScorerConfig()
    scorer = SemanticRewardScorer(backend, cfg)
    backend_config = {
        "model": backend.model_path,
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "pooling": backend.pooling,
        "query_instruction": backend.query_instruction,
        "passage_instruction": backend.passage_instruction,
        "revision": backend.revision,
    }
    response = "read the issue, inspect files, edit code, run tests, summarize. " * args.repeat
    steps = ["read the issue", "inspect files", "edit code", "run tests", "summarize"]

    if args.breakdown:
        print({"backend": backend_config})
        _run_breakdown(scorer, response, steps, args.iters, cfg)
    elif args.batch_size > 1:
        responses = [
            (
                f"sample {index}: read the issue, inspect files, edit code, "
                f"run tests, summarize sample {index}. "
            )
            * args.repeat
            for index in range(args.batch_size)
        ]
        print(
            {
                "backend": backend_config,
                **_run_batch_total(scorer, responses, [steps] * args.batch_size, args.iters),
            }
        )
    else:
        avg_ms = _run_total(scorer, response, steps, args.iters)
        print({"backend": backend_config, "iters": args.iters, "avg_ms": avg_ms})


if __name__ == "__main__":
    main()
