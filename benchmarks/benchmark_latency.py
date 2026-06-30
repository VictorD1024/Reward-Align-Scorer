import argparse
import time

from reward_align_scorer import ScorerConfig, SemanticRewardScorer
from reward_align_scorer.embedding import load_embedding_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=128)
    args = parser.parse_args()

    backend = load_embedding_backend(args.model_path)
    if backend is None:
        raise RuntimeError(f"cannot load embedding model: {args.model_path}")

    scorer = SemanticRewardScorer(backend, ScorerConfig())
    response = "read the issue, inspect files, edit code, run tests, summarize. " * args.repeat
    steps = ["read the issue", "inspect files", "edit code", "run tests", "summarize"]

    scorer.score(response, steps)
    start = time.perf_counter()
    for _ in range(args.iters):
        scorer.score(response, steps)
    elapsed = time.perf_counter() - start
    print({"iters": args.iters, "total_sec": elapsed, "avg_ms": elapsed / args.iters * 1000})


if __name__ == "__main__":
    main()

