from reward_align_scorer import ScorerConfig, SemanticRewardScorer
from reward_align_scorer.embedding import load_embedding_backend


def main():
    backend = load_embedding_backend("/path/to/bge-small-zh-v1.5")
    if backend is None:
        raise RuntimeError("Set a real local embedding model path before running this example.")

    scorer = SemanticRewardScorer(backend, ScorerConfig(threshold=0.65))
    result = scorer.score(
        response=(
            "I read the issue, inspected the relevant files, patched the implementation, "
            "ran tests, and summarized the fix."
        ),
        reference_steps=[
            "read the issue",
            "inspect relevant files",
            "modify the implementation",
            "run tests",
            "summarize the fix",
        ],
    )

    print("score:", result.score)
    print("match_rate:", result.match_rate)
    print("order_rate:", result.order_rate)
    print("matched:", result.matched_steps)
    print("unmatched:", result.unmatched_steps)


if __name__ == "__main__":
    main()
