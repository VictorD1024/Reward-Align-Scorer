from reward_align_scorer import ScorerConfig, SemanticRewardScorer
from reward_align_scorer.embedding import load_embedding_backend


def main():
    backend = load_embedding_backend("/path/to/bge-small-zh-v1.5")
    if backend is None:
        raise RuntimeError("Set a real local embedding model path before running this example.")

    scorer = SemanticRewardScorer(backend, ScorerConfig(threshold=0.65))
    result = scorer.score(
        response="先读取任务，再检查相关文件，修改实现后运行测试，最后总结修复结果。",
        reference_steps=["读取任务", "检查相关文件", "修改实现", "运行测试", "总结修复结果"],
    )

    print("score:", result.score)
    print("match_rate:", result.match_rate)
    print("order_rate:", result.order_rate)
    print("matched:", result.matched_steps)
    print("unmatched:", result.unmatched_steps)


if __name__ == "__main__":
    main()

