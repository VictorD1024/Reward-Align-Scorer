# Resume Project

Open-source Reward Align Scorer: designed a semantic reward scoring plugin for long-response RL post-training. The system converts slow LLM Judge or rule-based step evaluation into cached, batched, matrix-friendly semantic alignment. It encodes reference steps/checklists/trajectories and overlapping response windows into normalized embeddings, computes dense similarity with GPU/Ascend matrix multiplication, and applies monotonic alignment DP to enforce ordered matching.

The plugin targets `max_response_length=4096/8192` rollout settings where reward scoring can become the training bottleneck. It provides a veRL-compatible `compute_score` entrypoint, LRU embedding cache, sliding-window alignment, matched/unmatched step diagnostics, and benchmark scripts, enabling high-confidence samples to be scored without invoking an expensive LLM Judge.

