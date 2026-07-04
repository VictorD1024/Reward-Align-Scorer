# Design

<p align="right">
  <a href="./design.md">English</a> |
  <a href="./design.zh-CN.md">中文</a>
</p>

Reward Align Scorer is built around one engineering choice:

> Convert structured reward judgment into batched semantic alignment before calling a slow Judge.

The bottleneck in long-response RL reward is not raw FLOPs — it's one autoregressive LLM-Judge call per sample, serialized across the rollout batch. Every mechanism below exists to replace that call with batched tensor work.

## Pipeline

1. Parse reference structure from `reference_steps` (canonical) or `actions` (Agentic-RL alias) — checklist items, trajectory nodes, or subtasks.
2. Split the long response into overlapping, boundary-aware sliding windows.
3. Encode reference items and response windows into normalized embeddings (batched, LRU-cached).
4. Compute a dense similarity matrix with a single GEMM on GPU / Ascend NPU.
5. Run a threshold-gated monotonic alignment DP on CPU (numpy).
6. Return `score = match_rate × order_rate × max_score` plus interpretable alignment details.

## Why Sliding Windows

Sentence boundaries are brittle for long model responses. A required semantic unit may cross punctuation boundaries, and a 4096-token response may contain long explanations around short key actions. Sliding windows:

- preserve local context while bounding the number of comparison units (`max_windows`);
- are cut on real punctuation (`。；，\n<space>`), not hard char counts, so semantic units stay intact;
- run in a coarse→fine two-pass scheme with early-exit: a coarse pass (large window, relaxed threshold) catches full-coverage samples cheaply; the fine pass only runs when coarse is partial.

Windowing is the prerequisite that makes the rest of the pipeline batchable: it turns a variable-length response into a bounded set of fixed-length chunks, so a single batched encode and a single GEMM become possible.

## Why a Single GEMM

The expensive semantic comparison is one matrix multiply:

```text
sim = reference_embeddings @ window_embeddings.T
```

This replaces `num_steps × num_windows` separate Judge-style semantic judgments with one dense tensor op. Autoregressive decode (the real cost of an LLM Judge) is replaced by an encoder-only forward plus a GEMM, which is roughly two orders of magnitude cheaper per sample. An LRU text→embedding cache reuses encodings across the rollout batch and across training steps, so long-running reward workers approach zero encoding cost.

## Why Monotonic DP

Greedy matching is vulnerable to cascade errors: if the first reference item matches a later response segment, all remaining items are forced to search after that wrong position. Monotonic DP searches for a globally ordered path through the similarity matrix:

```text
dp[i,j] = max( dp[i,  j-1]                # skip window j
               dp[i-1,j  ]                # skip step i
               dp[i-1,j-1] + gain[i,j] )  # match (i,j)
gain[i,j] = sim[i,j] if sim[i,j] >= threshold else 0
```

The threshold gate makes low-similarity cells contribute 0, so a window cannot fake a match by being merely the "least bad" option.

## Why the DP Runs on CPU (numpy)

The DP matrix is small (`num_steps × num_windows`, typically a few thousand cells after window capping). On a GPU, a per-cell Python loop with scalar reads triggers a host↔device sync per cell (~50–100µs each), which dominates runtime and erases the GEMM's win. Running the DP on CPU numpy removes all host↔device sync. Measured on the actual `monotonic_align` recurrence, this is ~50× faster than the torch per-cell loop (e.g. 120ms → 2.3ms on a 50×256 matrix). The architecture therefore splits work by what each device is actually good at: GEMM on GPU/NPU, small DP on CPU.

## Why a Separate `order_rate`

The monotonic DP path is, by construction, ordered — so "fraction of steps matched via the DP" cannot detect a scrambled response. `order_rate` is computed independently of the DP path: for each matched step it takes the argmax window (the response location most likely to realize that step), then measures how many adjacent reference-step pairs have argmax windows in the expected order. This way a response that hits every step but in the wrong order still gets a low `order_rate` even when `match_rate` is high. The final score multiplies both signals: `score = match_rate × order_rate × max_score`.

## Why Per-Step Candidate Actions

A reference step in Agentic RL may be realizable by more than one action (different tools, different phrasings). A flat `list[str]` would force the user to pick one canonical description and miss semantically valid alternatives. The scorer therefore accepts `list[str | list[str]]`: a step that is a list is a set of candidate actions, and the step is credited if **any** candidate matches. Mechanically, each step's row in the similarity matrix is the element-wise **max** over its candidates' rows:

```text
sim_matrix[i, j] = max over candidates c of cos(step_i_candidate_c_emb, window_j_emb)
```

The DP, threshold gate, and `order_rate` are unchanged — they operate on the per-step max-similarity matrix. `matched_steps` reports the specific candidate that won at the matched window, so the diagnostic still tells you *which* action was realized. Unmatched multi-candidate steps are reported as the candidates joined by ` | `.

## Complexity

- Encoding: one batched encoder forward over `num_steps + num_windows` texts (amortized to ~0 by the LRU cache on repeated inputs).
- Similarity: one GEMM, `O(num_steps × num_windows × dim)`.
- DP: `O(num_steps × num_windows)`, on CPU, typically <1ms after window capping.
