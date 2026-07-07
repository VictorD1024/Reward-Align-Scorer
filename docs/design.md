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

## Recitation Defense (P0, optional)

A model can game pure semantic alignment by *reciting* the reference steps with filler ("first I will locate root cause, next I carefully apply the fix…") — no real execution, but every step name lands in a high-similarity window. The optional trace gate, enabled via `ScorerConfig(require_trace=True)` or the `REWARD_ALIGN_REQUIRE_TRACE=1` env var in the veRL adapter, scores such responses 0.

### Design: response-level gate

The gate checks the **whole response** (not each matched window) for execution evidence via `TraceDetector.has_trace`:

- **Tool/execution trace**: tool tags (`<edit_file>`), git diff markers (`diff --git`, `@@`, `--- a/`, `+++ b/`), code blocks, function defs/calls, repo file paths, test verdicts (`PASSED`/`FAILED`/`passed`/`fail*`).
- **Reasoning/analytical trace**: causal connectors (`because`, `due to`, `therefore`, `leads to`, `results in/from`) and verb-anchored noun phrases (`root cause is`, `the issue lies`). Noun-phrase signals are verb-anchored so a step *name* like `locate root cause` cannot self-trigger the gate.
- Before the reasoning check, every candidate step string is stripped from the response so a step name fragment cannot masquerade as reasoning.

A response with **neither** tool nor reasoning evidence anywhere is treated as recitation and scored 0; otherwise it scores normally with no per-step masking.

### Why response-level, not per-step

An earlier per-step variant masked each step-window pair whose window lacked trace. Calibrated on 100 real merged bug-fix PRs (numpy, scipy, pandas, scikit-learn, matplotlib, sympy, requests, flask), it produced **65/100 samples losing at least one step and a ~20% mean-score loss on genuine data**, for two structural reasons:

1. **Preparation steps have no execution evidence by nature** — `read the issue` is just a description of the bug; it has no tool call and usually no causal connector. The gate has no way to validate it, so it denied it 47×/100.
2. **The best-similarity window is often not the evidence window** — DP matches `apply the fix` to the PR-body sentence "I fixed it by…", while the actual code hunk lives in a different window. The gate checked the matched window and denied the step 31×/100.

The response-level variant eliminates both: 100/100 genuine PRs are preserved (mean score unchanged, zero extra zeroings beyond the ungated baseline), while padded recitation is still fully killed (demo: 1.0 → 0.0). The `denied` column and `[recall watch]` line in `benchmarks/dump_scores.py --require-trace` quantify the false-negative rate on any new dataset.

### Threat model and scope

The gate targets **lazy / padded recitation** — restating step names without doing the work. It does **not** detect *partial faking* (one genuine step providing trace that lets three recited steps pass); that is a harder, explicitly out-of-scope threat. The gate is a precision stopgap: turn it on when recitation is observed in rollouts, calibrate the patterns on a slice of real data, and watch the `denied` recall metric.

## Reference Step Design Guidelines

Semantic alignment does **not** verify internal actions — it checks whether the response text *covers* the semantic content of each step. Step wording therefore dominates reward quality.

### Tiers

| Tier | Meaning | Example | Scorer reliability |
| --- | --- | --- | --- |
| **observable** | Execution artifact should appear in the response | `run pytest on affected tests`, `patch src/module.py` | High — tool tags, diffs, verdicts |
| **reasoning** | Analytical content should appear | `explain why the overflow occurs`, `locate the root cause` | Medium — needs causal / diagnostic prose |
| **proxy** | Internal preparation; only matches via topic overlap | `read the linked issue`, `understand the bug report` | Low — matches bug-description paragraphs, not the act of reading |

Use `classify_step()` / `classify_steps()` from `reward_align_scorer.confidence` to audit a reference checklist before training.

### Rules of thumb

1. **Phrase steps as observable outputs**, not hidden actions: prefer `summarize the reported bug symptoms` over `read the issue`.
2. **Use candidate actions** when multiple phrasings or tools are valid: `["open web source", "open knowledge base"]`.
3. **Down-weight or omit pure proxy steps** in reward shaping; do not let them dominate the checklist.
4. **Keep execution steps** (edit, test, verify) — they anchor the signal and resist recitation.
5. **Calibrate `threshold` on task hard negatives**, not on generic paraphrase sets alone.

### Anti-patterns

| Bad step | Why | Better |
| --- | --- | --- |
| `read the issue` | Matches any bug-summary paragraph | `summarize the reported failure mode` |
| `understand the problem` | Not observable | `identify the failing function from the stack trace` |
| Seven proxy steps + one test step | Reward dominated by non-verifiable coverage | Balance observable / reasoning / proxy |

## Confidence Routing (Judge Fallback)

The library ships a built-in **confidence router** so semantic reward is not blindly trusted.

`assess_confidence(result, reference_steps)` returns a `ConfidenceReport` with:

- `confidence` — composite score in `[0, 1]`
- `fallback_recommended` — whether to defer to LLM-as-Judge
- `reasons` — human-readable triggers (`low_match_rate`, `low_margin`, `high_proxy_step_fraction`, …)
- `step_tiers` — per-step observability labels
- `signals` — raw metrics (`mean_matched_sim`, `mean_margin`, …)

`compute_score(..., return_details=True)` in the veRL adapter always includes `confidence`, `fallback_recommended`, and `fallback_reasons`.

Recommended production loop:

```text
details = compute_score(response, extra_info=..., return_details=True)
if details["fallback_recommended"]:
    reward = llm_judge(response, rubric)
else:
    reward = details["score"]
```

Tune `RoutingConfig` on your rollout slice. Run `benchmarks/reward_quality.py` to inspect fallback rates on genuine vs recitation samples.

Signals exported from `_score_once` stats:

- `matched_sims` — cosine at each aligned (step, window) pair
- `mean_matched_sim` / `min_matched_sim`
- `mean_margin` — average gap between best and second-best window per matched step

Low margin means the match is ambiguous (several windows tie) — a strong fallback trigger when `RoutingConfig.min_margin > 0`. On real PR-fix data, mean margin is often **~0.008**; the default keeps `min_margin=0` as a hard gate and uses margin only in the composite confidence score.

When `require_trace_when_high_match=True` (default), a response with `match_rate ≥ 0.95` but **no execution/reasoning trace** triggers `high_match_without_execution_trace`. This separates padded recitation (high match, no diff/code) from genuine PRs (high match, has trace) without enabling the hard trace score gate.

## Complexity

- Encoding: one batched encoder forward over `num_steps + num_windows` texts (amortized to ~0 by the LRU cache on repeated inputs).
- Similarity: one GEMM, `O(num_steps × num_windows × dim)`.
- DP: `O(num_steps × num_windows)`, on CPU, typically <1ms after window capping.
