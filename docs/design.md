# Design

<p align="right">
  <a href="./design.md">English</a> |
  <a href="./design.zh-CN.md">中文</a>
</p>

RAISE (Reward-Aligned Agent Trajectory Scorer) is built around one engineering choice:

> Convert structured reward judgment into batched semantic alignment before calling a slow Judge.

The bottleneck in long-response RL reward is not raw FLOPs — it's one autoregressive LLM-Judge call per sample, serialized across the rollout batch. Every mechanism below exists to replace that call with batched tensor work.

## Pipeline

1. Resolve reference structure from veRL/RLVR inputs, then normalize checklist
   items, trajectory nodes, or subtasks.
2. Split the long response into overlapping, boundary-aware sliding windows.
3. Encode reference items and response windows into normalized embeddings (batched, LRU-cached).
4. Compute a dense similarity matrix with a single GEMM on GPU / Ascend NPU.
5. Run a threshold-gated monotonic alignment DP on CPU (numpy).
6. Return `score = match_rate × order_rate × max_score` plus interpretable alignment details.

### RLVR field resolution

The standard veRL RewardManager extracts
`row["reward_model"]["ground_truth"]` and `row.get("extra_info", {})` from a
Parquet row before invoking `compute_score`. RAISE resolves steps in this
order:

1. `extra_info.reference_steps`;
2. `extra_info.actions`;
3. configured or known nested paths such as
   `metadata.reference_steps` and `rlvr.reference_steps`;
4. structured `ground_truth.reference_steps`, `ground_truth.actions`, or a
   ground-truth list.

Arrow/Pandas list-like values, native Python containers, and JSON-serialized
arrays/objects are normalized. Plain scalar ground-truth strings are excluded
to avoid confusing final-answer targets with process supervision.
`ReferenceStepsResolution.source` and the veRL detail field
`reference_steps_source` make the selected schema path auditable.

## Embedding Contract

The loader accepts a local model directory or a Hugging Face model ID and keeps
the embedding contract alongside the model: pooling (`cls` or attention-mask
`mean`), query instruction, passage instruction, and optional revision.
Reference-step candidates are encoded as queries; response windows are encoded
as passages. The role is part of the LRU key, preventing an asymmetric model
from incorrectly reusing the same raw text across its two towers.

Known model presets encode model-card requirements without hiding overrides.
For example, the multilingual E5 family uses mean pooling plus `query: ` and
`passage: ` prefixes. An explicit option always wins over a preset, including
an explicitly empty instruction. Similarity thresholds remain a scorer-level
setting because their calibration depends on the model, language mix, window
sizes, and rollout distribution.

## Workflow GroundTruth v1

Linear reference steps remain the smallest API. Long-horizon tasks can instead
use a validated `WorkflowGroundTruth`, which represents either a DAG or a
bounded cyclic state machine.

Each node carries:

- a required or optional role, stage, weight, and visit bound;
- an `action` contract with one or more valid alternatives;
- optional `evidence` and `outcome` contracts;
- optional environment preconditions and postconditions.

Edges are labeled `forward`, `alternative`, `retry`, `recovery`, or
`rollback`, with a finite traversal bound. Validation rejects unknown,
unreachable, or non-terminating nodes. Runtime path expansion is additionally
bounded by path length, finite state count, and total state expansions.

The default graph scorer builds a finite automaton whose state is
`(current node, node-visit counters, edge-traversal counters)`. Because every
transition increments a visit counter, this unrolling is acyclic even when the
source workflow contains retry or rollback cycles. Dynamic programming then
searches the product of bounded workflow states and ordered response windows.
Only the selected legal path receives full Action/Evidence/Outcome, stage, and
state scoring. This avoids embedding and scoring every complete terminal path.

`alignment_mode="enumerate"` retains the original bounded terminal-path
enumerator as a controlled ablation. Both modes expose truncation diagnostics;
graph mode additionally reports expanded states, transitions, terminal states,
and action-coverage estimate. Required and optional contracts remain separate,
so absent optional nodes are observable without lowering required coverage.
Repeated contracts use occurrence-specific monotonic matches for order.

Node reward is the normalized weighted completion of Action, Evidence, and
Outcome. Required node rewards aggregate into stage rewards; stage rewards
aggregate with explicit `stage_weights`, or with required node weights by
default. This exposes both local credit and hierarchical progress.

State verification is deliberately separate from text similarity. Expected
pre/post state is matched as a recursive subset of supplied observations, or
can be checked by a caller-provided validator. State validity may be reported
only, blended into semantic process reward, or configured as a hard gate.
RAISE does not manufacture environment truth from response text.

### Turn-level potential shaping

`WorkflowPotentialScorer` preserves agent turns as explicit passages and
computes a prefix potential after each turn. By default,
`Phi_t = required_workflow_coverage(prefix_t)`, and the dense signal is:

```text
r_t = scale * (gamma * Phi_t - Phi_(t-1))
```

At `gamma=1`, these increments telescope to the final potential. A trainer can
place each increment on the final generated token of its agent turn with
`assign_turn_rewards`. Environment outcome reward remains a separate logged
signal and is mixed with process reward only at the experiment layer.

### TraceEvent trust boundary

`WorkflowScorer.score_events()` accepts ordered `TraceEvent` objects emitted
by runtime adapters. The event stream, rather than model prose, is rendered
into semantic scorer input. Tool inputs/outputs are serialized
deterministically and clipped per field to bound encoder and logging costs.

Trust is explicit and two-part: an event must set `trusted=True` and its source
must belong to the configured trusted sources (`runtime` or `external` by
default). Consequently, a model-authored payload cannot promote itself by
writing `trusted=true`. Serialized mappings ignore their trusted bit by
default; accepting serialized trust is a separate opt-in that belongs only
after an authenticated runtime boundary.

Validation reconstructs node occurrences, legal edges, visit counts, and edge
traversal counts. It also checks event-ID uniqueness, workflow start, workflow
terminal, and occurrence sequencing. Invalid traces hard-gate reward by
default; soft penalty and non-terminal partial-trajectory modes are explicit
configuration choices. Path truncation, event-path/semantic-path disagreement,
and validation failures are surfaced to Judge fallback in the veRL adapter.

For each node occurrence, the first non-empty `state_before` and last non-empty
`state_after` are converted into state observations. This keeps three signals
separate and auditable:

```text
semantic contract coverage
runtime transition legality
environment state validity
```

### Runtime adapter boundary

`ToolCallRecord` is the framework-neutral input contract for runtime
integration. `RuntimeEventAdapter` dispatches exact custom handlers first,
then built-in pytest, shell, file, browser, HTTP, and generic handlers. The
resulting Action/Evidence/Outcome strings are stable scorer inputs; raw tool
input/output remains attached for audit and deterministic rendering.

Trust is a capability of the adapter instance, not a property accepted from a
record. This prevents a model-controlled tool payload from self-promoting.
The default is untrusted. A rollout worker may construct
`RuntimeEventAdapter(trusted_runtime=True)` only after authenticating record
provenance. Recursive sensitive-key redaction is enabled by default before
raw payloads enter `TraceEvent`.

veRL applies the same boundary to `extra_info["tool_calls"]`. Explicit
`trace_events` take precedence; otherwise the records are adapted once and
flow through the existing trace validator and scorer.

## Why Sliding Windows

Sentence boundaries are brittle for long model responses. A required semantic unit may cross punctuation boundaries, and a 4096-token response may contain long explanations around short key actions. Sliding windows:

- preserve local context while bounding the number of comparison units (`max_windows`);
- are cut on real punctuation (`。；，\n<space>`), not hard char counts, so semantic units stay intact;
- run in a coarse→fine two-pass scheme with early-exit: a coarse pass (large window, relaxed threshold) catches full-coverage samples cheaply; the fine pass only runs when coarse is partial.

Window sizes are currently specified in characters. The generated candidates
cover the complete response. If their count exceeds `max_windows`, RAISE
samples them evenly while retaining the first and final windows, rather than
keeping only a response prefix. `_score_once` reports `candidate_windows`,
`windows_downsampled`, `covered_chars`, `coverage_rate`, and `tail_covered` so
the cap can be calibrated on rollout data.

Windowing is the prerequisite that makes the rest of the pipeline batchable: it turns variable-length responses into bounded fixed-length chunks, enabling shared batch encoding followed by dense per-sample GEMMs.

`SemanticRewardScorer.score_batch()` applies this across samples. For each
coarse/fine pass it concatenates all active samples' step candidates and
windows into one embedder call, then slices the embeddings back into
sample-specific similarity matrices. The embedder also deduplicates identical
texts within that call before checking its process-local LRU cache. Encoder
inputs may still be split into micro-batches according to
`encode_batch_size`; GEMM and CPU DP remain per sample.

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

The monotonic DP path is, by construction, ordered — so "fraction of steps
matched via the DP" cannot detect a scrambled response. `order_rate` is
computed independently using every step whose strongest window clears the
threshold, including above-threshold steps omitted by DP. It compares all
reference-step pairs: a forward pair receives `1`, a same-window tie receives
`0.5`, and a reversed pair receives `0`. This Kendall-style signal prevents DP
from hiding disorder by dropping the offending step while avoiding an
all-or-nothing penalty when two concise actions share a window. The final score
multiplies both signals: `score = match_rate × order_rate × max_score`.

## Why Per-Step Candidate Actions

A reference step in Agentic RL may be realizable by more than one action (different tools, different phrasings). A flat `list[str]` would force the user to pick one canonical description and miss semantically valid alternatives. The scorer therefore accepts `list[str | list[str]]`: a step that is a list is a set of candidate actions, and the step is credited if **any** candidate matches. Mechanically, each step's row in the similarity matrix is the element-wise **max** over its candidates' rows:

```text
sim_matrix[i, j] = max over candidates c of cos(step_i_candidate_c_emb, window_j_emb)
```

The DP, threshold gate, and `order_rate` are unchanged — they operate on the per-step max-similarity matrix. `matched_steps` reports the specific candidate that won at the matched window, so the diagnostic still tells you *which* action was realized. Unmatched multi-candidate steps are reported as the candidates joined by ` | `.

## Recitation Defense (P0, optional)

A model can game pure semantic alignment by *reciting* the reference steps with filler ("first I will locate root cause, next I carefully apply the fix…") — no real execution, but every step name lands in a high-similarity window. The optional trace gate, enabled via `ScorerConfig(require_trace=True)` or the `RAISE_REQUIRE_TRACE=1` env var in the veRL adapter, scores such responses 0.

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

Use `classify_step()` / `classify_steps()` from `raise_scorer.core` to audit a reference checklist before training.

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

Tune `RoutingConfig` on your rollout slice. Run `benchmarks/reward_quality.py`
to inspect labeled ranking, threshold, pairwise, routing, and hard-negative
subtype metrics. The command also reports an unordered exact-substring
baseline, which is expected to score recitation highly and makes it visible
whether semantic alignment and order handling add real separation. The bundled
demo is only a synthetic regression seed; production thresholds require
task-specific rollout labels.

Signals exported from `_score_once` stats:

- `matched_sims` — cosine at each aligned (step, window) pair
- `mean_matched_sim` / `min_matched_sim`
- `mean_margin` — average gap between best and second-best window per matched step

Low margin means the match is ambiguous (several windows tie) — a strong fallback trigger when `RoutingConfig.min_margin > 0`. On real PR-fix data, mean margin is often **~0.008**; the default keeps `min_margin=0` as a hard gate and uses margin only in the composite confidence score.

When `require_trace_when_high_match=True` (default), a response with `match_rate ≥ 0.95` but **no execution/reasoning trace** triggers `high_match_without_execution_trace`. This separates padded recitation (high match, no diff/code) from genuine PRs (high match, has trace) without enabling the hard trace score gate.

The router also computes `intention_recitation_fraction`: the fraction of
reference steps whose exact candidate text appears only in a sentence with an
explicit future/intention marker (`will`, `plan to`, `intend to`, `将`, `计划`,
and related forms). At or above
`RoutingConfig.max_intention_recitation_fraction`, the sample falls back to a
Judge. This narrow signal catches partial fabrication where one genuine tool
trace is followed by recited claims for the remaining steps.

## Complexity

- Encoding: one batched encoder forward over `num_steps + num_windows` texts (amortized to ~0 by the LRU cache on repeated inputs).
- Similarity: one GEMM, `O(num_steps × num_windows × dim)`.
- DP: `O(num_steps × num_windows)`, on CPU, typically <1ms after window capping.
