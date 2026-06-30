# Design

Reward Align Scorer is built around one engineering choice:

> Convert structured reward judgment into batched semantic alignment before calling a slow Judge.

## Pipeline

1. Parse reference structure from `reference_steps`, checklist items, or trajectory nodes.
2. Split the long response into overlapping windows.
3. Encode reference items and response windows into normalized embeddings.
4. Compute a dense similarity matrix with matrix multiplication.
5. Run monotonic alignment DP over the similarity matrix.
6. Return score and interpretable alignment details.

## Why Sliding Windows

Sentence boundaries are brittle for long model responses. A required semantic unit may cross punctuation boundaries, and a 4096-token response may contain long explanations around short key actions. Sliding windows preserve local context while bounding the number of semantic comparison units.

## Why Monotonic DP

Greedy matching is vulnerable to cascade errors. If the first reference item matches a later response segment, all remaining items are forced to search after that wrong position. Monotonic DP searches for a globally ordered path through the similarity matrix.

## Complexity

The expensive semantic comparison is handled by batched embedding inference and a matrix multiply:

```text
sim = reference_embeddings @ window_embeddings.T
```

The DP step is `O(num_reference_items * num_windows)`, typically small after window capping.

