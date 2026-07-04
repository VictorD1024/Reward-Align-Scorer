"""Dump per-sample scores to eyeball reward hacking (P0 instrumentation).

Feed it a JSONL file where each line is::

    {"response": "...", "reference_steps": ["...", ...]}

It scores every sample, prints a compact table (score / match_rate /
order_rate / matched / unmatched), then prints the top-N by score with the
window text each matched step aligned to and the response head. Scan that
top-N by eye: if the highest-scoring responses just recite the step names
without any tool calls / code / output, recitation hacking is happening and
the P0 UNION TraceDetector is justified. The matched-window text is also the
real sample set to calibrate that detector's patterns on.
"""

import argparse
import json
import sys

from reward_align_scorer import ScorerConfig, SemanticRewardScorer
from reward_align_scorer.embedding import load_embedding_backend


def _load_samples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "response" not in obj or "reference_steps" not in obj:
                print(f"[warn] line {lineno} missing response/reference_steps, skipped", file=sys.stderr)
                continue
            yield obj


def main():
    parser = argparse.ArgumentParser(description="Dump per-sample scores to eyeball reward hacking.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True, help="JSONL file: each line {response, reference_steps}")
    parser.add_argument("--top", type=int, default=10, help="print top-N by score with matched window text")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    backend = load_embedding_backend(args.model_path)
    if backend is None:
        raise RuntimeError(f"cannot load embedding model: {args.model_path}")
    scorer = SemanticRewardScorer(backend, ScorerConfig(threshold=args.threshold))

    samples = list(_load_samples(args.input))
    if not samples:
        print("no samples", file=sys.stderr)
        sys.exit(1)

    rows = []
    for i, obj in enumerate(samples):
        r = scorer.score(obj["response"], obj["reference_steps"])
        rows.append((i, obj, r))

    print(f"{'idx':>4} {'score':>7} {'match':>6} {'order':>6}  matched / unmatched")
    print("-" * 80)
    for i, _obj, r in rows:
        print(
            f"{i:>4} {r.score:>7.3f} {r.match_rate:>6.2f} {r.order_rate:>6.2f}  "
            f"{r.matched_steps} / {r.unmatched_steps}"
        )

    print(f"\n=== top-{args.top} by score (eyeball: did it actually DO the work?) ===")
    ranked = sorted(rows, key=lambda x: x[2].score, reverse=True)[: args.top]
    for rank, (i, obj, r) in enumerate(ranked, 1):
        print(
            f"\n--- rank {rank} idx={i} score={r.score:.3f} "
            f"match={r.match_rate:.2f} order={r.order_rate:.2f} ---"
        )
        print(f"reference_steps: {obj['reference_steps']}")
        print(f"matched_steps: {r.matched_steps}")
        path_by_step = {}
        for si, wi in r.alignment_path:
            path_by_step.setdefault(si, wi)
        matched_step_indices = sorted(path_by_step)
        for k, si in enumerate(matched_step_indices):
            wi = path_by_step[si]
            wtext = r.windows[wi][0].replace("\n", " ")
            if len(wtext) > 200:
                wtext = wtext[:200] + "…"
            print(f"  step {si} ({r.matched_steps[k]}) -> window[{wi}]: {wtext}")
        head = obj["response"].strip().replace("\n", " ")
        if len(head) > 240:
            head = head[:240] + "…"
        print(f"response head: {head}")


if __name__ == "__main__":
    main()
