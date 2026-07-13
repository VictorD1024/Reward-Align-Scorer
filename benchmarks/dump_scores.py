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

from raise_scorer import ScorerConfig, SemanticRewardScorer
from raise_scorer.embedding import load_embedding_backend
from raise_scorer.scorer import _normalize_steps, _step_repr


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
    parser.add_argument(
        "--require-trace",
        action="store_true",
        help="enable P0 trace gate: deny matches whose window has no tool/reasoning evidence (kills padded recitation)",
    )
    args = parser.parse_args()

    backend = load_embedding_backend(args.model_path)
    if backend is None:
        raise RuntimeError(f"cannot load embedding model: {args.model_path}")
    scorer = SemanticRewardScorer(
        backend, ScorerConfig(threshold=args.threshold, require_trace=args.require_trace)
    )
    # When the trace gate is on, also run an ungated pass to surface steps the
    # gate DENIED on otherwise-matching responses — the recall risk to watch.
    # Share the embedder so the ungated pass reuses cached window embeddings.
    ungated = None
    if args.require_trace:
        ungated = SemanticRewardScorer(backend, ScorerConfig(threshold=args.threshold, require_trace=False))
        ungated.embedder = scorer.embedder

    samples = list(_load_samples(args.input))
    if not samples:
        print("no samples", file=sys.stderr)
        sys.exit(1)

    rows = []
    for i, obj in enumerate(samples):
        r = scorer.score(obj["response"], obj["reference_steps"])
        denied_names: list[str] = []
        if ungated is not None:
            u = ungated.score(obj["response"], obj["reference_steps"])
            gated_idx = {si for si, _ in r.alignment_path}
            ungated_idx = {si for si, _ in u.alignment_path}
            norm = _normalize_steps(obj["reference_steps"])
            denied_names = [_step_repr(norm[si]) for si in sorted(ungated_idx - gated_idx) if si < len(norm)]
        rows.append((i, obj, r, denied_names))

    if args.require_trace:
        print(
            f"{'idx':>4} {'score':>7} {'match':>6} {'order':>6} {'denied':>6}  matched / unmatched"
        )
        print("-" * 88)
        for i, _obj, r, denied in rows:
            print(
                f"{i:>4} {r.score:>7.3f} {r.match_rate:>6.2f} {r.order_rate:>6.2f} {len(denied):>6}  "
                f"{r.matched_steps} / {r.unmatched_steps}"
            )
        # Summary recall-watch: how many samples lost >=1 step to the gate.
        hurt = [i for i, _, _, d in rows if d]
        print(f"\n[recall watch] {len(hurt)}/{len(rows)} samples lost >=1 step to the trace gate: {hurt}")
    else:
        print(f"{'idx':>4} {'score':>7} {'match':>6} {'order':>6}  matched / unmatched")
        print("-" * 80)
        for i, _obj, r, _denied in rows:
            print(
                f"{i:>4} {r.score:>7.3f} {r.match_rate:>6.2f} {r.order_rate:>6.2f}  "
                f"{r.matched_steps} / {r.unmatched_steps}"
            )

    print(f"\n=== top-{args.top} by score (eyeball: did it actually DO the work?) ===")
    ranked = sorted(rows, key=lambda x: x[2].score, reverse=True)[: args.top]
    for rank, (i, obj, r, denied) in enumerate(ranked, 1):
        print(
            f"\n--- rank {rank} idx={i} score={r.score:.3f} "
            f"match={r.match_rate:.2f} order={r.order_rate:.2f} ---"
        )
        print(f"reference_steps: {obj['reference_steps']}")
        print(f"matched_steps: {r.matched_steps}")
        if denied:
            print(f"trace_denied (ungated matched, gated killed): {denied}")
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
