#!/usr/bin/env python3
"""Reward signal quality report: genuine vs recitation, confidence routing stats.

Usage:
    python benchmarks/reward_quality.py --model-path /path/to/embedding_model
    python benchmarks/reward_quality.py --model-path .demo_models/bge-small-en-v1.5 \\
        --input data/github_fix_trajectories/github_fixes.jsonl \\
        --demo benchmarks/demo_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raise_scorer.confidence import assess_confidence, classify_steps
from raise_scorer.embedding import load_embedding_backend
from raise_scorer.scorer import ScorerConfig, SemanticRewardScorer


def _load_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _summarize(name: str, rows: list[dict], scorer: SemanticRewardScorer) -> dict:
    scores = []
    confidences = []
    fallbacks = 0
    proxy_heavy = 0

    for obj in rows:
        steps = obj["reference_steps"]
        result = scorer.score(obj["response"], steps)
        report = assess_confidence(result, steps, response=obj["response"])
        scores.append(result.score)
        confidences.append(report.confidence)
        if report.fallback_recommended:
            fallbacks += 1
        tiers = classify_steps(steps)
        if sum(1 for t in tiers if t["tier"] == "proxy") / max(len(tiers), 1) >= 0.6:
            proxy_heavy += 1

    n = len(rows) or 1
    return {
        "name": name,
        "count": len(rows),
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
        "fallback_rate": fallbacks / n,
        "proxy_heavy_rate": proxy_heavy / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reward signal quality report")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", default="data/github_fix_trajectories/github_fixes.jsonl")
    parser.add_argument("--demo", default="benchmarks/demo_samples.jsonl")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--require-trace", action="store_true")
    args = parser.parse_args()

    backend = load_embedding_backend(args.model_path)
    if backend is None:
        print("[error] failed to load embedding backend", file=sys.stderr)
        return 1

    scorer = SemanticRewardScorer(
        backend,
        ScorerConfig(threshold=args.threshold, require_trace=args.require_trace),
    )

    genuine_path = Path(args.input)
    demo_path = Path(args.demo)
    if not genuine_path.exists():
        print(f"[warn] missing {genuine_path}, skipping genuine set", file=sys.stderr)
        genuine_rows = []
    else:
        genuine_rows = _load_jsonl(genuine_path)

    demo_rows = _load_jsonl(demo_path) if demo_path.exists() else []

    print("=== reward signal quality ===")
    if genuine_rows:
        g = _summarize("genuine (github PR-fix)", genuine_rows, scorer)
        print(f"\n[{g['name']}] n={g['count']}")
        print(f"  mean score:       {g['mean_score']:.3f}")
        print(f"  mean confidence:  {g['mean_confidence']:.3f}")
        print(f"  fallback rate:    {g['fallback_rate']:.1%}")
        print(f"  proxy-heavy refs: {g['proxy_heavy_rate']:.1%}")

    if demo_rows:
        labels = ["genuine+tool", "short recit", "paraphrase", "padded recit"]
        print("\n[demo samples]")
        for i, obj in enumerate(demo_rows):
            label = labels[i] if i < len(labels) else f"sample{i}"
            result = scorer.score(obj["response"], obj["reference_steps"])
            report = assess_confidence(result, obj["reference_steps"], response=obj["response"])
            print(
                f"  {label:<14} score={result.score:.3f} conf={report.confidence:.2f} "
                f"fallback={report.fallback_recommended} reasons={report.reasons or '-'}"
            )

    if genuine_rows and demo_rows:
        padded = demo_rows[-1]
        gen_mean = statistics.mean(
            scorer.score(r["response"], r["reference_steps"]).score for r in genuine_rows
        )
        pad_score = scorer.score(padded["response"], padded["reference_steps"]).score
        print("\n[separation]")
        print(f"  genuine mean score: {gen_mean:.3f}")
        print(f"  padded recit score: {pad_score:.3f}")
        if args.require_trace:
            print("  (trace gate enabled — padded recit should be 0.0)")

    print("\nTip: high fallback rate on genuine data → relax RoutingConfig or improve step design.")
    print("See docs/design.md#reference-step-design-guidelines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
