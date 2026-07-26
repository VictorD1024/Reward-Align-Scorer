#!/usr/bin/env python3
"""Labeled reward-quality benchmark with hard-negative diagnostics.

Input JSONL rows use this schema::

    {
      "id": "bugfix-genuine",
      "response": "...",
      "reference_steps": ["...", "..."],
      "label": 1,
      "subtype": "genuine_tool_trace",
      "pair_id": "bugfix-1"
    }

``label=1`` means the response should receive a high reward; ``label=0`` is a
hard negative. Rows sharing ``pair_id`` are also evaluated pairwise.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raise_scorer.backends import load_embedding_backend  # noqa: E402
from raise_scorer.core import (  # noqa: E402
    ScorerConfig,
    SemanticRewardScorer,
    assess_confidence,
    classify_steps,
)


def _load_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON root must be a list")
        return data

    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: each row must be a JSON object")
        rows.append(obj)
    return rows


def _coerce_label(value) -> int | None:
    if value is None:
        return None
    if value in (1, True, "1", "positive", "genuine"):
        return 1
    if value in (0, False, "0", "negative", "hard_negative"):
        return 0
    raise ValueError(f"unsupported label: {value!r}")


def _normalize_lexical(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def _lexical_coverage(response: str, reference_steps) -> float:
    """Unordered exact-substring baseline used to expose recitation hacking."""
    response = _normalize_lexical(response)
    if not reference_steps:
        return 0.0
    matched = 0
    for step in reference_steps:
        candidates = step if isinstance(step, (list, tuple)) else [step]
        normalized = [_normalize_lexical(str(candidate)) for candidate in candidates]
        if any(candidate and candidate in response for candidate in normalized):
            matched += 1
    return matched / len(reference_steps)


def _mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def binary_ranking_metrics(
    records: list[dict],
    score_key: str = "score",
    decision_threshold: float = 0.5,
) -> dict:
    """Compute dependency-free binary ranking and threshold metrics."""
    labeled = [record for record in records if record.get("label") in (0, 1)]
    positives = [record for record in labeled if record["label"] == 1]
    negatives = [record for record in labeled if record["label"] == 0]

    auroc = None
    if positives and negatives:
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                pos_score = float(positive[score_key])
                neg_score = float(negative[score_key])
                wins += 1.0 if pos_score > neg_score else 0.5 if pos_score == neg_score else 0.0
        auroc = wins / (len(positives) * len(negatives))

    average_precision = None
    if positives:
        # Group equal scores so AP is deterministic and tie-aware.
        groups: dict[float, list[dict]] = defaultdict(list)
        for record in labeled:
            groups[float(record[score_key])].append(record)
        seen = 0
        seen_positive = 0
        precision_sum = 0.0
        for score in sorted(groups, reverse=True):
            group = groups[score]
            group_positive = sum(record["label"] == 1 for record in group)
            seen += len(group)
            seen_positive += group_positive
            precision_sum += group_positive * (seen_positive / seen)
        average_precision = precision_sum / len(positives)

    true_positive = sum(
        record["label"] == 1 and float(record[score_key]) >= decision_threshold for record in labeled
    )
    false_positive = sum(
        record["label"] == 0 and float(record[score_key]) >= decision_threshold for record in labeled
    )
    correct = sum(
        (float(record[score_key]) >= decision_threshold) == bool(record["label"]) for record in labeled
    )
    return {
        "count": len(labeled),
        "positives": len(positives),
        "negatives": len(negatives),
        "positive_mean": _mean(float(record[score_key]) for record in positives),
        "negative_mean": _mean(float(record[score_key]) for record in negatives),
        "mean_separation": (
            _mean(float(record[score_key]) for record in positives)
            - _mean(float(record[score_key]) for record in negatives)
        ),
        "auroc": auroc,
        "average_precision": average_precision,
        "decision_threshold": decision_threshold,
        "accuracy": correct / len(labeled) if labeled else None,
        "true_positive_rate": true_positive / len(positives) if positives else None,
        "false_positive_rate": false_positive / len(negatives) if negatives else None,
    }


def pairwise_accuracy(records: list[dict], score_key: str = "score") -> dict:
    """Compare every labeled positive/negative pair sharing ``pair_id``."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("pair_id") and record.get("label") in (0, 1):
            groups[str(record["pair_id"])].append(record)

    comparisons = 0
    wins = 0.0
    complete_groups = 0
    for group in groups.values():
        positives = [record for record in group if record["label"] == 1]
        negatives = [record for record in group if record["label"] == 0]
        if not positives or not negatives:
            continue
        complete_groups += 1
        for positive in positives:
            for negative in negatives:
                comparisons += 1
                pos_score = float(positive[score_key])
                neg_score = float(negative[score_key])
                wins += 1.0 if pos_score > neg_score else 0.5 if pos_score == neg_score else 0.0
    return {
        "groups": complete_groups,
        "comparisons": comparisons,
        "accuracy": wins / comparisons if comparisons else None,
    }


def _subtype_summary(records: list[dict], decision_threshold: float) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("subtype", "unspecified"))].append(record)

    output = {}
    for subtype, group in sorted(grouped.items()):
        negatives = [record for record in group if record.get("label") == 0]
        positives = [record for record in group if record.get("label") == 1]
        output[subtype] = {
            "count": len(group),
            "mean_score": _mean(record["score"] for record in group),
            "mean_trusted_score": _mean(record["trusted_score"] for record in group),
            "mean_confidence": _mean(record["confidence"] for record in group),
            "fallback_rate": _mean(float(record["fallback_recommended"]) for record in group),
            "semantic_false_accept_rate": (
                _mean(float(record["score"] >= decision_threshold) for record in negatives)
                if negatives
                else None
            ),
            "routed_false_accept_rate": (
                _mean(float(record["trusted_score"] >= decision_threshold) for record in negatives)
                if negatives
                else None
            ),
            "semantic_false_reject_rate": (
                _mean(float(record["score"] < decision_threshold) for record in positives)
                if positives
                else None
            ),
            "routed_false_reject_rate": (
                _mean(float(record["trusted_score"] < decision_threshold) for record in positives)
                if positives
                else None
            ),
        }
    return output


def _score_rows(rows: list[dict], scorer: SemanticRewardScorer) -> list[dict]:
    records = []
    for index, obj in enumerate(rows):
        if "response" not in obj or "reference_steps" not in obj:
            raise ValueError(f"row {index} must contain response and reference_steps")
        steps = obj["reference_steps"]
        response = str(obj["response"])
        result = scorer.score(response, steps)
        report = assess_confidence(result, steps, response=response)
        tiers = classify_steps(steps)
        proxy_fraction = sum(tier["tier"] == "proxy" for tier in tiers) / max(len(tiers), 1)
        records.append(
            {
                "id": str(obj.get("id", f"sample-{index}")),
                "label": _coerce_label(obj.get("label")),
                "subtype": str(obj.get("subtype", "unspecified")),
                "pair_id": obj.get("pair_id"),
                "score": result.score,
                "trusted_score": result.score if not report.fallback_recommended else 0.0,
                "lexical_score": _lexical_coverage(response, steps),
                "confidence": report.confidence,
                "fallback_recommended": report.fallback_recommended,
                "fallback_reasons": report.reasons,
                "match_rate": result.match_rate,
                "order_rate": result.order_rate,
                "proxy_step_fraction": proxy_fraction,
                "trace_denied": result.stats.get("trace_gate") == "denied",
            }
        )
    return records


def build_report(records: list[dict], decision_threshold: float = 0.5) -> dict:
    positives = [record for record in records if record.get("label") == 1]
    negatives = [record for record in records if record.get("label") == 0]
    return {
        "dataset": {
            "count": len(records),
            "labeled": len(positives) + len(negatives),
            "positives": len(positives),
            "negatives": len(negatives),
        },
        "semantic": binary_ranking_metrics(records, "score", decision_threshold),
        "routed": binary_ranking_metrics(records, "trusted_score", decision_threshold),
        "lexical_baseline": binary_ranking_metrics(records, "lexical_score", decision_threshold),
        "semantic_pairwise": pairwise_accuracy(records, "score"),
        "routed_pairwise": pairwise_accuracy(records, "trusted_score"),
        "lexical_pairwise": pairwise_accuracy(records, "lexical_score"),
        "routing": {
            "positive_fallback_rate": _mean(
                float(record["fallback_recommended"]) for record in positives
            ),
            "negative_fallback_rate": _mean(
                float(record["fallback_recommended"]) for record in negatives
            ),
            "positive_trace_denial_rate": _mean(float(record["trace_denied"]) for record in positives),
            "negative_trace_denial_rate": _mean(float(record["trace_denied"]) for record in negatives),
        },
        "by_subtype": _subtype_summary(records, decision_threshold),
        "samples": records,
    }


def _format_metric(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_report(report: dict, show_samples: bool = False) -> None:
    dataset = report["dataset"]
    print(
        f"dataset: n={dataset['count']} labeled={dataset['labeled']} "
        f"positive={dataset['positives']} negative={dataset['negatives']}"
    )
    print(f"{'metric':<24} {'semantic':>10} {'routed':>10} {'lexical':>10}")
    print("-" * 58)
    for key in (
        "positive_mean",
        "negative_mean",
        "mean_separation",
        "auroc",
        "average_precision",
        "true_positive_rate",
    ):
        print(
            f"{key:<24} {_format_metric(report['semantic'][key]):>10} "
            f"{_format_metric(report['routed'][key]):>10} "
            f"{_format_metric(report['lexical_baseline'][key]):>10}"
        )
    print(
        f"{'pairwise_accuracy':<24} "
        f"{_format_metric(report['semantic_pairwise']['accuracy']):>10} "
        f"{_format_metric(report['routed_pairwise']['accuracy']):>10} "
        f"{_format_metric(report['lexical_pairwise']['accuracy']):>10}"
    )
    print(
        f"{'false_positive_rate':<24} "
        f"{_format_metric(report['semantic']['false_positive_rate']):>10} "
        f"{_format_metric(report['routed']['false_positive_rate']):>10} "
        f"{_format_metric(report['lexical_baseline']['false_positive_rate']):>10}"
    )

    print("\nby subtype:")
    for subtype, values in report["by_subtype"].items():
        print(
            f"  {subtype:<28} n={values['count']:<3} "
            f"score={values['mean_score']:.3f} trusted={values['mean_trusted_score']:.3f} "
            f"fallback={values['fallback_rate']:.1%} "
            f"routed_false_accept={_format_metric(values['routed_false_accept_rate'])}"
        )

    if show_samples:
        print("\nsamples:")
        for record in report["samples"]:
            label = "?" if record["label"] is None else record["label"]
            print(
                f"  {record['id']:<28} y={label} score={record['score']:.3f} "
                f"order={record['order_rate']:.2f} conf={record['confidence']:.2f} "
                f"fallback={record['fallback_recommended']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Labeled reward signal quality benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", help="optional additional JSON/JSONL dataset")
    parser.add_argument("--demo", default="benchmarks/demo_samples.jsonl")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--pooling", choices=("cls", "mean"))
    parser.add_argument("--query-instruction")
    parser.add_argument("--passage-instruction")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-trace", action="store_true")
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--min-auroc", type=float)
    parser.add_argument("--min-pairwise-accuracy", type=float)
    parser.add_argument(
        "--gate-score",
        choices=("semantic", "routed"),
        default="routed",
        help="score family used by --min-* quality gates",
    )
    args = parser.parse_args()

    backend = load_embedding_backend(
        args.model_path,
        pooling=args.pooling,
        query_instruction=args.query_instruction,
        passage_instruction=args.passage_instruction,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if backend is None:
        print("[error] failed to load embedding backend", file=sys.stderr)
        return 1
    scorer = SemanticRewardScorer(
        backend,
        ScorerConfig(threshold=args.threshold, require_trace=args.require_trace),
    )

    rows = []
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[error] missing input dataset: {input_path}", file=sys.stderr)
            return 1
        for index, row in enumerate(_load_jsonl(input_path)):
            enriched = dict(row)
            enriched.setdefault("id", f"external-sample-{index}")
            enriched.setdefault("subtype", "external")
            rows.append(enriched)

    demo_path = Path(args.demo)
    if demo_path.exists():
        rows.extend(_load_jsonl(demo_path))
    if not rows:
        print("[error] no benchmark samples", file=sys.stderr)
        return 1

    records = _score_rows(rows, scorer)
    report = build_report(records, decision_threshold=args.decision_threshold)
    report["config"] = {
        "model_path": args.model_path,
        "semantic_threshold": args.threshold,
        "decision_threshold": args.decision_threshold,
        "require_trace": args.require_trace,
        "pooling": backend.pooling,
        "query_instruction": backend.query_instruction,
        "passage_instruction": backend.passage_instruction,
        "revision": backend.revision,
        "input": args.input,
        "demo": args.demo,
        "gate_score": args.gate_score,
    }
    _print_report(report, show_samples=args.show_samples)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {output_path}")

    failures = []
    auroc = report[args.gate_score]["auroc"]
    pairwise = report[f"{args.gate_score}_pairwise"]["accuracy"]
    if args.min_auroc is not None and (auroc is None or auroc < args.min_auroc):
        failures.append(f"AUROC {_format_metric(auroc)} < {args.min_auroc:.3f}")
    if args.min_pairwise_accuracy is not None and (
        pairwise is None or pairwise < args.min_pairwise_accuracy
    ):
        failures.append(
            f"pairwise accuracy {_format_metric(pairwise)} < {args.min_pairwise_accuracy:.3f}"
        )
    if failures:
        for failure in failures:
            print(f"[quality gate failed] {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
