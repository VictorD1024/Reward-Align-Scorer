from pathlib import Path

import pytest

from benchmarks.reward_quality import (
    _lexical_coverage,
    _load_jsonl,
    binary_ranking_metrics,
    build_report,
    pairwise_accuracy,
)


def _record(label, score, pair_id=None, subtype="test"):
    return {
        "label": label,
        "score": score,
        "trusted_score": score,
        "lexical_score": score,
        "pair_id": pair_id,
        "subtype": subtype,
        "confidence": score,
        "fallback_recommended": False,
        "trace_denied": False,
    }


def test_binary_ranking_metrics_perfect_separation():
    records = [
        _record(1, 0.9),
        _record(1, 0.8),
        _record(0, 0.2),
        _record(0, 0.1),
    ]

    metrics = binary_ranking_metrics(records)

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)


def test_binary_ranking_metrics_handle_ties():
    records = [_record(1, 0.5), _record(0, 0.5)]

    metrics = binary_ranking_metrics(records)

    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_pairwise_accuracy_uses_only_complete_groups():
    records = [
        _record(1, 0.9, "good"),
        _record(0, 0.2, "good"),
        _record(1, 0.5, "tie"),
        _record(0, 0.5, "tie"),
        _record(1, 0.8, "positive-only"),
    ]

    metrics = pairwise_accuracy(records)

    assert metrics["groups"] == 2
    assert metrics["comparisons"] == 2
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_lexical_coverage_supports_candidate_actions():
    score = _lexical_coverage(
        "I chose to open the knowledge base and answer with citation.",
        [["open web source", "open the knowledge base"], "answer with citation"],
    )

    assert score == pytest.approx(1.0)


def test_build_report_includes_semantic_baseline_routing_and_subtypes():
    records = [
        _record(1, 0.9, "pair", "genuine"),
        _record(0, 0.1, "pair", "recitation"),
    ]

    report = build_report(records)

    assert report["dataset"] == {
        "count": 2,
        "labeled": 2,
        "positives": 1,
        "negatives": 1,
    }
    assert report["semantic"]["auroc"] == pytest.approx(1.0)
    assert report["routed"]["auroc"] == pytest.approx(1.0)
    assert report["semantic_pairwise"]["accuracy"] == pytest.approx(1.0)
    assert set(report["by_subtype"]) == {"genuine", "recitation"}


def test_routed_metrics_separate_fallbacks_from_trusted_scores():
    positive = _record(1, 0.8, "pair", "genuine")
    negative = _record(0, 0.9, "pair", "recitation")
    negative["trusted_score"] = 0.0
    negative["fallback_recommended"] = True

    report = build_report([positive, negative])

    assert report["semantic"]["auroc"] == pytest.approx(0.0)
    assert report["routed"]["auroc"] == pytest.approx(1.0)
    assert report["semantic_pairwise"]["accuracy"] == pytest.approx(0.0)
    assert report["routed_pairwise"]["accuracy"] == pytest.approx(1.0)


def test_demo_dataset_has_labeled_hard_negatives_and_complete_pairs():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "demo_samples.jsonl"
    rows = _load_jsonl(path)

    assert len(rows) >= 10
    assert {row["label"] for row in rows} == {0, 1}
    assert all(row.get("id") and row.get("subtype") for row in rows)

    pair_labels = {}
    for row in rows:
        pair_labels.setdefault(row["pair_id"], set()).add(row["label"])
    assert pair_labels
    assert all(labels == {0, 1} for labels in pair_labels.values())
