import pytest

pytest.importorskip("transformers")

from reward_align_scorer.verl_adapter import parse_reference_steps


def test_canonical_key():
    assert parse_reference_steps({"reference_steps": ["a", "b"]}) == ["a", "b"]


def test_actions_alias():
    assert parse_reference_steps({"actions": ["x", "y"]}) == ["x", "y"]


def test_priority_reference_over_actions():
    assert parse_reference_steps({"reference_steps": ["a"], "actions": ["b"]}) == ["a"]


def test_string_input_splits_on_punctuation():
    assert parse_reference_steps({"reference_steps": "a, b；c\nd"}) == ["a", "b", "c", "d"]


def test_nested_candidate_lists_preserved():
    # A step may be a list of candidate actions; any one matching credits the step.
    assert parse_reference_steps({"reference_steps": ["search", ["open web", "open kb"], "answer"]}) == [
        "search",
        ["open web", "open kb"],
        "answer",
    ]


def test_empty_and_missing():
    assert parse_reference_steps({}) == []
    assert parse_reference_steps({"reference_steps": []}) == []
