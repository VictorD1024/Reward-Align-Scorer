from types import SimpleNamespace

import pytest

import raise_scorer.integrations.verl as verl_adapter
from raise_scorer.integrations import (
    normalize_reference_steps,
    resolve_reference_steps,
)


def test_canonical_extra_info_fields_keep_priority():
    resolution = resolve_reference_steps(
        {
            "reference_steps": ["canonical"],
            "actions": ["alias"],
            "metadata": {"reference_steps": ["nested"]},
        },
        {"reference_steps": ["ground truth"]},
    )

    assert resolution.steps == ["canonical"]
    assert resolution.source == "extra_info.reference_steps"


@pytest.mark.parametrize(
    ("extra_info", "expected_source"),
    [
        (
            {"metadata": {"reference_steps": ["inspect", "test"]}},
            "extra_info.metadata.reference_steps",
        ),
        (
            {"rlvr": {"actions": ["inspect", "test"]}},
            "extra_info.rlvr.actions",
        ),
        (
            {
                "reward_model": {
                    "ground_truth": {
                        "reference_steps": ["inspect", "test"],
                    }
                }
            },
            "extra_info.reward_model.ground_truth.reference_steps",
        ),
    ],
)
def test_known_nested_rlvr_paths_are_supported(extra_info, expected_source):
    resolution = resolve_reference_steps(extra_info)

    assert resolution.steps == ["inspect", "test"]
    assert resolution.source == expected_source


def test_custom_nested_field_path_is_supported():
    resolution = resolve_reference_steps(
        {"task": {"annotation": {"plan": ["inspect", "patch", "test"]}}},
        field_paths=["task.annotation.plan"],
    )

    assert resolution.steps == ["inspect", "patch", "test"]
    assert resolution.source == "extra_info.task.annotation.plan"


@pytest.mark.parametrize(
    ("ground_truth", "expected_source"),
    [
        (
            {"reference_steps": ["inspect", "test"]},
            "ground_truth.reference_steps",
        ),
        (
            {"actions": ["inspect", "test"]},
            "ground_truth.actions",
        ),
        (
            ["inspect", "test"],
            "ground_truth",
        ),
        (
            '["inspect", "test"]',
            "ground_truth",
        ),
        (
            '{"reference_steps": ["inspect", "test"]}',
            "ground_truth.reference_steps",
        ),
    ],
)
def test_structured_parquet_ground_truth_is_supported(
    ground_truth,
    expected_source,
):
    resolution = resolve_reference_steps({}, ground_truth)

    assert resolution.steps == ["inspect", "test"]
    assert resolution.source == expected_source


def test_plain_scalar_ground_truth_is_not_misread_as_process_steps():
    resolution = resolve_reference_steps({}, "42")

    assert resolution.steps == []
    assert resolution.source is None


def test_arrow_or_pandas_list_like_values_are_converted_with_tolist():
    arrow_like = SimpleNamespace(tolist=lambda: ["inspect", "test"])

    assert normalize_reference_steps(arrow_like) == ["inspect", "test"]


def test_step_structs_and_candidate_actions_are_normalized():
    steps = normalize_reference_steps(
        [
            {"action": "inspect repository"},
            {"actions": ["patch code", "apply workaround"]},
            {"unsupported": "ignored"},
        ]
    )

    assert steps == [
        "inspect repository",
        ["patch code", "apply workaround"],
    ]


def test_invalid_custom_path_configuration_is_rejected():
    with pytest.raises(TypeError, match="reference_steps_paths"):
        resolve_reference_steps({}, field_paths=123)
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_reference_steps({}, field_paths=["..."])


def test_compute_score_uses_structured_ground_truth_and_reports_source(monkeypatch):
    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: None)

    details = verl_adapter.compute_score(
        solution_str="inspect repository",
        ground_truth={"reference_steps": ["inspect repository"]},
        extra_info={},
        return_details=True,
    )

    assert details["score"] == pytest.approx(1.0)
    assert details["reference_steps_source"] == "ground_truth.reference_steps"


def test_compute_score_accepts_custom_reference_steps_path(monkeypatch):
    monkeypatch.setattr(verl_adapter, "_get_scorer", lambda: None)

    details = verl_adapter.compute_score(
        solution_str="run tests",
        extra_info={"labels": {"process": ["run tests"]}},
        reference_steps_paths=["labels.process"],
        return_details=True,
    )

    assert details["score"] == pytest.approx(1.0)
    assert details["reference_steps_source"] == "extra_info.labels.process"
