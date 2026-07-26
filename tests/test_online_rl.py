import json
from pathlib import Path

import pytest

from raise_scorer.experiments import (
    EpisodeResult,
    OnlineRLSuite,
    aggregate_episode_results,
    assign_turn_rewards,
    default_three_domain_suite,
)


def test_default_suite_is_three_by_four_by_three():
    suite = default_three_domain_suite()

    assert [domain.name for domain in suite.domains] == ["code", "browser", "tool"]
    assert suite.runs == 36
    assert OnlineRLSuite.from_dict(suite.to_dict()) == suite


def test_checked_in_suite_matches_the_default_protocol():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "online_rl"
        / "three_domain.json"
    )

    suite = OnlineRLSuite.from_dict(json.loads(path.read_text()))

    assert suite == default_three_domain_suite()


def test_turn_rewards_are_placed_on_agent_turn_boundaries():
    token_rewards = assign_turn_rewards(8, [2, 5, 7], [0.2, -0.1, 0.9])

    assert token_rewards == [0.0, 0.0, 0.2, 0.0, 0.0, -0.1, 0.0, 0.9]
    assert sum(token_rewards) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        assign_turn_rewards(8, [5, 2], [0.2, 0.3])


def test_episode_aggregation_keeps_seed_level_statistics():
    rows = [
        EpisodeResult("code", "graph", 1, "a", True, 1.0, 0.8, 0.9, 4),
        EpisodeResult(
            "code",
            "graph",
            1,
            "b",
            False,
            0.0,
            0.4,
            0.2,
            6,
            invalid_transitions=1,
        ),
    ]

    summary = aggregate_episode_results(rows)[0]

    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["mean_turns"] == pytest.approx(5.0)
    assert summary["invalid_transition_rate"] == pytest.approx(0.5)
