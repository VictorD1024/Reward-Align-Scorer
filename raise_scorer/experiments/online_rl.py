"""Three-domain online-RL protocol without environment-specific adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DomainProtocol:
    name: str
    environment: str
    train_split: str
    evaluation_split: str
    primary_metric: str
    max_turns: int
    required_event_fields: tuple[str, ...] = (
        "event_id",
        "action",
        "node_id",
        "status",
        "source",
        "trusted",
    )

    def __post_init__(self) -> None:
        if not self.name or not self.environment:
            raise ValueError("domain name and environment must not be empty")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")


@dataclass(frozen=True)
class RewardAblation:
    name: str
    alignment_mode: str
    turn_potential: bool
    outcome_weight: float
    process_weight: float

    def __post_init__(self) -> None:
        if self.alignment_mode not in {"none", "enumerate", "graph"}:
            raise ValueError("alignment_mode must be none, enumerate, or graph")
        if self.outcome_weight < 0 or self.process_weight < 0:
            raise ValueError("reward weights must be non-negative")
        if self.outcome_weight + self.process_weight <= 0:
            raise ValueError("at least one reward weight must be positive")


@dataclass(frozen=True)
class OnlineRLSuite:
    domains: tuple[DomainProtocol, ...]
    ablations: tuple[RewardAblation, ...]
    seeds: tuple[int, ...] = (1, 2, 3)
    algorithm: str = "GRPO"

    def __post_init__(self) -> None:
        if len(self.domains) != 3:
            raise ValueError("the standard suite must contain exactly three domains")
        if len({domain.name for domain in self.domains}) != len(self.domains):
            raise ValueError("domain names must be unique")
        if len({ablation.name for ablation in self.ablations}) != len(self.ablations):
            raise ValueError("ablation names must be unique")
        if not self.seeds:
            raise ValueError("at least one random seed is required")

    @property
    def runs(self) -> int:
        return len(self.domains) * len(self.ablations) * len(self.seeds)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OnlineRLSuite":
        return cls(
            domains=tuple(DomainProtocol(**item) for item in data["domains"]),
            ablations=tuple(RewardAblation(**item) for item in data["ablations"]),
            seeds=tuple(int(seed) for seed in data.get("seeds", (1, 2, 3))),
            algorithm=str(data.get("algorithm", "GRPO")),
        )


def default_three_domain_suite() -> OnlineRLSuite:
    """Return the paper-facing 3 domains × 4 rewards × 3 seeds matrix."""
    return OnlineRLSuite(
        domains=(
            DomainProtocol(
                name="code",
                environment="SWE-Gym",
                train_split="SWE-Gym train",
                evaluation_split="SWE-bench Verified",
                primary_metric="resolved_rate",
                max_turns=40,
            ),
            DomainProtocol(
                name="browser",
                environment="BrowserGym",
                train_split="WebArena train",
                evaluation_split="WebArena Verified",
                primary_metric="task_success_rate",
                max_turns=30,
            ),
            DomainProtocol(
                name="tool",
                environment="BFCL executable sandbox",
                train_split="BFCL multi-turn train",
                evaluation_split="BFCL V4 agentic",
                primary_metric="overall_accuracy",
                max_turns=20,
            ),
        ),
        ablations=(
            RewardAblation(
                name="outcome_only",
                alignment_mode="none",
                turn_potential=False,
                outcome_weight=1.0,
                process_weight=0.0,
            ),
            RewardAblation(
                name="linear_terminal",
                alignment_mode="enumerate",
                turn_potential=False,
                outcome_weight=0.5,
                process_weight=0.5,
            ),
            RewardAblation(
                name="graph_terminal",
                alignment_mode="graph",
                turn_potential=False,
                outcome_weight=0.5,
                process_weight=0.5,
            ),
            RewardAblation(
                name="graph_turn_potential",
                alignment_mode="graph",
                turn_potential=True,
                outcome_weight=0.5,
                process_weight=0.5,
            ),
        ),
    )


def assign_turn_rewards(
    sequence_length: int,
    turn_end_token_indices: Sequence[int],
    turn_rewards: Sequence[float],
) -> list[float]:
    """Place each turn reward on that turn's final response token.

    The output can be added directly to a trainer's token-level reward tensor.
    Indices are zero-based in the completion sequence and must be strictly
    increasing, which prevents accidental reward placement on prompt tokens or
    on the wrong agent turn.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(turn_end_token_indices) != len(turn_rewards):
        raise ValueError("turn boundaries and rewards must have the same length")
    boundaries = [int(index) for index in turn_end_token_indices]
    if any(index < 0 or index >= sequence_length for index in boundaries):
        raise ValueError("turn boundary is outside the completion sequence")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("turn boundaries must be strictly increasing")

    token_rewards = [0.0] * sequence_length
    for index, reward in zip(boundaries, turn_rewards):
        token_rewards[index] += float(reward)
    return token_rewards


@dataclass(frozen=True)
class EpisodeResult:
    domain: str
    ablation: str
    seed: int
    episode_id: str
    success: bool
    outcome_reward: float
    process_reward: float
    total_reward: float
    turns: int
    invalid_transitions: int = 0
    wall_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.domain or not self.ablation or not self.episode_id:
            raise ValueError("episode identity fields must not be empty")
        if self.turns < 0 or self.invalid_transitions < 0:
            raise ValueError("episode counts must be non-negative")
        if self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be non-negative")


def aggregate_episode_results(results: Sequence[EpisodeResult]) -> list[dict]:
    """Aggregate runs by domain, reward ablation, and seed."""
    groups: dict[tuple[str, str, int], list[EpisodeResult]] = {}
    for result in results:
        groups.setdefault(
            (result.domain, result.ablation, result.seed),
            [],
        ).append(result)

    summaries = []
    for (domain, ablation, seed), episodes in sorted(groups.items()):
        summaries.append(
            {
                "domain": domain,
                "ablation": ablation,
                "seed": seed,
                "episodes": len(episodes),
                "success_rate": mean(float(item.success) for item in episodes),
                "mean_outcome_reward": mean(item.outcome_reward for item in episodes),
                "mean_process_reward": mean(item.process_reward for item in episodes),
                "mean_total_reward": mean(item.total_reward for item in episodes),
                "mean_turns": mean(item.turns for item in episodes),
                "invalid_transition_rate": mean(
                    float(item.invalid_transitions > 0) for item in episodes
                ),
                "mean_wall_time_seconds": mean(
                    item.wall_time_seconds for item in episodes
                ),
            }
        )
    return summaries


__all__ = [
    "DomainProtocol",
    "EpisodeResult",
    "OnlineRLSuite",
    "RewardAblation",
    "aggregate_episode_results",
    "assign_turn_rewards",
    "default_three_domain_suite",
]
