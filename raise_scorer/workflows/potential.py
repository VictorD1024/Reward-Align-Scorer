"""Turn-level potential shaping over workflow-prefix scores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from .engine import WorkflowGroundTruth, WorkflowReward, WorkflowScorer


@dataclass(frozen=True)
class PotentialRewardConfig:
    """Configuration for ``r_t = scale * (gamma * Phi_t - Phi_{t-1})``."""

    gamma: float = 1.0
    scale: float = 1.0
    # Coverage is the stable prefix-progress signal. Full ``score`` also
    # contains an all-pairs order term that can temporarily fall when a long
    # response window begins to contain several otherwise ordered actions.
    potential_field: str = "required_coverage"
    turn_separator: str = "\n\n"

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.scale < 0:
            raise ValueError("scale must be non-negative")
        if self.potential_field not in {
            "score",
            "semantic_score",
            "required_coverage",
        }:
            raise ValueError(
                "potential_field must be score, semantic_score, or required_coverage"
            )


@dataclass
class TurnReward:
    turn_index: int
    potential: float
    shaped_reward: float
    best_path: list[str]
    required_coverage: float
    order_rate: float
    workflow_reward: WorkflowReward


@dataclass
class PotentialRewardResult:
    turns: list[TurnReward]
    gamma: float
    scale: float
    potential_field: str

    @property
    def potentials(self) -> list[float]:
        return [turn.potential for turn in self.turns]

    @property
    def rewards(self) -> list[float]:
        return [turn.shaped_reward for turn in self.turns]

    @property
    def final_potential(self) -> float:
        return self.turns[-1].potential if self.turns else 0.0

    @property
    def total_shaped_reward(self) -> float:
        return sum(self.rewards)

    def to_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "scale": self.scale,
            "potential_field": self.potential_field,
            "potentials": self.potentials,
            "rewards": self.rewards,
            "final_potential": self.final_potential,
            "total_shaped_reward": self.total_shaped_reward,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "potential": turn.potential,
                    "shaped_reward": turn.shaped_reward,
                    "best_path": turn.best_path,
                    "required_coverage": turn.required_coverage,
                    "order_rate": turn.order_rate,
                    "workflow_reward": turn.workflow_reward.to_dict(),
                }
                for turn in self.turns
            ],
        }


class WorkflowPotentialScorer:
    """Convert workflow prefix progress into dense per-turn rewards."""

    def __init__(
        self,
        workflow_scorer: WorkflowScorer,
        config: Optional[PotentialRewardConfig] = None,
    ):
        self.workflow_scorer = workflow_scorer
        self.config = config or PotentialRewardConfig()

    def score_turns(
        self,
        turns: Sequence[str],
        workflow: WorkflowGroundTruth,
        **score_kwargs,
    ) -> PotentialRewardResult:
        """Score cumulative textual prefixes at agent-turn boundaries."""
        if isinstance(turns, (str, bytes)):
            raise TypeError("turns must be a sequence of strings")
        turns = list(turns)
        prefixes = []
        parts = []
        for turn in turns:
            parts.append("" if turn is None else str(turn))
            prefixes.append(self.config.turn_separator.join(parts))
        rewards = [
            self.workflow_scorer.score(prefix, workflow, **score_kwargs)
            for prefix in prefixes
        ]
        potentials = None
        if self.config.potential_field == "required_coverage":
            from .alignment import align_workflow_response

            potentials = [
                align_workflow_response(
                    "",
                    workflow,
                    self.workflow_scorer.semantic_scorer,
                    max_path_nodes=self.workflow_scorer.config.max_path_nodes,
                    max_states=self.workflow_scorer.config.max_graph_states,
                    max_expansions=self.workflow_scorer.config.max_expansions,
                    passages=turns[:index],
                    action_weight=self.workflow_scorer.config.action_weight,
                    evidence_weight=self.workflow_scorer.config.evidence_weight,
                    outcome_weight=self.workflow_scorer.config.outcome_weight,
                ).estimated_coverage
                for index in range(1, len(turns) + 1)
            ]
        return self._shape(rewards, potentials=potentials)

    def score_event_prefixes(
        self,
        events: Sequence[Any],
        workflow: WorkflowGroundTruth,
        *,
        validation_config=None,
        **score_kwargs,
    ) -> PotentialRewardResult:
        """Score trusted runtime-event prefixes without requiring early terminal states."""
        from ..runtime.events import TraceValidationConfig

        events = list(events)
        if validation_config is None:
            validation_config = TraceValidationConfig(require_terminal=False)
        elif isinstance(validation_config, dict):
            validation_config = TraceValidationConfig(
                **{**validation_config, "require_terminal": False}
            )
        else:
            validation_config = replace(validation_config, require_terminal=False)
        rewards = [
            self.workflow_scorer.score_events(
                events[:index],
                workflow,
                validation_config=validation_config,
                **score_kwargs,
            )
            for index in range(1, len(events) + 1)
        ]
        potentials = None
        if self.config.potential_field == "required_coverage":
            from ..runtime.events import render_trace_events
            from .alignment import align_workflow_response

            passages = [render_trace_events([event]) for event in events]
            potentials = [
                align_workflow_response(
                    "",
                    workflow,
                    self.workflow_scorer.semantic_scorer,
                    max_path_nodes=self.workflow_scorer.config.max_path_nodes,
                    max_states=self.workflow_scorer.config.max_graph_states,
                    max_expansions=self.workflow_scorer.config.max_expansions,
                    passages=passages[:index],
                    action_weight=self.workflow_scorer.config.action_weight,
                    evidence_weight=self.workflow_scorer.config.evidence_weight,
                    outcome_weight=self.workflow_scorer.config.outcome_weight,
                ).estimated_coverage
                for index in range(1, len(events) + 1)
            ]
        return self._shape(rewards, potentials=potentials)

    def _shape(
        self,
        rewards: Sequence[WorkflowReward],
        *,
        potentials: Optional[Sequence[float]] = None,
    ) -> PotentialRewardResult:
        turns = []
        previous = 0.0
        for index, reward in enumerate(rewards):
            potential = (
                float(potentials[index])
                if potentials is not None
                else float(getattr(reward, self.config.potential_field))
            )
            shaped = self.config.scale * (
                self.config.gamma * potential - previous
            )
            turns.append(
                TurnReward(
                    turn_index=index,
                    potential=potential,
                    shaped_reward=shaped,
                    best_path=list(reward.best_path),
                    required_coverage=(
                        potential
                        if self.config.potential_field == "required_coverage"
                        else reward.required_coverage
                    ),
                    order_rate=reward.order_rate,
                    workflow_reward=reward,
                )
            )
            previous = potential
        return PotentialRewardResult(
            turns=turns,
            gamma=self.config.gamma,
            scale=self.config.scale,
            potential_field=self.config.potential_field,
        )


__all__ = [
    "PotentialRewardConfig",
    "PotentialRewardResult",
    "TurnReward",
    "WorkflowPotentialScorer",
]
