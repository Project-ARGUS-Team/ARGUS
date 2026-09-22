"""Stage 2 relevance scoring for ARGUS."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Protocol

from argus.simulation.context import AgentContext


@dataclass(frozen=True, slots=True)
class RelevanceSignals:
    """The five normalized relevance signals used by ARGUS."""

    spatial_relevance: float
    interaction_probability: float
    goal_importance: float
    event_participation: float
    social_connectivity: float


@dataclass(frozen=True, slots=True)
class RelevanceScore:
    """A normalized aggregate relevance score and its component signals."""

    signals: RelevanceSignals
    score: float


class IRelevanceScorer(Protocol):
    """Interface for components that calculate agent relevance."""

    def score(self, context: AgentContext) -> RelevanceScore:
        """Calculate relevance for one agent context."""


@dataclass(frozen=True, slots=True)
class RuleBasedRelevanceScorer:
    """Deterministic Stage 2 relevance scorer.

    Stage 2 is intentionally descriptive: the score is recorded but does
    not change scheduling. All signals are normalized to [0, 1] and equally
    weighted by default.
    """

    spatial_radius: float = 25.0
    interaction_scale: float = 3.0

    def score(self, context: AgentContext) -> RelevanceScore:
        signals = RelevanceSignals(
            spatial_relevance=self._spatial_relevance(context),
            interaction_probability=self._interaction_probability(context),
            goal_importance=self._clamp01(context.goal.importance),
            event_participation=self._event_participation(context),
            social_connectivity=self._social_connectivity(context),
        )
        score = (
            signals.spatial_relevance
            + signals.interaction_probability
            + signals.goal_importance
            + signals.event_participation
            + signals.social_connectivity
        ) / 5.0
        return RelevanceScore(signals=signals, score=score)

    def _spatial_relevance(self, context: AgentContext) -> float:
        if not context.active_events:
            return 0.0
        nearest_distance = min(event.distance for event in context.active_events)
        return self._clamp01(1.0 - nearest_distance / self.spatial_radius)

    def _interaction_probability(self, context: AgentContext) -> float:
        nearby_count = len(context.nearby_agents)
        return self._clamp01(1.0 - exp(-nearby_count / self.interaction_scale))

    def _event_participation(self, context: AgentContext) -> float:
        if not context.active_events:
            return 0.0
        participating = sum(
            1 for event in context.active_events if event.is_participant
        )
        return participating / len(context.active_events)

    def _social_connectivity(self, context: AgentContext) -> float:
        population = len(context.nearby_agents) + 1
        if population <= 1:
            return 0.0
        return self._clamp01(
            len(context.social_connections) / (population - 1)
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
