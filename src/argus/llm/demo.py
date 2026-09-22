"""Deterministic cognitive provider for the baseline demonstration."""

from __future__ import annotations

from argus.simulation.agent import Action, ActionType
from argus.simulation.context import AgentContext, StateDelta


class ScenarioLLMProvider:
    """A deterministic provider that produces simple goal-directed behavior.

    This provider intentionally models the cognitive interface without an
    external API. It is useful for reproducible baseline demonstrations and
    experiments.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def request_cognitive_update(self, context: AgentContext) -> StateDelta:
        self.call_count += 1

        if context.active_events:
            event = min(context.active_events, key=lambda item: item.distance)
            if event.is_participant:
                return StateDelta(
                    action=Action(
                        action_type=ActionType.INTERACT,
                        target_position=event.position,
                    )
                )

        if context.goal.target_position is None:
            return StateDelta(action=Action(action_type=ActionType.WAIT))

        return StateDelta(
            action=Action(
                action_type=ActionType.MOVE,
                target_position=context.goal.target_position,
            )
        )
