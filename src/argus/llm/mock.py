"""Deterministic mock cognitive provider."""

from argus.simulation.agent import Action, ActionType
from argus.simulation.context import AgentContext, StateDelta


class MockLLMProvider:
    """Simple deterministic provider used for development and testing."""

    def __init__(self) -> None:
        self.call_count = 0

    def request_cognitive_update(self, context: AgentContext) -> StateDelta:
        self.call_count += 1

        if context.goal.target_position is None:
            return StateDelta(action=Action(action_type=ActionType.WAIT))

        return StateDelta(
            action=Action(
                action_type=ActionType.MOVE,
                target_position=context.goal.target_position,
            )
        )
