"""Simulation world and movement mechanics."""

from dataclasses import dataclass

from argus.simulation.agent import AgentState, ActionType, Vector2


@dataclass(frozen=True, slots=True)
class World:
    """Static properties of the simulation environment."""

    width: float = 100.0
    height: float = 100.0
    tick_duration: float = 1.0

    def clamp_position(self, position: Vector2) -> Vector2:
        """Keep a position inside the world's rectangular bounds."""
        return Vector2(
            x=min(max(position.x, 0.0), self.width),
            y=min(max(position.y, 0.0), self.height),
        )

    def advance_agent(self, agent: AgentState) -> None:
        """Advance one agent by one simulation tick."""
        if not agent.active:
            return

        action = agent.current_action
        if action is None or action.action_type != ActionType.MOVE:
            return

        new_position = agent.position + agent.velocity * self.tick_duration
        agent.position = self.clamp_position(new_position)
