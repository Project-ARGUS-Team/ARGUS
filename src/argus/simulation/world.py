"""Simulation world and movement mechanics."""

from dataclasses import dataclass
import math

from argus.simulation.agent import ActionType, AgentState, Vector2


@dataclass(frozen=True, slots=True)
class RoadSegment:
    """A connected road segment in the simulation navigation network."""

    road_id: str
    name: str
    start: Vector2
    end: Vector2
    speed_limit: float = 6.0
    capacity: int = 8
    pedestrian_allowed: bool = True
    vehicle_allowed: bool = True

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


@dataclass(frozen=True, slots=True)
class World:
    """Static properties of the simulation environment."""

    width: float = 100.0
    height: float = 100.0
    tick_duration: float = 1.0

    def clamp_position(self, position: Vector2) -> Vector2:
        return Vector2(
            x=min(max(position.x, 0.0), self.width),
            y=min(max(position.y, 0.0), self.height),
        )

    def advance_agent(self, agent: AgentState) -> None:
        if not agent.active:
            return

        action = agent.current_action
        if action is None or action.action_type != ActionType.MOVE:
            return

        target = action.target_position
        if target is None:
            return

        dx = target.x - agent.position.x
        dy = target.y - agent.position.y
        distance = math.hypot(dx, dy)
        step_distance = math.hypot(
            agent.velocity.x,
            agent.velocity.y,
        ) * self.tick_duration

        # Movement targets include intermediate road waypoints. Never let an
        # agent overshoot one: doing so makes the route controller reverse
        # direction on the next tick and produces visible junction oscillation.
        if distance <= step_distance:
            agent.position = self.clamp_position(target)
            agent.velocity = Vector2(0.0, 0.0)
            return

        new_position = agent.position + agent.velocity * self.tick_duration
        agent.position = self.clamp_position(new_position)
