"""Deterministic cognitive provider for the baseline demonstration."""

from __future__ import annotations

import math

from argus.simulation.agent import Action, ActionType
from argus.simulation.context import AgentContext, StateDelta
from argus.simulation.world import Vector2


class ScenarioLLMProvider:
    """Deterministic cognitive provider used by the baseline demonstration.

    The provider models a simple recurring daily routine rather than a single
    route. It reacts to active events first, then follows the agent's current
    routine, and finally adds small amounts of social variation.
    """

    def __init__(
        self,
        route_points: tuple[Vector2, ...] = (),
        *,
        cycle_ticks: int = 48,
        dwell_ticks: int = 10,
        arrival_radius: float = 4.0,
        social_radius: float = 6.0,
    ) -> None:
        if cycle_ticks <= 0:
            raise ValueError("cycle_ticks must be positive")
        if not 0 <= dwell_ticks < cycle_ticks:
            raise ValueError("dwell_ticks must be in [0, cycle_ticks)")
        if arrival_radius <= 0:
            raise ValueError("arrival_radius must be positive")
        if social_radius <= 0:
            raise ValueError("social_radius must be positive")

        self.route_points = route_points
        self.cycle_ticks = cycle_ticks
        self.dwell_ticks = dwell_ticks
        self.arrival_radius = arrival_radius
        self.social_radius = social_radius
        self.call_count = 0

    @staticmethod
    def _distance(a: Vector2, b: Vector2) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _agent_index(agent_id: str) -> int:
        try:
            return max(0, int(agent_id.rsplit("-", 1)[-1]) - 1)
        except ValueError:
            return 0

    def _route_target(self, context: AgentContext) -> Vector2 | None:
        if not self.route_points:
            return context.goal.target_position

        index = self._agent_index(context.agent_id)
        phase = (context.simulation_tick + index * 7) % (
            len(self.route_points) * self.cycle_ticks
        )
        route_index = phase // self.cycle_ticks
        return self.route_points[route_index]

    def request_cognitive_update(self, context: AgentContext) -> StateDelta:
        self.call_count += 1

        # Unexpected events take priority over the normal routine.
        participating_events = [
            event for event in context.active_events if event.is_participant
        ]
        if participating_events:
            event = min(participating_events, key=lambda item: item.distance)
            if event.distance <= self.arrival_radius:
                return StateDelta(
                    action=Action(
                        action_type=ActionType.INTERACT,
                        target_position=event.position,
                    ),
                )
            return StateDelta(
                action=Action(
                    action_type=ActionType.MOVE,
                    target_position=event.position,
                ),
            )

        routine = context.current_routine
        if routine is not None:
            activity = routine.activity
            goal = context.goal
            if (
                goal.target_position != routine.target_position
                or goal.description != routine.description
            ):
                goal = type(goal)(
                    goal_id=(
                        f"{context.agent_id}-"
                        f"{context.simulation_tick // 240:04d}-"
                        f"{routine.activity.value}"
                    ),
                    description=routine.description,
                    target_position=routine.target_position,
                    importance=routine.importance,
                )

            distance = self._distance(
                context.position,
                routine.target_position,
            )

            if activity.value == "social" and context.nearby_agents:
                nearby = min(
                    context.nearby_agents,
                    key=lambda item: item.distance,
                )
                if nearby.distance <= self.social_radius:
                    return StateDelta(
                        goal=goal,
                        activity=activity,
                        action=Action(
                            action_type=ActionType.INTERACT,
                            target_agent_id=nearby.agent_id,
                            target_position=nearby.position,
                        ),
                    )

            if distance <= self.arrival_radius:
                return StateDelta(
                    goal=goal,
                    activity=activity,
                    action=Action(
                        action_type=ActionType.WAIT,
                        target_position=routine.target_position,
                    ),
                )

            return StateDelta(
                goal=goal,
                activity=activity,
                action=Action(
                    action_type=ActionType.MOVE,
                    target_position=routine.target_position,
                ),
            )

        # Fallback for agents created outside the city scenario.
        target = self._route_target(context)
        if target is None:
            return StateDelta(action=Action(action_type=ActionType.WAIT))

        distance = self._distance(context.position, target)
        index = self._agent_index(context.agent_id)
        phase = (context.simulation_tick + index * 7) % self.cycle_ticks
        if distance <= self.arrival_radius and phase >= (
            self.cycle_ticks - self.dwell_ticks
        ):
            return StateDelta(action=Action(action_type=ActionType.WAIT))

        return StateDelta(
            action=Action(
                action_type=ActionType.MOVE,
                target_position=target,
            )
        )
