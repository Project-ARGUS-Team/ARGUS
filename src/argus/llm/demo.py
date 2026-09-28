"""Deterministic cognitive provider for the baseline demonstration."""

from __future__ import annotations

import math

from argus.simulation.agent import Action, ActionType
from argus.simulation.context import AgentContext, StateDelta
from argus.simulation.world import Vector2
from argus.simulation.scenario import RoadSegment


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
        roads: tuple[RoadSegment, ...] = (),
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
        self.roads = roads
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



    def _road_waypoint(self, position: Vector2, target: Vector2, agent_id: str = "") -> Vector2:
        """Return a waypoint on the pedestrian network toward a target."""
        if not self.roads:
            return target

        # A small deterministic fraction of trips takes a pedestrian shortcut.
        index = self._agent_index(agent_id)
        if (index + int(target.x * 3) + int(target.y * 5)) % 13 == 0:
            return target

        nodes: list[Vector2] = []
        edges: dict[tuple[float, float], list[tuple[Vector2, float]]] = {}
        for road in self.roads:
            for a, b in ((road.start, road.end), (road.end, road.start)):
                ka = (a.x, a.y)
                nodes.extend((a, b))
                edges.setdefault(ka, []).append(
                    (b, self._distance(a, b))
                )

        unique: dict[tuple[float, float], Vector2] = {
            (node.x, node.y): node for node in nodes
        }
        start = min(unique.values(), key=lambda p: self._distance(position, p))
        goal = min(unique.values(), key=lambda p: self._distance(target, p))

        distances = {(start.x, start.y): 0.0}
        previous: dict[tuple[float, float], tuple[float, float] | None] = {
            (start.x, start.y): None
        }
        unvisited = set(distances)
        while unvisited:
            current_key = min(
                unvisited,
                key=lambda key: distances.get(key, float("inf"))
                + self._distance(unique[key], goal),
            )
            unvisited.remove(current_key)
            if current_key == (goal.x, goal.y):
                break
            for neighbour, cost in edges.get(current_key, ()):
                key = (neighbour.x, neighbour.y)
                new_distance = distances[current_key] + cost
                if new_distance < distances.get(key, float("inf")):
                    distances[key] = new_distance
                    previous[key] = current_key
                    unvisited.add(key)

        goal_key = (goal.x, goal.y)
        if goal_key not in previous:
            return target

        path = [goal_key]
        while path[-1] != (start.x, start.y):
            parent = previous.get(path[-1])
            if parent is None:
                break
            path.append(parent)
        path.reverse()

        if len(path) <= 1:
            return target
        waypoint = unique[path[1]]
        return waypoint if self._distance(position, waypoint) > self.arrival_radius else target

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

            # Never abandon an unfinished trip just because the clock crossed
            # into the next activity. Physical arrival takes precedence.
            if (
                context.current_action is not None
                and context.current_action.action_type == ActionType.MOVE
                and context.current_action.target_position is not None
                and distance > self.arrival_radius
                and self._distance(
                    context.position,
                    context.current_action.target_position,
                ) > self.arrival_radius
            ):
                target = self._road_waypoint(
                    context.position,
                    context.current_action.target_position,
                    context.agent_id,
                )
                return StateDelta(
                    goal=goal,
                    activity=activity,
                    action=Action(
                        action_type=ActionType.MOVE,
                        target_position=target,
                    ),
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
