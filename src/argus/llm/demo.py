"""Deterministic cognitive provider for the baseline demonstration."""

from __future__ import annotations

import math

from argus.simulation.agent import Action, ActionType, ActivityType
from argus.simulation.context import AgentContext, StateDelta
from argus.simulation.world import RoadSegment, Vector2


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
        # Per-agent committed road routes. A route is recomputed only when
        # the destination/road constraints change or the current route ends.
        self._route_cache = {}

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



    def _road_route(
        self,
        position: Vector2,
        target: Vector2,
        blocked_road_ids: tuple[str, ...] = (),
        vehicle: bool = False,
    ) -> tuple[Vector2, ...]:
        """Build a deterministic shortest path through the allowed road graph."""
        if not self.roads:
            return (target,)

        blocked = set(blocked_road_ids)
        nodes = {}
        edges = {}

        for road in self.roads:
            if road.road_id in blocked:
                continue
            if vehicle and not road.vehicle_allowed:
                continue
            if not vehicle and not road.pedestrian_allowed:
                continue

            a_key = (road.start.x, road.start.y)
            b_key = (road.end.x, road.end.y)
            nodes[a_key] = road.start
            nodes[b_key] = road.end
            edges.setdefault(a_key, []).append((road.end, road.length))
            edges.setdefault(b_key, []).append((road.start, road.length))

        if not nodes:
            return (target,)

        start = min(nodes.values(), key=lambda p: self._distance(position, p))
        goal = min(nodes.values(), key=lambda p: self._distance(target, p))
        start_key = (start.x, start.y)
        goal_key = (goal.x, goal.y)

        # Dijkstra is deliberately used here instead of repeatedly selecting
        # the nearest graph node. Once a trip starts, this produces one
        # deterministic route through the road graph.
        distances = {start_key: 0.0}
        previous = {}
        unvisited = set(nodes)

        while unvisited:
            current = min(
                unvisited,
                key=lambda key: distances.get(key, float("inf")),
            )
            if current not in distances:
                break
            unvisited.remove(current)

            if current == goal_key:
                path = [current]
                while path[-1] != start_key:
                    parent = previous.get(path[-1])
                    if parent is None:
                        return (target,)
                    path.append(parent)
                path.reverse()
                return tuple(nodes[key] for key in path[1:])

            for neighbour, cost in edges.get(current, ()):
                neighbour_key = (neighbour.x, neighbour.y)
                new_distance = distances[current] + cost
                if new_distance < distances.get(neighbour_key, float("inf")):
                    distances[neighbour_key] = new_distance
                    previous[neighbour_key] = current

        return (target,)

    def _road_waypoint(
        self,
        position: Vector2,
        target: Vector2,
        agent_id: str = "",
        blocked_road_ids: tuple[str, ...] = (),
        vehicle: bool = False,
    ) -> Vector2:
        """Return the next waypoint from a committed road route."""
        if not self.roads:
            return target

        cache_key = (
            (target.x, target.y),
            tuple(sorted(blocked_road_ids)),
            vehicle,
        )
        cached = self._route_cache.get(agent_id)
        if cached is None or cached[:3] != cache_key:
            route = self._road_route(
                position,
                target,
                blocked_road_ids,
                vehicle,
            )
            cached = (
                cache_key[0],
                cache_key[1],
                cache_key[2],
                (position.x, position.y),
                route,
            )
            self._route_cache[agent_id] = cached

        route = cached[4]
        if not route:
            return target

        remaining = list(route)
        while len(remaining) > 1 and self._distance(position, remaining[0]) <= self.arrival_radius:
            remaining.pop(0)

        if not remaining:
            self._route_cache.pop(agent_id, None)
            return target

        self._route_cache[agent_id] = (
            cached[0],
            cached[1],
            cached[2],
            cached[3],
            tuple(remaining),
        )
        waypoint = remaining[0]

        if len(remaining) == 1 and self._distance(position, waypoint) <= self.arrival_radius:
            self._route_cache.pop(agent_id, None)
            return target

        return waypoint

    def request_cognitive_update(self, context: AgentContext) -> StateDelta:
        self.call_count += 1

        # Unexpected events take priority over the normal routine.
        participating_events = [
            event for event in context.active_events if event.is_participant
        ]
        if participating_events:
            event = min(participating_events, key=lambda item: item.distance)
            if event.distance <= self.arrival_radius:
                self._route_cache.pop(context.agent_id, None)
                return StateDelta(
                    action=Action(
                        action_type=ActionType.INTERACT,
                        target_position=event.position,
                    ),
                )

            blocked = (
                context.closed_road_ids
                if context.transport_mode == "car"
                else ()
            )
            target = self._road_waypoint(
                context.position,
                event.position,
                context.agent_id,
                blocked,
                context.transport_mode == "car",
            )
            return StateDelta(
                activity=ActivityType.COMMUTE,
                travel_destination=event.position,
                action=Action(
                    action_type=ActionType.MOVE,
                    target_position=target,
                ),
            )

        # Persistent memories can bias the next day toward places and people
        # that mattered previously, approximating Smallville-style retrieval.
        remembered_contacts = {
            related_id
            for memory in context.memories
            if memory.kind in {"interaction", "reflection"}
            for related_id in memory.related_agent_ids
        }
        preferred_social_id = next(
            (
                item.agent_id
                for item in sorted(
                    context.nearby_agents,
                    key=lambda item: item.distance,
                )
                if item.agent_id in remembered_contacts
            ),
            None,
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
                        f"{context.simulation_tick // 720:04d}-"
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
            # into the next activity. The explicit travel destination is the
            # physical commitment; the action target may be a road waypoint.
            trip_destination = context.travel_destination

            if (
                trip_destination is not None
                and self._distance(context.position, trip_destination)
                > self.arrival_radius
            ):
                blocked = (
                    context.closed_road_ids
                    if context.transport_mode == "car"
                    else ()
                )
                # Keep following the current road waypoint until it is
                # reached. Recomputing the nearest graph node every tick can
                # make an agent flip between two junctions while it is
                # between them.
                current_target = (
                    context.current_action.target_position
                    if context.current_action is not None
                    and context.current_action.action_type == ActionType.MOVE
                    else None
                )
                if (
                    current_target is not None
                    and self._distance(context.position, current_target)
                    > self.arrival_radius
                    and self._distance(current_target, trip_destination)
                    < self._distance(context.position, trip_destination)
                ):
                    target = current_target
                else:
                    target = self._road_waypoint(
                        context.position,
                        trip_destination,
                        context.agent_id,
                        blocked,
                        context.transport_mode == "car",
                    )
                trip_goal = type(goal)(
                    goal_id=(
                        f"{context.agent_id}-"
                        f"{context.simulation_tick // 720:04d}-trip"
                    ),
                    description="Continue current trip",
                    target_position=trip_destination,
                    importance=goal.importance,
                )
                return StateDelta(
                    goal=trip_goal,
                    activity=ActivityType.COMMUTE,
                    travel_destination=trip_destination,
                    action=Action(
                        action_type=ActionType.MOVE,
                        target_position=target,
                    ),
                )

            if activity.value == "social" and context.nearby_agents:
                nearby = next(
                    (
                        item
                        for item in context.nearby_agents
                        if item.agent_id == preferred_social_id
                    ),
                    None,
                )
                if nearby is None:
                    nearby = min(
                        (
                            item
                            for item in context.nearby_agents
                            if item.agent_id in context.social_connections
                        ),
                        key=lambda item: item.distance,
                        default=None,
                    )
                if nearby is not None and nearby.distance <= self.social_radius:
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
                self._route_cache.pop(context.agent_id, None)
                return StateDelta(
                    goal=goal,
                    activity=activity,
                    action=Action(
                        action_type=ActionType.WAIT,
                        target_position=routine.target_position,
                    ),
                )

            current_target = (
                context.current_action.target_position
                if context.current_action is not None
                and context.current_action.action_type == ActionType.MOVE
                else None
            )
            if (
                current_target is not None
                and self._distance(context.position, current_target)
                    > self.arrival_radius
                and self._distance(current_target, routine.target_position)
                    < self._distance(context.position, routine.target_position)
            ):
                target = current_target
            else:
                target = self._road_waypoint(
                    context.position,
                    routine.target_position,
                    context.agent_id,
                    context.closed_road_ids
                    if context.transport_mode == "car"
                    else (),
                    context.transport_mode == "car",
                )
            return StateDelta(
                goal=goal,
                activity=activity,
                travel_destination=routine.target_position,
                action=Action(
                    action_type=ActionType.MOVE,
                    target_position=target,
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
