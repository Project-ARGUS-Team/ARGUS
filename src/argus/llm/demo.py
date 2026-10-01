"""Deterministic cognitive provider for the baseline demonstration."""

from __future__ import annotations

import math
import random

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


    def _daily_seed(self, context: AgentContext, salt: int = 0) -> int:
        """Create a stable pseudo-random seed for this agent and day."""
        index = self._agent_index(context.agent_id) + 1
        day = context.simulation_tick // 720
        return (index * 73856093 + day * 19349663 + salt * 83492791) & 0xFFFFFFFF

    def _is_day_off(self, context: AgentContext) -> bool:
        """Return whether this agent takes an occasional deterministic day off."""
        return random.Random(self._daily_seed(context, 1)).random() < 0.18

    def _sleep_start_tick(self, context: AgentContext) -> int:
        """Choose a stable late-night bedtime for this agent and day."""
        rng = random.Random(self._daily_seed(context, 17))
        return 480 + int(rng.random() * 61)

    def _is_sleeping(self, context: AgentContext) -> bool:
        """Return whether the agent is currently in its nightly sleep window."""
        local_tick = context.simulation_tick % 720
        sleep_start = self._sleep_start_tick(context)
        return local_tick >= sleep_start or local_tick < 45

    def _leisure_target(self, context: AgentContext, salt: int = 0) -> Vector2:
        """Choose a leisure destination from the agent's personal options."""
        if context.profile is None:
            return context.position
        options = context.profile.leisure_options
        if not options:
            options = tuple(
                dict.fromkeys(
                    entry.target_position
                    for entry in context.profile.routine
                    if entry.activity in {
                        ActivityType.LEISURE,
                        ActivityType.SOCIAL,
                        ActivityType.EAT,
                        ActivityType.SHOP,
                    }
                )
            )
        if not options and context.profile.leisure_position is not None:
            options = (context.profile.leisure_position,)
        if not options:
            return context.position
        return random.Random(self._daily_seed(context, 31 + salt)).choice(options)

    def _day_off_activity(self, context: AgentContext) -> tuple[ActivityType, str, Vector2, float]:
        """Choose an activity and destination for a day off."""
        local_tick = context.simulation_tick % 720
        if local_tick < 45:
            return ActivityType.HOME, "Morning at home", context.profile.home_position, 0.45
        if local_tick < 180:
            return ActivityType.LEISURE, "Day off outing", self._leisure_target(context), 0.65
        if local_tick < 225:
            return ActivityType.EAT, "Lunch outing", self._leisure_target(context, 1), 0.55
        if local_tick < 360:
            return ActivityType.LEISURE, "Free afternoon", self._leisure_target(context, 2), 0.65
        if local_tick < 480:
            return ActivityType.SOCIAL, "Spending time with people", self._leisure_target(context, 3), 0.70
        return ActivityType.HOME, "Evening at home", context.profile.home_position, 0.40

    def _nearby_interaction(self, context: AgentContext) -> StateDelta | None:
        """Occasionally turn a nearby encounter into a short conversation."""
        if context.profile is None:
            return None
        candidates = [
            item for item in context.nearby_agents
            if item.distance <= self.social_radius and not item.asleep
        ]
        if not candidates:
            return None

        rng = random.Random(self._daily_seed(context, context.simulation_tick + 101))
        nearby = rng.choice(candidates)
        relationship = dict(context.relationship_strengths).get(nearby.agent_id, 0.5)
        proximity = max(0.0, 1.0 - nearby.distance / self.social_radius)
        # Encounters are deliberately rare. An agent can pass another person
        # many times without starting a conversation; proximity only nudges
        # the probability upward.
        chance = 0.0008 + proximity * 0.0035
        chance *= 0.75 + context.profile.social_preference * 0.50
        remembered_contacts = {
            related_id
            for memory in context.memories
            if memory.kind in {"conversation", "reflection"}
            for related_id in memory.related_agent_ids
        }
        if nearby.agent_id in remembered_contacts:
            chance *= 1.75

        if rng.random() >= min(0.012, chance):
            return None

        return StateDelta(
            activity=ActivityType.SOCIAL,
            action=Action(
                action_type=ActionType.INTERACT,
                target_agent_id=nearby.agent_id,
                target_position=nearby.position,
                duration=1.0,
            ),
        )

    def request_cognitive_update(self, context: AgentContext) -> StateDelta:
        self.call_count += 1

        # Sleep is a hard state: no social interaction, events, or routine
        # activity can interrupt the nightly rest period.
        if self._is_sleeping(context) and context.profile is not None:
            self._route_cache.pop(context.agent_id, None)
            goal = type(context.goal)(
                goal_id=f"{context.agent_id}-{context.simulation_tick // 720:04d}-sleep",
                description="Sleeping",
                target_position=context.profile.home_position,
                importance=0.35,
            )
            return StateDelta(
                goal=goal,
                activity=ActivityType.SLEEP,
                action=Action(
                    action_type=ActionType.WAIT,
                    target_position=context.profile.home_position,
                ),
            )

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

        interaction = self._nearby_interaction(context)
        if interaction is not None:
            return interaction

        # Retrieved episodic memories are already considered by the social
        # interaction model above; routine behavior remains otherwise unchanged.

        routine = context.current_routine

        if routine is not None:
            if self._is_day_off(context) and routine.activity in {
                ActivityType.COMMUTE,
                ActivityType.WORK,
                ActivityType.STUDY,
                ActivityType.EAT,
            }:
                activity, description, target_position, importance = self._day_off_activity(context)
                goal = type(context.goal)(
                    goal_id=f"{context.agent_id}-{context.simulation_tick // 720:04d}-day-off",
                    description=description,
                    target_position=target_position,
                    importance=importance,
                )
                distance = self._distance(context.position, target_position)
                if distance <= self.arrival_radius:
                    self._route_cache.pop(context.agent_id, None)
                    return StateDelta(
                        goal=goal,
                        activity=activity,
                        action=Action(
                            action_type=ActionType.WAIT,
                            target_position=target_position,
                        ),
                    )
                target = self._road_waypoint(
                    context.position,
                    target_position,
                    context.agent_id,
                    context.closed_road_ids if context.transport_mode == "car" else (),
                    context.transport_mode == "car",
                )
                return StateDelta(
                    goal=goal,
                    activity=activity,
                    travel_destination=target_position,
                    action=Action(
                        action_type=ActionType.MOVE,
                        target_position=target,
                    ),
                )

            activity = routine.activity
            routine_target = routine.target_position
            if activity in {
                ActivityType.LEISURE,
                ActivityType.SOCIAL,
                ActivityType.EAT,
                ActivityType.SHOP,
            }:
                routine_target = self._leisure_target(
                    context,
                    70 + routine.start_tick,
                )

            goal = context.goal
            if (
                goal.target_position != routine_target
                or goal.description != routine.description
            ):
                goal = type(goal)(
                    goal_id=(
                        f"{context.agent_id}-"
                        f"{context.simulation_tick // 720:04d}-"
                        f"{routine.activity.value}"
                    ),
                    description=routine.description,
                    target_position=routine_target,
                    importance=routine.importance,
                )

            distance = self._distance(
                context.position,
                routine_target,
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
                # The committed route cache is the source of truth for
                # intermediate road waypoints. Do not preserve the previous
                # action target based on geometric closeness: at a junction,
                # that can keep an agent following the wrong branch.
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

            if distance <= self.arrival_radius:
                self._route_cache.pop(context.agent_id, None)
                return StateDelta(
                    goal=goal,
                    activity=activity,
                    action=Action(
                        action_type=ActionType.WAIT,
                        target_position=routine_target,
                    ),
                )

            target = self._road_waypoint(
                context.position,
                routine_target,
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
