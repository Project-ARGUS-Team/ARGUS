"""Simulation state and deterministic simulation loop."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING

from argus.simulation.agent import ActionType, AgentState, Goal, Vector2
from argus.simulation.context import (
    AgentContext,
    AgentObservation,
    EventObservation,
    StateDelta,
)
from argus.simulation.events import WorldEvent
from argus.simulation.world import RoadSegment, World

if TYPE_CHECKING:
    from argus.llm.gateway import LLMGateway


@dataclass(slots=True)
class SimulationState:
    """Authoritative state of a running simulation."""

    tick: int = 0
    simulation_time: float = 0.0
    agents: dict[str, AgentState] | None = None
    events: dict[str, WorldEvent] | None = None

    def __post_init__(self) -> None:
        if self.agents is None:
            self.agents = {}
        if self.events is None:
            self.events = {}


class Simulation:
    """Owns simulation progression and authoritative world state."""

    def __init__(
        self,
        state: SimulationState,
        world: World,
        roads: tuple[RoadSegment, ...] = (),
    ) -> None:
        self.state = state
        self.world = world
        self.roads = roads

    @classmethod
    def create(
        cls,
        agent_count: int = 0,
        seed: int = 42,
        world: World | None = None,
        roads: tuple[RoadSegment, ...] = (),
    ) -> Simulation:
        """Create a deterministic simulation populated with agents."""
        if agent_count < 0:
            raise ValueError("agent_count must be non-negative")

        world = world or World()
        rng = random.Random(seed)

        agents: dict[str, AgentState] = {}
        for index in range(agent_count):
            agent_id = f"agent-{index + 1:04d}"
            position = Vector2(
                x=rng.uniform(0.0, world.width),
                y=rng.uniform(0.0, world.height),
            )
            goal = Goal(
                goal_id=f"goal-{index + 1:04d}",
                description="Explore the simulation world",
                target_position=None,
                importance=0.5,
            )
            agents[agent_id] = AgentState(
                agent_id=agent_id,
                position=position,
                velocity=Vector2(0.0, 0.0),
                goal=goal,
            )

        return cls(
            state=SimulationState(agents=agents),
            world=world,
            roads=roads,
        )

    @property
    def current_tick(self) -> int:
        """Return the current simulation tick."""
        return self.state.tick

    @property
    def simulation_time(self) -> float:
        """Return the current simulation time."""
        return self.state.simulation_time

    @property
    def agents(self) -> dict[str, AgentState]:
        """Return authoritative agents."""
        return self.state.agents

    def build_agent_context(self, agent_id: str) -> AgentContext:
        """Build a read-only cognitive context for an agent."""
        agent = self.state.agents[agent_id]

        nearby_agents = []
        for other in self.state.agents.values():
            if other.agent_id == agent_id or not other.active:
                continue

            dx = other.position.x - agent.position.x
            dy = other.position.y - agent.position.y
            distance = (dx * dx + dy * dy) ** 0.5
            nearby_agents.append(
                AgentObservation(
                    agent_id=other.agent_id,
                    position=other.position,
                    distance=distance,
                )
            )

        current_routine = None
        if agent.profile is not None:
            current_routine = next(
                (
                    entry
                    for entry in agent.profile.routine
                    if entry.contains(self.current_tick, 720)
                ),
                None,
            )

        closed_road_ids = tuple(
            sorted(
                {
                    road_id
                    for event in self.state.events.values()
                    if event.is_active(self.current_tick)
                    and event.event_type == "road_closure"
                    for road_id in event.affected_road_ids
                }
            )
        )
        traffic_factor = self._traffic_factor(agent_id)

        active_events = []
        for event in self.state.events.values():
            if event.is_active(self.current_tick):
                dx = event.position.x - agent.position.x
                dy = event.position.y - agent.position.y
                distance = (dx * dx + dy * dy) ** 0.5
                active_events.append(
                    EventObservation(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        position=event.position,
                        distance=distance,
                        is_participant=agent_id in event.participants,
                        importance=event.importance,
                        affected_road_ids=event.affected_road_ids,
                    )
                )

        return AgentContext(
            agent_id=agent.agent_id,
            simulation_tick=self.current_tick,
            simulation_time=self.simulation_time,
            position=agent.position,
            velocity=agent.velocity,
            goal=agent.goal,
            current_action=agent.current_action,
            plan=tuple(agent.plan),
            profile=agent.profile,
            current_activity=agent.current_activity,
            transport_mode=agent.transport_mode.value,
            current_routine=current_routine,
            travel_destination=agent.travel_destination,
            closed_road_ids=closed_road_ids,
            traffic_factor=traffic_factor,
            nearby_agents=tuple(nearby_agents),
            active_events=tuple(active_events),
            social_connections=tuple(sorted(agent.social_connections)),
        )

    def apply_state_delta(self, agent_id: str, delta: StateDelta) -> None:
        """Apply a cognitive result to authoritative agent state."""
        agent = self.state.agents[agent_id]

        if delta.goal is not None:
            agent.goal = delta.goal
        if delta.activity is not None:
            agent.current_activity = delta.activity
        if delta.travel_destination is not None:
            agent.travel_destination = delta.travel_destination

        if delta.plan:
            agent.plan = list(delta.plan)

        if delta.action is None:
            return

        agent.set_action(delta.action)

        if delta.action.action_type == ActionType.MOVE:
            target = delta.action.target_position
            if target is None:
                agent.velocity = Vector2(0.0, 0.0)
                return

            dx = target.x - agent.position.x
            dy = target.y - agent.position.y
            distance = (dx * dx + dy * dy) ** 0.5

            if distance == 0.0:
                agent.velocity = Vector2(0.0, 0.0)
                return

            base_speed = 2.0 if agent.transport_mode.value == "walk" else 6.0
            speed = base_speed * self._traffic_factor(agent_id)
            agent.velocity = Vector2(
                x=(dx / distance) * speed,
                y=(dy / distance) * speed,
            )
        else:
            agent.velocity = Vector2(0.0, 0.0)
            agent.travel_destination = None

    def _traffic_factor(self, agent_id: str) -> float:
        """Estimate local traffic pressure for a moving agent."""
        agent = self.state.agents[agent_id]
        if agent.transport_mode.value != "car":
            return 1.0

        congestion = 0
        for other in self.state.agents.values():
            if other.agent_id == agent_id or not other.active:
                continue
            if other.transport_mode.value != "car":
                continue
            dx = other.position.x - agent.position.x
            dy = other.position.y - agent.position.y
            if dx * dx + dy * dy <= 12.0 * 12.0:
                congestion += 1

        factor = max(0.45, 1.0 - congestion * 0.08)
        for event in self.state.events.values():
            if not event.is_active(self.current_tick):
                continue
            dx = event.position.x - agent.position.x
            dy = event.position.y - agent.position.y
            distance = (dx * dx + dy * dy) ** 0.5
            if event.event_type == "market_rush" and distance <= 25.0:
                factor = max(0.45, factor - 0.20)
            elif event.event_type in {
                "public_gathering",
                "community_fair",
                "sports_event",
                "school_event",
            } and distance <= 20.0:
                factor = max(0.45, factor - 0.10)
            elif event.event_type == "minor_accident" and distance <= 15.0:
                factor = max(0.45, factor - 0.18)
            elif event.event_type == "medical_alert" and distance <= 12.0:
                factor = max(0.45, factor - 0.08)
        return factor

    def request_cognitive_update(
        self,
        agent_id: str,
        gateway: LLMGateway,
    ) -> StateDelta:
        """Request and apply one cognitive update for an agent."""
        context = self.build_agent_context(agent_id)
        delta = gateway.request_cognitive_update(context)
        self.apply_state_delta(agent_id, delta)
        return delta

    def tick(self) -> None:
        """Advance the simulation by exactly one tick."""
        for agent in self.state.agents.values():
            self.world.advance_agent(agent)

        self.state.tick += 1
        self.state.simulation_time += self.world.tick_duration

    def run(self, ticks: int) -> None:
        """Advance the simulation by the requested number of ticks."""
        if ticks < 0:
            raise ValueError("ticks must be non-negative")

        for _ in range(ticks):
            self.tick()
