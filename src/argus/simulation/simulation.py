"""Simulation state and deterministic simulation loop."""

from dataclasses import dataclass
import random

from argus.simulation.agent import ActionType, AgentState, Goal, Vector2
from argus.simulation.context import AgentContext, AgentObservation, EventObservation, StateDelta
from argus.simulation.events import WorldEvent
from argus.simulation.world import World


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

    def __init__(self, state: SimulationState, world: World) -> None:
        self.state = state
        self.world = world

    @classmethod
    def create(
        cls,
        agent_count: int = 0,
        seed: int = 42,
        world: World | None = None,
    ) -> "Simulation":
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

        return cls(state=SimulationState(agents=agents), world=world)

    @property
    def current_tick(self) -> int:
        return self.state.tick

    @property
    def simulation_time(self) -> float:
        return self.state.simulation_time

    @property
    def agents(self) -> dict[str, AgentState]:
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
            nearby_agents=tuple(nearby_agents),
            active_events=tuple(active_events),
            social_connections=tuple(sorted(agent.social_connections)),
        )

    def apply_state_delta(self, agent_id: str, delta: StateDelta) -> None:
        """Apply a cognitive result to authoritative agent state."""
        agent = self.state.agents[agent_id]

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

            speed = 1.0
            agent.velocity = Vector2(
                x=(dx / distance) * speed,
                y=(dy / distance) * speed,
            )
        else:
            agent.velocity = Vector2(0.0, 0.0)

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
