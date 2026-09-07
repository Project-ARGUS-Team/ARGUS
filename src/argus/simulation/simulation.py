"""Simulation state and deterministic simulation loop."""

from dataclasses import dataclass
import random

from argus.simulation.agent import AgentState, Goal, Vector2
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

        return cls(
            state=SimulationState(agents=agents),
            world=world,
        )

    @property
    def current_tick(self) -> int:
        """Return the current simulation tick."""
        return self.state.tick

    @property
    def simulation_time(self) -> float:
        """Return elapsed simulation time in seconds."""
        return self.state.simulation_time

    @property
    def agents(self) -> dict[str, AgentState]:
        """Return the simulation's agents."""
        return self.state.agents

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
