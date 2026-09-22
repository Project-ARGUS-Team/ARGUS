"""Uniform full-frequency baseline scheduler."""

from __future__ import annotations

from argus.llm.gateway import LLMGateway
from argus.simulation.simulation import Simulation


class BaselineScheduler:
    """Run the Stage 1 uniform full-frequency cognitive baseline.

    Every active agent receives exactly one cognitive update before the
    simulation advances by one tick. No relevance or adaptive scheduling
    logic is performed here.
    """

    def __init__(self, simulation: Simulation, gateway: LLMGateway) -> None:
        self.simulation = simulation
        self.gateway = gateway
        self.total_cognitive_updates = 0

    @property
    def current_tick(self) -> int:
        """Return the simulation tick managed by this scheduler."""
        return self.simulation.current_tick

    def step(self) -> int:
        """Perform one baseline scheduling step and return update count."""
        updates = 0

        for agent in self.simulation.agents.values():
            if not agent.active:
                continue

            self.simulation.request_cognitive_update(agent.agent_id, self.gateway)
            updates += 1

        self.simulation.tick()
        self.total_cognitive_updates += updates
        return updates

    def run(self, ticks: int) -> int:
        """Run the baseline for the requested number of ticks."""
        if ticks < 0:
            raise ValueError("ticks must be non-negative")

        updates = 0
        for _ in range(ticks):
            updates += self.step()

        return updates
