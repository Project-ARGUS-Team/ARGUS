"""LLM gateway interfaces."""

from typing import Protocol

from argus.simulation.context import AgentContext, StateDelta


class LLMGateway(Protocol):
    """Interface through which the simulation requests cognitive updates."""

    def request_cognitive_update(self, context: AgentContext) -> StateDelta:
        """Produce a structured cognitive update for an agent."""
