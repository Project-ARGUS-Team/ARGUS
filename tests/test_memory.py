"""Tests for the memory cognition layer."""

from argus.llm.memory import MemoryCandidate, MemoryManager
from argus.simulation import Simulation


class FakeMemoryProvider:
    def summarize_day(self, agent, memories):
        return (
            MemoryCandidate(
                summary="Arun had a useful conversation with Maya.",
                importance=0.8,
                related_agent_ids=("agent-0002",),
            ),
        )


def test_memory_manager_preserves_provider_output() -> None:
    simulation = Simulation.create(agent_count=2, seed=42)
    manager = MemoryManager(FakeMemoryProvider())
    agent = simulation.agents["agent-0001"]

    simulation.add_memory(
        "agent-0001",
        kind="conversation",
        summary="Talked with Maya about work.",
        importance=0.8,
        related_agent_ids=("agent-0002",),
    )

    result = manager.summarize_day(
        agent,
        tuple(agent.memories),
    )

    assert len(result) == 1
    assert result[0].summary == "Arun had a useful conversation with Maya."
    assert result[0].related_agent_ids == ("agent-0002",)
