"""Tests for LLM-backed daily memory reflection."""

from argus.llm import (
    DeterministicMemoryProvider,
    MemoryCandidate,
    MemoryManager,
)
from argus.scheduling import BaselineScheduler
from argus.simulation import Simulation


class FakeMemoryProvider:
    def __init__(self):
        self.calls = 0

    def summarize_day(self, agent, memories):
        self.calls += 1
        return (
            MemoryCandidate(
                summary="Arun had a useful conversation with Maya.",
                importance=0.8,
                related_agent_ids=("agent-0002",),
            ),
        )


def test_deterministic_memory_fallback_returns_candidate() -> None:
    simulation = Simulation.create(agent_count=2, seed=42)
    agent = simulation.agents["agent-0001"]

    simulation.add_memory(
        "agent-0001",
        kind="conversation",
        summary="Talked with Maya about work.",
        importance=0.8,
        related_agent_ids=("agent-0002",),
    )

    provider = DeterministicMemoryProvider()
    result = provider.summarize_day(agent, tuple(agent.memories))

    assert len(result) == 1
    assert "Maya" in result[0].summary
    assert result[0].related_agent_ids == ("agent-0002",)


def test_scheduler_invokes_memory_manager_once_per_day() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    simulation.add_memory(
        agent.agent_id,
        kind="experience",
        summary="Spent time at leisure during day 0.",
        importance=0.45,
    )

    provider = FakeMemoryProvider()
    manager = MemoryManager(provider)
    scheduler = BaselineScheduler(
        simulation,
        __import__("argus.llm", fromlist=["MockLLMProvider"]).MockLLMProvider(),
        memory_manager=manager,
    )

    scheduler.run(ticks=720)

    assert provider.calls == 1
    reflections = [
        memory
        for memory in agent.memories
        if memory.kind == "reflection"
    ]
    assert len(reflections) == 1


class FailingMemoryProvider:
    def summarize_day(self, agent, memories):
        raise RuntimeError("Ollama unavailable")


def test_memory_manager_uses_fallback_on_provider_failure() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    simulation.add_memory(
        agent.agent_id,
        kind="experience",
        summary="Visited the village café.",
        importance=0.6,
    )

    manager = MemoryManager(
        FailingMemoryProvider(),
        DeterministicMemoryProvider(),
    )
    result = manager.summarize_day(agent, tuple(agent.memories))

    assert len(result) == 1
    assert "village café" in result[0].summary.lower()
