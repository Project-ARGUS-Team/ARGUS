"""Tests for lightweight episodic memory retrieval."""

from argus.llm.memory_retrieval import MemoryRetriever
from argus.simulation import Simulation


def test_retriever_prefers_relevant_memories() -> None:
    simulation = Simulation.create(agent_count=2, seed=42)
    simulation.add_memory(
        "agent-0001",
        kind="experience",
        summary="Worked at the office all afternoon.",
        importance=0.4,
    )
    simulation.add_memory(
        "agent-0001",
        kind="conversation",
        summary="Talked with Maya about her weekend plans.",
        importance=0.8,
        related_agent_ids=("agent-0002",),
    )
    simulation.add_memory(
        "agent-0001",
        kind="experience",
        summary="Visited the village market.",
        importance=0.5,
    )

    memories = tuple(simulation.agents["agent-0001"].memories)
    result = MemoryRetriever(limit=1).retrieve(
        memories,
        current_tick=720,
        query="talk Maya weekend",
        related_agent_ids=("agent-0002",),
    )

    assert len(result) == 1
    assert result[0].kind == "conversation"
    assert "Maya" in result[0].summary


def test_retriever_can_surface_an_older_relevant_memory() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    for index in range(10):
        simulation.add_memory(
            "agent-0001",
            kind="experience",
            summary=f"Routine event {index}.",
            importance=0.3,
        )

    simulation.add_memory(
        "agent-0001",
        kind="conversation",
        summary="Maya told Arun about her weekend plans.",
        importance=0.8,
        related_agent_ids=("agent-0002",),
    )

    memories = tuple(simulation.agents["agent-0001"].memories)
    result = MemoryRetriever(limit=3).retrieve(
        memories,
        current_tick=720,
        query="Maya weekend plans",
        related_agent_ids=("agent-0002",),
    )

    assert any("Maya" in memory.summary for memory in result)


def test_retriever_returns_chronological_context() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    for summary in ("First event", "Second event", "Third event"):
        simulation.add_memory(
            "agent-0001",
            kind="experience",
            summary=summary,
            importance=0.5,
        )

    memories = tuple(simulation.agents["agent-0001"].memories)
    result = MemoryRetriever(limit=2).retrieve(
        memories,
        current_tick=720,
        query="event",
    )

    assert [memory.summary for memory in result] == ["Second event", "Third event"]
