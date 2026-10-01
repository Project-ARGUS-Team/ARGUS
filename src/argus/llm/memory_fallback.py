"""Deterministic fallback memory cognition."""

from argus.simulation.agent import AgentState, MemoryRecord
from argus.llm.memory import MemoryCandidate, MemoryCognitionProvider


class DeterministicMemoryProvider(MemoryCognitionProvider):
    """Summarize important daily experiences without an external model."""

    def summarize_day(
        self,
        agent: AgentState,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryCandidate, ...]:
        if not memories:
            return ()

        highlights = sorted(
            memories,
            key=lambda memory: memory.importance,
            reverse=True,
        )[:3]
        summary = "; ".join(memory.summary for memory in highlights)
        related = tuple(
            sorted(
                {
                    related_id
                    for memory in highlights
                    for related_id in memory.related_agent_ids
                }
            )
        )
        importance = min(
            1.0,
            sum(memory.importance for memory in highlights) / len(highlights),
        )
        return (
            MemoryCandidate(
                summary=summary[:200],
                importance=importance,
                related_agent_ids=related,
            ),
        )
