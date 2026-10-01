"""Lightweight episodic memory retrieval for ARGUS."""

from __future__ import annotations

import re
from dataclasses import dataclass

from argus.simulation.agent import MemoryRecord


@dataclass(frozen=True, slots=True)
class MemoryRetriever:
    """Rank stored memories without requiring an embedding model or database."""

    limit: int = 8
    recency_weight: float = 0.35
    importance_weight: float = 0.30
    relevance_weight: float = 0.25
    relationship_weight: float = 0.10

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    def retrieve(
        self,
        memories: tuple[MemoryRecord, ...] | list[MemoryRecord],
        *,
        current_tick: int,
        query: str = "",
        related_agent_ids: tuple[str, ...] = (),
    ) -> tuple[MemoryRecord, ...]:
        if not memories:
            return ()

        query_tokens = self._tokens(query)
        related = set(related_agent_ids)
        max_tick = max(memory.tick for memory in memories)

        scored: list[tuple[float, int, MemoryRecord]] = []
        for index, memory in enumerate(memories):
            age = max(0, max_tick - memory.tick)
            recency = 1.0 / (1.0 + age / 720.0)
            overlap = 0.0
            if query_tokens:
                memory_tokens = self._tokens(memory.summary)
                overlap = len(query_tokens & memory_tokens) / len(query_tokens)

            relationship = (
                1.0
                if related and related.intersection(memory.related_agent_ids)
                else 0.0
            )

            score = (
                self.recency_weight * recency
                + self.importance_weight * memory.importance
                + self.relevance_weight * overlap
                + self.relationship_weight * relationship
            )
            scored.append((score, index, memory))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [item[2] for item in scored[: self.limit]]

        # Preserve chronological order in the context so the LLM sees a
        # coherent sequence rather than a relevance-ranked list.
        selected.sort(key=lambda memory: (memory.tick, memory.memory_id))
        return tuple(selected)
