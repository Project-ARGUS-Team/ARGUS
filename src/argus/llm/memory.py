"""Local Ollama-backed memory cognition for ARGUS."""

from __future__ import annotations

import json
from urllib import request
from dataclasses import dataclass
from typing import Protocol

from argus.simulation.agent import AgentState, MemoryRecord


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """One memory proposed by a memory cognition provider."""

    summary: str
    importance: float = 0.5
    related_agent_ids: tuple[str, ...] = ()


class MemoryCognitionProvider(Protocol):
    """Interface for turning agent experiences into persistent memories."""

    def summarize_day(
        self,
        agent: AgentState,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryCandidate, ...]:
        """Return a small set of durable memories for the agent."""


class OllamaMemoryProvider:
    """Use a local Ollama model to create structured episodic memories.

    The provider is deliberately separate from the main LLM gateway: Qwen
    cannot directly change authoritative simulation state.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.call_count = 0

    def summarize_day(
        self,
        agent: AgentState,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryCandidate, ...]:
        if not memories:
            return ()

        self.call_count += 1
        profile = agent.profile
        identity = (
            f"Name: {profile.name}; occupation: {profile.occupation}"
            if profile is not None
            else f"Agent ID: {agent.agent_id}"
        )
        experience_lines = "\n".join(
            f"- [{memory.kind}] {memory.summary}"
            for memory in memories[-20:]
        )

        prompt = f"""You are the memory system of a simulated person.

{identity}

Review the person's experiences from one simulation day. Extract only durable,
future-useful episodic memories. Do not invent events, people, emotions, or
facts that are not supported by the experiences.

Return JSON only in this exact shape:
{{
  "memories": [
    {{
      "summary": "short first-person or third-person memory",
      "importance": 0.0,
      "related_agent_ids": ["agent-0001"]
    }}
  ]
}}

Rules:
- Return 0 to 3 memories.
- Keep each summary under 25 words.
- importance must be between 0 and 1.
- related_agent_ids must contain only IDs appearing in the experiences.
- Prefer socially meaningful events, repeated experiences, unusual events, or
  information likely to matter on a later day.
- Do not simply copy every experience.

Experiences:
{experience_lines}
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))

        content = raw.get("response", "")
        parsed = json.loads(content)
        candidates = []
        for item in parsed.get("memories", []):
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "")).strip()
            if not summary:
                continue
            importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
            related = tuple(
                agent_id
                for agent_id in item.get("related_agent_ids", [])
                if isinstance(agent_id, str)
                and agent_id in agent.relationships
            )
            candidates.append(
                MemoryCandidate(
                    summary=summary,
                    importance=importance,
                    related_agent_ids=related,
                )
            )
        return tuple(candidates[:3])


class MemoryManager:
    """Apply model-generated memories while keeping the simulation authoritative."""

    def __init__(self, provider: MemoryCognitionProvider) -> None:
        self.provider = provider

    def summarize_day(
        self,
        agent: AgentState,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryCandidate, ...]:
        return self.provider.summarize_day(agent, memories)
