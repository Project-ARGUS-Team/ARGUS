"""LLM-backed conversation generation for ARGUS interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import request

from argus.simulation.agent import AgentState


@dataclass(frozen=True, slots=True)
class ConversationCandidate:
    """One generated conversation topic and compact exchange."""

    topic: str
    summary: str


class ConversationCognitionProvider(Protocol):
    """Generate one short conversation when two agents interact."""

    def generate(
        self,
        speaker: AgentState,
        listener: AgentState,
        recent_topics: tuple[str, ...] = (),
    ) -> ConversationCandidate:
        """Generate a conversation summary."""


class OllamaConversationProvider:
    """Use local Ollama to generate an interpretable interaction summary."""

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

    def generate(
        self,
        speaker: AgentState,
        listener: AgentState,
        recent_topics: tuple[str, ...] = (),
    ) -> ConversationCandidate:
        self.call_count += 1
        speaker_name = speaker.profile.name if speaker.profile else speaker.agent_id
        listener_name = listener.profile.name if listener.profile else listener.agent_id
        speaker_job = speaker.profile.occupation if speaker.profile else "resident"
        listener_job = listener.profile.occupation if listener.profile else "resident"
        conversation_history = "\n".join(
            f"- {memory}" for memory in recent_topics
        ) if recent_topics else "none"

        prompt = f"""You are generating one brief, believable conversation for a simulated town.

Speaker: {speaker_name}, {speaker_job}
Listener: {listener_name}, {listener_job}

Previous conversation memories:
{conversation_history}

Return JSON only:
{{
  "topic": "short topic",
  "summary": "one short third-person sentence describing the actual information exchanged"
}}

Rules:
- Use only ordinary everyday topics: work, study, food, commute, weekend plans,
  hobbies, local news, family plans, errands, or the local community.
- Treat previous conversation memories as persistent context and build naturally on them.
- If a previous memory contains a plan, preference, question, or detail, reference it
  when it makes sense instead of starting from scratch.
- Avoid repeating the same topic unless the new conversation clearly follows up on it.
- The summary should describe one concrete piece of information, plan, preference,
  question, or follow-up from the conversation, not merely name the topic.
- Keep the summary as one short third-person sentence, under 25 words.
- Do not include dialogue or quotation marks.
- The conversation is simulated, so you may create plausible everyday details, but
  they must remain consistent with the agents' known profiles and previous memories.
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.4},
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

        parsed = json.loads(raw.get("response", "{}"))
        topic = str(parsed.get("topic", "everyday plans")).strip()
        summary = str(parsed.get("summary", "")).strip()

        if not summary:
            raise ValueError("Conversation provider returned no summary")

        return ConversationCandidate(
            topic=topic[:80],
            summary=summary[:200],
        )
