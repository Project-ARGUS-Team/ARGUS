"""LLM-backed conversation generation for ARGUS interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
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
        timeout: float = 90.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.call_count = 0
        self.last_debug: dict[str, object] = {}

    def generate(
        self,
        speaker: AgentState,
        listener: AgentState,
        recent_topics: tuple[str, ...] = (),
    ) -> ConversationCandidate:
        speaker_name = speaker.profile.name if speaker.profile else speaker.agent_id
        listener_name = listener.profile.name if listener.profile else listener.agent_id
        speaker_job = speaker.profile.occupation if speaker.profile else "resident"
        listener_job = listener.profile.occupation if listener.profile else "resident"
        conversation_history = "\n".join(
            f"- {memory}" for memory in recent_topics
        ) if recent_topics else "none"

        for attempt in range(2):
            self.call_count += 1
            retry_instruction = (
                "\nIMPORTANT: Your previous attempt repeated an earlier conversation. "
                "Generate a genuinely new development and do not reuse its wording."
                if attempt == 1 else ""
            )
            prompt = f"""You are generating one brief, believable conversation for a simulated town.

Speaker: {speaker_name}, {speaker_job}
Listener: {listener_name}, {listener_job}

Previous conversation memories:
{conversation_history}

These memories are persistent facts, not text to summarize or repeat.
The new conversation must add something new. If a previous memory contains a plan,
preference, question, or detail, treat it as remembered background and advance it
with a new development, follow-up, or related detail.{retry_instruction}

Return JSON only:
{{
  "topic": "short topic",
  "summary": "one short third-person sentence describing only the NEW information exchanged"
}}

Rules:
- Use only ordinary everyday topics: work, study, food, commute, weekend plans,
  hobbies, local news, family plans, errands, or the local community.
- A follow-up is allowed, but the summary must describe what changed or was newly
  learned in this conversation, not restate the previous memory.
- Do not copy phrases or sentences from previous conversation memories.
- Avoid repeating the same topic unless the new conversation clearly advances it.
- The summary should contain one concrete new piece of information, plan, preference,
  question, decision, or follow-up.
- Keep the summary as one short third-person sentence, under 25 words.
- Do not include dialogue or quotation marks.
- The conversation is simulated, so plausible everyday details are allowed, but they
  must remain consistent with the agents' known profiles and previous memories.
"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.5},
            }
            self.last_debug = {
                "model": self.model,
                "speaker": speaker_name,
                "listener": listener_name,
                "attempt": attempt + 1,
                "recent_topics": recent_topics,
                "prompt": prompt,
            }
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                f"{self.base_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            print(f"[ARGUS CONVERSATION] speaker={speaker_name!r} listener={listener_name!r} model={self.model!r} attempt={attempt + 1} recent_topics={recent_topics!r} requesting Ollama...", flush=True)
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                self.last_debug["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[ARGUS CONVERSATION ERROR] {type(exc).__name__}: {exc}", flush=True)
                raise

            self.last_debug["raw_response"] = raw.get("response", "")
            print(f"[ARGUS CONVERSATION] Ollama response={raw.get('response', '')!r}", flush=True)
            parsed = json.loads(raw.get("response", "{}"))
            topic = str(parsed.get("topic", "everyday plans")).strip()
            summary = str(parsed.get("summary", "")).strip()

            self.last_debug["topic"] = topic
            self.last_debug["summary"] = summary

            if not summary:
                self.last_debug["validation"] = "empty_summary"
                raise ValueError("Conversation provider returned no summary")

            normalized = " ".join(summary.lower().split())
            duplicate = any(
                normalized == " ".join(previous.lower().split())
                or SequenceMatcher(
                    None,
                    normalized,
                    " ".join(previous.lower().split()),
                ).ratio() >= 0.82
                or normalized in " ".join(previous.lower().split())
                or " ".join(previous.lower().split()) in normalized
                for previous in recent_topics
            )
            self.last_debug["duplicate"] = duplicate
            if not duplicate or attempt == 1:
                self.last_debug["validation"] = "accepted"
                print(f"[ARGUS CONVERSATION] accepted topic={topic!r} summary={summary!r} duplicate={duplicate}", flush=True)
                return ConversationCandidate(
                    topic=topic[:80],
                    summary=summary[:200],
                )

        self.last_debug["validation"] = "duplicate_after_retry"
        raise ValueError("Conversation provider could not generate a new exchange")
