"""LLM gateway contracts and deterministic providers."""

from argus.llm.demo import ScenarioLLMProvider
from argus.llm.conversation_llm import ConversationCandidate, ConversationCognitionProvider, OllamaConversationProvider
from argus.llm.gateway import LLMGateway
from argus.llm.memory import (
    MemoryCandidate,
    MemoryCognitionProvider,
    MemoryManager,
    OllamaMemoryProvider,
)
from argus.llm.memory_fallback import DeterministicMemoryProvider
from argus.llm.mock import MockLLMProvider

__all__ = [
    "LLMGateway",
    "ConversationCandidate",
    "ConversationCognitionProvider",
    "OllamaConversationProvider",
    "MemoryCandidate",
    "MemoryCognitionProvider",
    "MemoryManager",
    "DeterministicMemoryProvider",
    "MockLLMProvider",
    "OllamaMemoryProvider",
    "ScenarioLLMProvider",
]
