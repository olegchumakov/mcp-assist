"""Shared memory, personality, and embedding objects for the integration."""

from __future__ import annotations

from .embedding_service import EmbeddingService
from .memory_manager import MemoryManager
from .personality_manager import PersonalityManager

_RUNTIME_KEY = "assist_runtime"


class AssistRuntime:
    """One set of stores for every conversation profile."""

    def __init__(self, hass) -> None:
        """Create the stores. Embedding weights stay unloaded until prepare."""
        self.memory = MemoryManager(hass)
        self.personality = PersonalityManager(hass)
        self.embeddings = EmbeddingService(hass, self.memory)


def get_assist_runtime(hass) -> AssistRuntime:
    """Return the process-wide runtime, creating it on first use."""
    from .const import DOMAIN

    domain = hass.data.setdefault(DOMAIN, {})
    runtime = domain.get(_RUNTIME_KEY)
    if runtime is None:
        runtime = AssistRuntime(hass)
        domain[_RUNTIME_KEY] = runtime
    return runtime
