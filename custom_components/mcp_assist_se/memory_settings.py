"""Shared memory and personality settings.

Missing keys keep their defaults, so existing installs pick up the new
behaviour without a config migration and without touching stored memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .const import (
    CONF_EMBEDDING_BASE_URL,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_PROVIDER,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_PERSONALITY,
    CONF_LOCAL_EMBEDDING_MODEL,
    CONF_MEMORY_AUTO_RETRIEVE,
    CONF_MEMORY_DEDUPE_THRESHOLD,
    CONF_MEMORY_FORGET_THRESHOLD,
    CONF_MEMORY_INCLUDE_LAST_TURN,
    CONF_MEMORY_SIMILARITY_THRESHOLD,
    CONF_MEMORY_TOP_K,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENABLE_MEMORY_TOOLS,
    DEFAULT_ENABLE_PERSONALITY,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_MEMORY_AUTO_RETRIEVE,
    DEFAULT_MEMORY_DEDUPE_THRESHOLD,
    DEFAULT_MEMORY_FORGET_THRESHOLD,
    DEFAULT_MEMORY_INCLUDE_LAST_TURN,
    DEFAULT_MEMORY_SIMILARITY_THRESHOLD,
    DEFAULT_MEMORY_TOP_K,
    EMBEDDING_PROVIDER_LOCAL,
    EMBEDDING_PROVIDER_OPENAI,
)
from .local_embedder import canonical_local_model, local_model_spec

_TRUE_STRINGS = {"1", "true", "yes", "on"}


def _clamped_float(value: Any, default: float) -> float:
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def as_bool(value: Any, default: bool) -> bool:
    """Coerce a stored config value to bool."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().casefold() in _TRUE_STRINGS
    return bool(value)


def normalize_memory_options(user_input: dict[str, Any] | None) -> dict[str, Any]:
    """Return stored memory/personality options with defaults and bounds applied."""
    source = user_input or {}
    provider = str(source.get(CONF_EMBEDDING_PROVIDER) or DEFAULT_EMBEDDING_PROVIDER).strip()
    if provider not in (EMBEDDING_PROVIDER_LOCAL, EMBEDDING_PROVIDER_OPENAI):
        provider = DEFAULT_EMBEDDING_PROVIDER

    try:
        top_k = int(source.get(CONF_MEMORY_TOP_K, DEFAULT_MEMORY_TOP_K))
    except (TypeError, ValueError):
        top_k = DEFAULT_MEMORY_TOP_K
    top_k = max(1, min(8, top_k))

    try:
        threshold = float(
            source.get(CONF_MEMORY_SIMILARITY_THRESHOLD, DEFAULT_MEMORY_SIMILARITY_THRESHOLD)
        )
    except (TypeError, ValueError):
        threshold = DEFAULT_MEMORY_SIMILARITY_THRESHOLD
    threshold = max(0.0, min(1.0, threshold))
    dedupe_threshold = _clamped_float(
        source.get(CONF_MEMORY_DEDUPE_THRESHOLD), DEFAULT_MEMORY_DEDUPE_THRESHOLD
    )
    forget_threshold = _clamped_float(
        source.get(CONF_MEMORY_FORGET_THRESHOLD), DEFAULT_MEMORY_FORGET_THRESHOLD
    )

    openai_model = " ".join(
        str(source.get(CONF_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL).split()
    ).strip() or DEFAULT_EMBEDDING_MODEL
    base_url = str(source.get(CONF_EMBEDDING_BASE_URL) or DEFAULT_EMBEDDING_BASE_URL).strip().rstrip("/")

    return {
        CONF_ENABLE_PERSONALITY: as_bool(
            source.get(CONF_ENABLE_PERSONALITY), DEFAULT_ENABLE_PERSONALITY
        ),
        CONF_MEMORY_AUTO_RETRIEVE: as_bool(
            source.get(CONF_MEMORY_AUTO_RETRIEVE), DEFAULT_MEMORY_AUTO_RETRIEVE
        ),
        CONF_MEMORY_TOP_K: top_k,
        CONF_MEMORY_SIMILARITY_THRESHOLD: threshold,
        CONF_MEMORY_DEDUPE_THRESHOLD: dedupe_threshold,
        CONF_MEMORY_FORGET_THRESHOLD: forget_threshold,
        CONF_MEMORY_INCLUDE_LAST_TURN: as_bool(
            source.get(CONF_MEMORY_INCLUDE_LAST_TURN), DEFAULT_MEMORY_INCLUDE_LAST_TURN
        ),
        CONF_EMBEDDING_PROVIDER: provider,
        CONF_LOCAL_EMBEDDING_MODEL: canonical_local_model(
            source.get(CONF_LOCAL_EMBEDDING_MODEL) or DEFAULT_LOCAL_EMBEDDING_MODEL
        ),
        CONF_EMBEDDING_MODEL: openai_model,
        CONF_EMBEDDING_BASE_URL: base_url,
    }


@dataclass(frozen=True)
class MemorySettings:
    """Effective memory and personality settings for one turn."""

    memory_tools: bool
    personality: bool
    auto_retrieve: bool
    top_k: int
    threshold: float
    dedupe_threshold: float
    forget_threshold: float
    include_last_turn: bool
    provider: str
    local_model: str
    openai_model: str
    openai_base_url: str

    @property
    def cache_key(self) -> str:
        """Vector cache key. Changing model or provider misses the cache on purpose."""
        if self.provider == EMBEDDING_PROVIDER_OPENAI:
            return f"openai:{self.openai_model}"
        return f"local:{self.local_model}"


def read_memory_settings(hass) -> MemorySettings:
    """Read settings from the shared system entry, then defaults."""
    from . import get_system_entry

    system_entry = get_system_entry(hass)
    source: dict[str, Any] = {}
    if system_entry is not None:
        source.update(system_entry.data)
        source.update(system_entry.options)
    else:
        from .const import DOMAIN

        for entry in hass.config_entries.async_entries(DOMAIN):
            source.update(entry.data)
            source.update(entry.options)
            break

    normalized = normalize_memory_options(source)
    spec = local_model_spec(normalized[CONF_LOCAL_EMBEDDING_MODEL])
    return MemorySettings(
        memory_tools=as_bool(source.get(CONF_ENABLE_MEMORY_TOOLS), DEFAULT_ENABLE_MEMORY_TOOLS),
        personality=normalized[CONF_ENABLE_PERSONALITY],
        auto_retrieve=normalized[CONF_MEMORY_AUTO_RETRIEVE],
        top_k=normalized[CONF_MEMORY_TOP_K],
        threshold=normalized[CONF_MEMORY_SIMILARITY_THRESHOLD],
        dedupe_threshold=normalized[CONF_MEMORY_DEDUPE_THRESHOLD],
        forget_threshold=normalized[CONF_MEMORY_FORGET_THRESHOLD],
        include_last_turn=normalized[CONF_MEMORY_INCLUDE_LAST_TURN],
        provider=normalized[CONF_EMBEDDING_PROVIDER],
        local_model=spec.model_id,
        openai_model=normalized[CONF_EMBEDDING_MODEL],
        openai_base_url=normalized[CONF_EMBEDDING_BASE_URL],
    )
