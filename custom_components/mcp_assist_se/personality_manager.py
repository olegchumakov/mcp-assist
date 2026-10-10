"""Persisted speaking-style instructions injected into every prompt."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .memory_rank import (
    MAX_PERSONALITY_INSTRUCTIONS,
    MAX_PERSONALITY_TOTAL_CHARS,
    format_personality_block,
    normalize_text,
    personality_char_count,
    validate_personality_text,
)

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}_personality"


class PersonalityManager:
    """Store a short list of behaviour instructions."""

    def __init__(self, hass) -> None:
        """Initialize the personality store."""
        self.hass = hass
        self._store: Store[dict[str, Any]] = Store(hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._lock = asyncio.Lock()
        self._loaded = False
        self._instructions: list[dict[str, Any]] = []

    async def async_initialize(self) -> None:
        """Load persisted instructions."""
        async with self._lock:
            await self._ensure_loaded_locked()

    async def async_view(self) -> list[dict[str, Any]]:
        """Return instructions, oldest first."""
        async with self._lock:
            await self._ensure_loaded_locked()
            return [dict(item) for item in self._instructions]

    async def async_prompt_block(self) -> str:
        """Personality section for the system prompt."""
        items = await self.async_view()
        return format_personality_block(items)

    async def async_set(
        self,
        text: str,
        *,
        instruction_id: str | None = None,
    ) -> dict[str, Any]:
        """Add an instruction, or replace one when ``instruction_id`` is set."""
        error = validate_personality_text(text)
        if error:
            return {"ok": False, "message": error, "instructions": []}
        normalized = normalize_text(text) or ""
        target_id = normalize_text(instruction_id)

        async with self._lock:
            await self._ensure_loaded_locked()
            if target_id:
                for item in self._instructions:
                    if item["id"] != target_id:
                        continue
                    others = [entry for entry in self._instructions if entry["id"] != target_id]
                    if personality_char_count(others) + len(normalized) > MAX_PERSONALITY_TOTAL_CHARS:
                        return {
                            "ok": False,
                            "message": (
                                "That would exceed the "
                                f"{MAX_PERSONALITY_TOTAL_CHARS} character personality block. "
                                "Shorten it or remove another instruction."
                            ),
                            "instructions": [dict(entry) for entry in self._instructions],
                        }
                    item["text"] = normalized
                    item["updated_at"] = dt_util.utcnow().isoformat()
                    await self._save_locked()
                    return {
                        "ok": True,
                        "message": (
                            f"Replaced personality instruction [{item['id']}]. "
                            "Follow it in this reply and on later turns."
                        ),
                        "instruction": dict(item),
                        "instructions": [dict(entry) for entry in self._instructions],
                    }
                return {
                    "ok": False,
                    "message": f"No personality instruction with id {target_id}.",
                    "instructions": [dict(entry) for entry in self._instructions],
                }

            if len(self._instructions) >= MAX_PERSONALITY_INSTRUCTIONS:
                return {
                    "ok": False,
                    "message": (
                        f"Personality block is full ({MAX_PERSONALITY_INSTRUCTIONS}). "
                        "Replace or remove one before adding another."
                    ),
                    "instructions": [dict(entry) for entry in self._instructions],
                }
            if personality_char_count(self._instructions) + len(normalized) > MAX_PERSONALITY_TOTAL_CHARS:
                return {
                    "ok": False,
                    "message": (
                        "That would exceed the "
                        f"{MAX_PERSONALITY_TOTAL_CHARS} character personality block. "
                        "Shorten it or remove another instruction."
                    ),
                    "instructions": [dict(entry) for entry in self._instructions],
                }

            for item in self._instructions:
                if item["text"].casefold() == normalized.casefold():
                    return {
                        "ok": True,
                        "message": f"Personality instruction [{item['id']}] is already stored.",
                        "instruction": dict(item),
                        "instructions": [dict(entry) for entry in self._instructions],
                    }

            created = {
                "id": uuid4().hex[:12],
                "text": normalized,
                "created_at": dt_util.utcnow().isoformat(),
                "updated_at": dt_util.utcnow().isoformat(),
            }
            self._instructions.append(created)
            await self._save_locked()
            return {
                "ok": True,
                "message": (
                    f"Added personality instruction [{created['id']}]. "
                    "Follow it in this reply and on later turns."
                ),
                "instruction": dict(created),
                "instructions": [dict(entry) for entry in self._instructions],
            }

    async def async_remove(
        self,
        *,
        instruction_id: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Remove one instruction by id or by a substring of its text."""
        target_id = normalize_text(instruction_id)
        target_query = normalize_text(query)
        if not target_id and not target_query:
            return {
                "ok": False,
                "message": "instruction_id or query is required.",
                "removed": [],
            }

        async with self._lock:
            await self._ensure_loaded_locked()
            removed: list[dict[str, Any]] = []
            kept: list[dict[str, Any]] = []
            query_folded = target_query.casefold() if target_query else ""
            matched = False
            for item in self._instructions:
                if target_id and item["id"] == target_id:
                    removed.append(dict(item))
                    matched = True
                    continue
                if (
                    not target_id
                    and query_folded
                    and query_folded in item["text"].casefold()
                    and not matched
                ):
                    removed.append(dict(item))
                    matched = True
                    continue
                kept.append(item)
            self._instructions = kept
            if removed:
                await self._save_locked()

        if not removed:
            return {
                "ok": False,
                "message": "No matching personality instruction was removed.",
                "removed": [],
            }
        return {
            "ok": True,
            "message": f"Removed personality instruction [{removed[0]['id']}].",
            "removed": removed,
        }

    async def async_reset(self) -> dict[str, Any]:
        """Clear every personality instruction."""
        async with self._lock:
            await self._ensure_loaded_locked()
            removed = [dict(item) for item in self._instructions]
            self._instructions = []
            await self._save_locked()
        return {
            "ok": True,
            "message": "Cleared the personality block.",
            "removed_count": len(removed),
        }

    async def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        stored = await self._store.async_load()
        raw_items = stored.get("instructions", []) if isinstance(stored, dict) else []
        self._instructions = [
            normalized
            for item in raw_items
            if (normalized := self._normalize_loaded(item)) is not None
        ]
        self._loaded = True

    async def _save_locked(self) -> None:
        await self._store.async_save({"instructions": list(self._instructions)})

    @staticmethod
    def _normalize_loaded(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        instruction_id = normalize_text(item.get("id"))
        text = normalize_text(item.get("text"))
        created_at = normalize_text(item.get("created_at"))
        if not instruction_id or not text or not created_at:
            return None
        if len(text) > 240:
            text = text[:240].rstrip()
        return {
            "id": instruction_id,
            "text": text,
            "created_at": created_at,
            "updated_at": normalize_text(item.get("updated_at")) or created_at,
        }
