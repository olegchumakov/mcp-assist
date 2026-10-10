"""Persistent memory storage for MCP Assist.

Memories are stored in Home Assistant's storage and persist until they are
explicitly forgotten by the user. There is no automatic expiry and no
automatic pruning of stored memories.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .memory_rank import normalize_text, public_memory, rank_memories, round_vector

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}_memory_store"


class MemoryManager:
    """Manage shared persisted memories for MCP Assist."""

    def __init__(self, hass) -> None:
        """Initialize the memory manager."""
        self.hass = hass
        self._store: Store[dict[str, Any]] = Store(
            hass,
            _STORAGE_VERSION,
            _STORAGE_KEY,
        )
        self._lock = asyncio.Lock()
        self._loaded = False
        self._memories: list[dict[str, Any]] = []

    async def async_initialize(self) -> None:
        """Load persisted memories."""
        async with self._lock:
            await self._ensure_loaded_locked()

    async def async_shutdown(self) -> None:
        """Flush memory state if needed."""
        return None

    async def remember(
        self,
        text: str,
        *,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Store a memory entry that persists until explicitly forgotten."""
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("memory text is required")

        normalized_category = normalize_text(category)

        async with self._lock:
            await self._ensure_loaded_locked()
            memory = {
                "id": uuid4().hex[:12],
                "text": normalized_text,
                "category": normalized_category,
                "created_at": dt_util.utcnow().isoformat(),
            }
            self._memories.append(memory)
            await self._save_locked()

        return dict(memory)

    async def recall(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Recall memories by query or category."""
        normalized_query = normalize_text(query)
        normalized_category = normalize_text(category)
        effective_limit = max(1, int(limit))

        async with self._lock:
            await self._ensure_loaded_locked()
            memories = list(self._memories)

        ranked = rank_memories(
            memories,
            query=normalized_query,
            category=normalized_category,
        )
        items = [public_memory(item) for item in ranked[:effective_limit]]
        return {
            "items": items,
            "total_found": len(ranked),
            "returned_count": len(items),
            "remaining_count": max(0, len(ranked) - len(items)),
        }

    async def forget(
        self,
        *,
        memory_id: str | None = None,
        query: str | None = None,
        category: str | None = None,
        delete_all_matches: bool = False,
    ) -> dict[str, Any]:
        """Forget one or more memories by id or search criteria."""
        normalized_id = normalize_text(memory_id)
        normalized_query = normalize_text(query)
        normalized_category = normalize_text(category)
        if not normalized_id and not normalized_query and not normalized_category:
            raise ValueError("memory_id or query/category is required")

        async with self._lock:
            await self._ensure_loaded_locked()
            target_ids: set[str]
            deleted: list[dict[str, Any]] = []

            if normalized_id:
                target_ids = {normalized_id}
            else:
                ranked = rank_memories(
                    list(self._memories),
                    query=normalized_query,
                    category=normalized_category,
                )
                if not ranked:
                    return {
                        "deleted": [],
                        "deleted_count": 0,
                    }
                if delete_all_matches:
                    target_ids = {item["id"] for item in ranked}
                else:
                    target_ids = {ranked[0]["id"]}

            kept: list[dict[str, Any]] = []
            for memory in self._memories:
                if memory["id"] in target_ids:
                    deleted.append(public_memory(memory))
                else:
                    kept.append(memory)

            self._memories = kept
            if deleted:
                await self._save_locked()

        return {
            "deleted": deleted,
            "deleted_count": len(deleted),
        }

    async def list_all(self) -> list[dict[str, Any]]:
        """Return all memories, newest first."""
        async with self._lock:
            await self._ensure_loaded_locked()
            items = sorted(
                (public_memory(item) for item in self._memories),
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
        return items

    async def async_list_raw(self) -> list[dict[str, Any]]:
        """Return stored memories, including cached embeddings."""
        async with self._lock:
            await self._ensure_loaded_locked()
            return [dict(item) for item in self._memories]

    async def async_set_embeddings(
        self,
        updates: dict[str, list[float]],
        model_key: str,
    ) -> None:
        """Cache vectors for memories. ``model_key`` is stored with each vector."""
        if not updates:
            return
        async with self._lock:
            await self._ensure_loaded_locked()
            changed = False
            for memory in self._memories:
                vector = updates.get(memory["id"])
                if vector is None:
                    continue
                memory["embedding"] = round_vector(vector)
                memory["embedding_model"] = model_key
                changed = True
            if changed:
                await self._save_locked()

    async def _ensure_loaded_locked(self) -> None:
        """Load persisted storage on first use."""
        if self._loaded:
            return

        stored = await self._store.async_load()
        raw_items = stored.get("items", []) if isinstance(stored, dict) else []
        self._memories = [
            normalized
            for item in raw_items
            if (normalized := self._normalize_loaded_memory(item)) is not None
        ]
        self._loaded = True

    async def _save_locked(self) -> None:
        """Persist the current memory list."""
        await self._store.async_save({"items": list(self._memories)})

    def _normalize_loaded_memory(self, item: Any) -> dict[str, Any] | None:
        """Normalize a loaded memory entry from storage."""
        if not isinstance(item, dict):
            return None

        memory_id = normalize_text(item.get("id"))
        text = normalize_text(item.get("text"))
        created_at = normalize_text(item.get("created_at"))
        if not memory_id or not text or not created_at:
            return None

        loaded: dict[str, Any] = {
            "id": memory_id,
            "text": text,
            "category": normalize_text(item.get("category")),
            "created_at": created_at,
        }
        embedding_model = normalize_text(item.get("embedding_model"))
        embedding = item.get("embedding")
        if embedding_model and isinstance(embedding, list) and embedding:
            try:
                loaded["embedding"] = round_vector(embedding)
                loaded["embedding_model"] = embedding_model
            except (TypeError, ValueError):
                pass
        return loaded
