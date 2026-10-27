"""Automatic memory retrieval: local embeddings, optional remote embeddings, keyword fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .const import EMBEDDING_PROVIDER_LOCAL, EMBEDDING_PROVIDER_OPENAI
from .embedding_api import embeddings_url, parse_openai_embeddings
from .local_embedder import (
    BertTextEmbedder,
    append_download_chunk,
    apply_prefix,
    discard_partial_download,
    file_is_complete,
    finalize_partial_download,
    huggingface_file_url,
    load_embedder,
    local_model_spec,
    model_is_downloaded,
    prepare_partial_download,
)
from .memory_manager import MemoryManager
from .memory_rank import (
    build_search_text,
    cosine_similarity,
    format_merge_result,
    keyword_auto_matches,
    memories_missing_embedding,
    plan_duplicate_merges,
    rank_by_cosine,
    round_vector,
)
from .memory_settings import MemorySettings, read_memory_settings

_LOGGER = logging.getLogger(__name__)

_BACKFILL_LIMIT = 16
_DOWNLOAD_TIMEOUT_SECONDS = 180
_OPENAI_TIMEOUT_SECONDS = 2.5
_PREPARE_RETRY_SECONDS = 300
_USER_AGENT = "mcp-assist-se/0.24.1"


@dataclass(frozen=True)
class RetrievalResult:
    """Memories to inject, plus how they were chosen."""

    items: list[dict[str, Any]]
    mode: str
    reason: str


class EmbeddingService:
    """Embed memories locally by default and retrieve the closest ones."""

    def __init__(self, hass, memory: MemoryManager) -> None:
        """Initialize the service. The model stays unloaded until prepare runs."""
        self.hass = hass
        self._memory = memory
        self._prepare_lock = asyncio.Lock()
        self._compute_lock = asyncio.Lock()
        self._prepare_task: asyncio.Task | None = None
        self._embedder: BertTextEmbedder | None = None
        self._ready_model_id: str | None = None
        self._failed_until = 0.0
        self._failure_reason = ""
        self._noted_base_url = ""
        self._noted_api_key = ""

    def note_profile_endpoint(self, base_url: str, api_key: str) -> None:
        """Remember the conversation profile's endpoint for the optional remote embedder."""
        self._noted_base_url = (base_url or "").rstrip("/")
        self._noted_api_key = api_key or ""

    def is_ready(self, model_id: str) -> bool:
        """True when that local model is loaded."""
        return self._embedder is not None and self._ready_model_id == model_id

    def schedule_prepare(self) -> None:
        """Start a background download and load when local embeddings are configured."""
        settings = read_memory_settings(self.hass)
        if not settings.memory_tools or settings.provider != EMBEDDING_PROVIDER_LOCAL:
            if settings.provider != EMBEDDING_PROVIDER_LOCAL:
                self._embedder = None
                self._ready_model_id = None
            return
        if self.is_ready(settings.local_model):
            return
        if time.monotonic() < self._failed_until:
            return
        if self._prepare_task is not None and not self._prepare_task.done():
            return
        self._prepare_task = self.hass.async_create_task(
            self._prepare(settings.local_model)
        )

    async def async_embed_new_memory(self, memory_id: str, text: str) -> None:
        """Cache an embedding for a memory that was just stored. Never raises."""
        try:
            settings = read_memory_settings(self.hass)
            if not settings.memory_tools:
                return
            if settings.provider == EMBEDDING_PROVIDER_OPENAI:
                await self._embed_and_store_openai(settings, {memory_id: text}, role="passage")
                return
            self.schedule_prepare()
            await self._embed_local_when_ready(settings.local_model, {memory_id: text}, role="passage")
        except Exception as err:
            _LOGGER.warning("Could not embed new memory %s: %s", memory_id, err)

    async def async_retrieve(self, query: str, previous: str | None) -> RetrievalResult:
        """Return the memories that should be injected for this utterance."""
        settings = read_memory_settings(self.hass)
        if not settings.memory_tools or not settings.auto_retrieve:
            return RetrievalResult([], "skipped", "disabled")

        search_text = build_search_text(
            query,
            previous,
            include_previous=settings.include_last_turn,
        )
        if not search_text:
            return RetrievalResult([], "skipped", "empty")

        try:
            memories = await self._memory.async_list_raw()
        except Exception as err:
            _LOGGER.warning("Could not load memories for retrieval: %s", err)
            return RetrievalResult([], "skipped", "load failed")
        if not memories:
            return RetrievalResult([], "skipped", "no memories")

        started = time.perf_counter()
        if settings.provider == EMBEDDING_PROVIDER_OPENAI:
            ranked = await self._retrieve_openai(settings, memories, search_text)
            if ranked is not None:
                _LOGGER.info(
                    "Memory retrieval: %d fact(s) via openai embeddings in %.0f ms",
                    len(ranked),
                    (time.perf_counter() - started) * 1000,
                )
                return RetrievalResult(ranked, "openai", settings.openai_model)
            fallback = keyword_auto_matches(memories, search_text, limit=settings.top_k)
            _LOGGER.info(
                "Memory retrieval: openai embeddings unavailable, %d keyword fact(s)",
                len(fallback),
            )
            return RetrievalResult(fallback, "keyword", "openai embeddings unavailable")

        if not self.is_ready(settings.local_model):
            self.schedule_prepare()
            fallback = keyword_auto_matches(memories, search_text, limit=settings.top_k)
            reason = self._failure_reason or "local model not loaded"
            _LOGGER.info(
                "Memory retrieval: %d keyword fact(s) (%s)",
                len(fallback),
                reason,
            )
            return RetrievalResult(fallback, "keyword", reason)

        ranked = await self._retrieve_local(settings, memories, search_text)
        if ranked is None:
            fallback = keyword_auto_matches(memories, search_text, limit=settings.top_k)
            _LOGGER.info(
                "Memory retrieval: local embeddings failed, %d keyword fact(s)",
                len(fallback),
            )
            return RetrievalResult(fallback, "keyword", "local embedding failed")

        _LOGGER.info(
            "Memory retrieval: %d fact(s) via %s in %.0f ms",
            len(ranked),
            settings.local_model,
            (time.perf_counter() - started) * 1000,
        )
        return RetrievalResult(ranked, "local", settings.local_model)

    async def _retrieve_local(
        self,
        settings: MemorySettings,
        memories: list[dict[str, Any]],
        search_text: str,
    ) -> list[dict[str, Any]] | None:
        spec = local_model_spec(settings.local_model)
        missing = memories_missing_embedding(
            memories, settings.cache_key, limit=_BACKFILL_LIMIT
        )
        texts = [apply_prefix(search_text, spec, "query")]
        texts.extend(apply_prefix(str(item["text"]), spec, "passage") for item in missing)
        vectors = await self._embed_loaded(texts)
        if not vectors or len(vectors) != len(texts):
            return None
        if missing:
            updates = {
                item["id"]: round_vector(vector)
                for item, vector in zip(missing, vectors[1:])
            }
            for item, vector in zip(missing, vectors[1:]):
                item["embedding"] = updates[item["id"]]
                item["embedding_model"] = settings.cache_key
            await self._memory.async_set_embeddings(updates, settings.cache_key)
        return rank_by_cosine(
            memories,
            vectors[0],
            model_key=settings.cache_key,
            threshold=settings.threshold,
            limit=settings.top_k,
        )

    async def _retrieve_openai(
        self,
        settings: MemorySettings,
        memories: list[dict[str, Any]],
        search_text: str,
    ) -> list[dict[str, Any]] | None:
        missing = memories_missing_embedding(
            memories, settings.cache_key, limit=_BACKFILL_LIMIT
        )
        pending = {item["id"]: str(item["text"]) for item in missing}
        # Embed the query in the same call when we already have work, otherwise alone.
        query_key = "__query__"
        payload_items = {query_key: search_text, **pending}
        vectors = await self._openai_embed(settings, list(payload_items.values()))
        if not vectors or len(vectors) != len(payload_items):
            return None
        query_vector = vectors[0]
        if pending:
            updates = {
                memory_id: round_vector(vector)
                for memory_id, vector in zip(pending, vectors[1:])
            }
            for item in missing:
                item["embedding"] = updates[item["id"]]
                item["embedding_model"] = settings.cache_key
            await self._memory.async_set_embeddings(updates, settings.cache_key)
        return rank_by_cosine(
            memories,
            query_vector,
            model_key=settings.cache_key,
            threshold=settings.threshold,
            limit=settings.top_k,
        )

    async def _embed_and_store_openai(
        self,
        settings: MemorySettings,
        memories: dict[str, str],
        *,
        role: str,
    ) -> None:
        del role  # OpenAI embeddings are not prefixed.
        vectors = await self._openai_embed(settings, list(memories.values()))
        if not vectors or len(vectors) != len(memories):
            return
        updates = {
            memory_id: round_vector(vector)
            for memory_id, vector in zip(memories, vectors)
        }
        await self._memory.async_set_embeddings(updates, settings.cache_key)

    async def _embed_local_when_ready(
        self,
        model_id: str,
        memories: dict[str, str],
        *,
        role: str,
    ) -> None:
        for _ in range(90):
            if self.is_ready(model_id):
                break
            if time.monotonic() < self._failed_until and not self.is_ready(model_id):
                return
            await asyncio.sleep(1)
        else:
            return
        if not self.is_ready(model_id):
            return
        spec = local_model_spec(model_id)
        texts = [apply_prefix(text, spec, role) for text in memories.values()]  # type: ignore[arg-type]
        vectors = await self._embed_loaded(texts)
        if not vectors or len(vectors) != len(memories):
            return
        settings = read_memory_settings(self.hass)
        if settings.cache_key != f"local:{model_id}":
            return
        updates = {
            memory_id: round_vector(vector)
            for memory_id, vector in zip(memories, vectors)
        }
        await self._memory.async_set_embeddings(updates, settings.cache_key)

    async def _embed_loaded(self, texts: list[str]) -> list[list[float]] | None:
        embedder = self._embedder
        if embedder is None or not texts:
            return None
        try:
            async with self._compute_lock:
                return await self.hass.async_add_executor_job(embedder.embed, texts)
        except Exception as err:
            _LOGGER.warning("Local embedding failed: %s", err)
            return None

    async def _prepare(self, model_id: str) -> None:
        async with self._prepare_lock:
            if self.is_ready(model_id):
                return
            spec = local_model_spec(model_id)
            directory = self._model_dir(spec.model_id)
            try:
                downloaded = await self.hass.async_add_executor_job(
                    model_is_downloaded, directory, spec
                )
                if not downloaded:
                    _LOGGER.info(
                        "Downloading local embedding model %s (~%d MB) into %s",
                        spec.model_id,
                        spec.download_mb,
                        directory,
                    )
                    await self._download(spec, directory)
                _LOGGER.info("Loading local embedding model %s", spec.model_id)
                embedder = await self.hass.async_add_executor_job(
                    load_embedder, directory, spec
                )
                async with self._compute_lock:
                    await self.hass.async_add_executor_job(embedder.embed, ["."])
                self._embedder = embedder
                self._ready_model_id = spec.model_id
                self._failure_reason = ""
                _LOGGER.info(
                    "Local embedding model ready: %s (%d MB download, about %d MB RAM)",
                    spec.model_id,
                    spec.download_mb,
                    spec.ram_mb,
                )
            except Exception as err:
                self._embedder = None
                self._ready_model_id = None
                self._failed_until = time.monotonic() + _PREPARE_RETRY_SECONDS
                self._failure_reason = str(err)
                _LOGGER.warning(
                    "Local embedding model %s is unavailable (%s). "
                    "Memory retrieval will use keyword matching and retry later.",
                    spec.model_id,
                    err,
                )
            else:
                await self._maybe_merge_duplicates()

    async def async_similarities(
        self,
        query: str,
        items: list[dict[str, Any]],
    ) -> dict[str, float] | None:
        """Cosine of ``query`` against each item, or None if embeddings are unavailable."""
        cleaned = [
            item
            for item in items
            if item.get("id") and str(item.get("text") or "").strip()
        ]
        if not cleaned or not str(query or "").strip():
            return {}
        settings = read_memory_settings(self.hass)
        vectors = await self._embed_texts(
            settings,
            [str(query), *(str(item["text"]) for item in cleaned)],
            query_count=1,
        )
        if not vectors or len(vectors) != len(cleaned) + 1:
            return None
        return {
            str(item["id"]): cosine_similarity(vectors[0], vectors[index + 1])
            for index, item in enumerate(cleaned)
        }

    async def async_similarity_matrix(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]] | None:
        """Pairwise cosine for duplicate grouping, or None if embeddings are unavailable."""
        cleaned = [
            item
            for item in items
            if item.get("id") and str(item.get("text") or "").strip()
        ]
        if len(cleaned) < 2:
            return {}
        settings = read_memory_settings(self.hass)
        vectors = await self._embed_texts(
            settings,
            [str(item["text"]) for item in cleaned],
            query_count=0,
        )
        if not vectors or len(vectors) != len(cleaned):
            return None
        matrix: dict[str, dict[str, float]] = {}
        for left_index, left in enumerate(cleaned):
            row: dict[str, float] = {}
            for right_index, right in enumerate(cleaned):
                if left_index == right_index:
                    continue
                row[str(right["id"])] = cosine_similarity(
                    vectors[left_index], vectors[right_index]
                )
            matrix[str(left["id"])] = row
        return matrix

    async def async_merge_duplicates(self) -> dict[str, Any]:
        """Merge near-duplicate memories and personality lines into the newest of each group."""
        from .runtime import get_assist_runtime

        settings = read_memory_settings(self.hass)
        memories = await self._memory.async_list_raw()
        memory_matrix = await self.async_similarity_matrix(memories)
        if memory_matrix is None:
            return {
                "ok": False,
                "message": "Embeddings are not available yet, so no duplicates were merged.",
                "memory_groups": [],
                "personality_groups": [],
            }
        memory_groups = plan_duplicate_merges(
            memories, memory_matrix, settings.auto_merge_threshold
        )
        for group in memory_groups:
            deleted = await self._memory.async_delete_ids(
                {str(item["id"]) for item in group["remove"]}
            )
            deleted_ids = {str(item.get("id")) for item in deleted}
            group["remove"] = [
                item for item in group["remove"] if str(item.get("id")) in deleted_ids
            ]
            for item in group["remove"]:
                _LOGGER.warning(
                    "Merged duplicate memory %s into %s. Removed: %s. Kept: %s.",
                    item.get("id"),
                    group["keep"].get("id"),
                    item.get("text"),
                    group["keep"].get("text"),
                )

        personality = get_assist_runtime(self.hass).personality
        instructions = await personality.async_view()
        personality_groups: list[dict[str, Any]] = []
        if len(instructions) >= 2:
            personality_matrix = await self.async_similarity_matrix(instructions)
            if personality_matrix is None:
                _LOGGER.warning(
                    "Could not score personality instructions, so those duplicates were left in place"
                )
            else:
                personality_groups = plan_duplicate_merges(
                    instructions, personality_matrix, settings.auto_merge_threshold
                )
                for group in personality_groups:
                    deleted = await personality.async_delete_ids(
                        {str(item["id"]) for item in group["remove"]}
                    )
                    deleted_ids = {str(item.get("id")) for item in deleted}
                    group["remove"] = [
                        item
                        for item in group["remove"]
                        if str(item.get("id")) in deleted_ids
                    ]
                    for item in group["remove"]:
                        _LOGGER.warning(
                            "Merged duplicate personality instruction %s into %s. Removed: %s. Kept: %s.",
                            item.get("id"),
                            group["keep"].get("id"),
                            item.get("text"),
                            group["keep"].get("text"),
                        )
        await self._memory.async_set_merge_revision()
        return {
            "ok": True,
            "message": format_merge_result(memory_groups, personality_groups),
            "memory_groups": memory_groups,
            "personality_groups": personality_groups,
        }

    async def _maybe_merge_duplicates(self) -> None:
        """Run the upgrade merge once, after the local model is loaded."""
        try:
            if await self._memory.async_get_merge_revision() >= 1:
                return
            result = await self.async_merge_duplicates()
            if not result.get("ok"):
                _LOGGER.warning("Duplicate cleanup skipped: %s", result.get("message"))
                return
            removed = sum(len(group["remove"]) for group in result["memory_groups"])
            removed += sum(len(group["remove"]) for group in result["personality_groups"])
            if removed:
                _LOGGER.warning(
                    "Duplicate cleanup removed %d older item(s) and kept the newest text in each group.",
                    removed,
                )
            else:
                _LOGGER.info("Duplicate cleanup found nothing to merge.")
        except Exception as err:
            _LOGGER.warning("Duplicate cleanup failed: %s", err)

    async def _embed_texts(
        self,
        settings: MemorySettings,
        texts: list[str],
        *,
        query_count: int,
    ) -> list[list[float]] | None:
        """Embed texts. The first ``query_count`` items are queries; the rest are passages."""
        if settings.provider == EMBEDDING_PROVIDER_OPENAI:
            return await self._openai_embed(settings, texts)
        if not self.is_ready(settings.local_model):
            return None
        spec = local_model_spec(settings.local_model)
        prefixed = [
            apply_prefix(text, spec, "query" if index < query_count else "passage")
            for index, text in enumerate(texts)
        ]
        return await self._embed_loaded(prefixed)

    async def _download(self, spec, directory: Path) -> None:
        timeout = aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT_SECONDS)
        headers = {"User-Agent": _USER_AGENT}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            for filename, expected_size in spec.files.items():
                destination = directory / filename
                complete = await self.hass.async_add_executor_job(
                    file_is_complete, destination, expected_size
                )
                if complete:
                    continue
                await self._download_file(
                    session,
                    huggingface_file_url(spec.repo, filename),
                    destination,
                    expected_size,
                )

    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        destination: Path,
        expected_size: int | None,
    ) -> None:
        """Stream one model file. Every filesystem call runs in the executor."""
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            await self.hass.async_add_executor_job(prepare_partial_download, partial)
            async with session.get(url) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_chunked(1 << 20):
                    if not chunk:
                        continue
                    await self.hass.async_add_executor_job(
                        append_download_chunk, partial, chunk
                    )
            await self.hass.async_add_executor_job(
                finalize_partial_download, partial, destination, expected_size
            )
        except Exception:
            try:
                await self.hass.async_add_executor_job(discard_partial_download, partial)
            except Exception:
                _LOGGER.warning(
                    "Could not remove partial download %s", partial, exc_info=True
                )
            raise

    def _model_dir(self, model_id: str) -> Path:
        safe = model_id.replace("/", "--")
        return Path(self.hass.config.path("mcp_assist_se", "models", safe))

    def _openai_target(self, settings: MemorySettings) -> tuple[str, str, str]:
        base_url = settings.openai_base_url or self._noted_base_url
        api_key = self._noted_api_key
        if settings.openai_base_url and settings.openai_base_url != self._noted_base_url:
            api_key = self._api_key_for_url(settings.openai_base_url) or api_key
        if not base_url:
            base_url, found_key = self._first_profile_endpoint()
            api_key = api_key or found_key
        return base_url, api_key, settings.openai_model

    def _api_key_for_url(self, base_url: str) -> str:
        from .const import CONF_API_KEY, CONF_LMSTUDIO_URL, DOMAIN, SYSTEM_ENTRY_UNIQUE_ID

        target = base_url.rstrip("/")
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
                continue
            entry_url = (
                entry.options.get(CONF_LMSTUDIO_URL)
                or entry.data.get(CONF_LMSTUDIO_URL)
                or ""
            ).rstrip("/")
            if entry_url == target:
                return entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY) or ""
        return ""

    def _first_profile_endpoint(self) -> tuple[str, str]:
        from .const import CONF_API_KEY, CONF_LMSTUDIO_URL, DOMAIN, SYSTEM_ENTRY_UNIQUE_ID

        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
                continue
            entry_url = (
                entry.options.get(CONF_LMSTUDIO_URL)
                or entry.data.get(CONF_LMSTUDIO_URL)
                or ""
            ).rstrip("/")
            if not entry_url:
                continue
            api_key = entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY) or ""
            return entry_url, api_key
        return "", ""

    async def _openai_embed(
        self, settings: MemorySettings, texts: list[str]
    ) -> list[list[float]] | None:
        base_url, api_key, model = self._openai_target(settings)
        if not base_url or not model or not texts:
            _LOGGER.warning("OpenAI embeddings skipped: base URL or model is empty")
            return None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = aiohttp.ClientTimeout(total=_OPENAI_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    embeddings_url(base_url),
                    json={"model": model, "input": texts},
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        _LOGGER.warning(
                            "OpenAI embeddings HTTP %s: %s",
                            response.status,
                            body[:300],
                        )
                        return None
                    payload = await response.json()
        except Exception as err:
            _LOGGER.warning("OpenAI embeddings request failed: %s", err)
            return None
        return parse_openai_embeddings(payload)
