"""MCP Server for Home Assistant entity discovery."""

import asyncio
import ipaddress
import json
import logging
import math
from functools import partial
from typing import Any, Dict, List
from urllib.parse import urlparse
from datetime import timedelta

from aiohttp import web, WSMsgType
from aiohttp.web_ws import WebSocketResponse

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.components.recorder import get_instance, history
from homeassistant.util import dt as dt_util

from .access_control import (
    authorize_action,
    external_tool_allowed,
    filter_discovery_rows,
    filter_index,
    grant_for_person,
    grant_from_payload,
    is_unrestricted,
    load_access_config,
    refusal_tool_result,
    strip_targeting,
    entity_visible,
)
from .access_hass import snapshot_from_hass
from .action_verify import (
    CONFIRM_WAIT_SECONDS,
    POLL_SECONDS,
    SETTLED_WAIT_SECONDS,
    expects_change,
    facts_from_context,
    guard_empty_activation,
    is_dead_snapshot,
    judge_entity,
    observation_changed,
    public_snapshot,
    render_verification,
    snapshot_from_mapping,
)
from .const import (
    MCP_SERVER_NAME,
    MCP_PROTOCOL_VERSION,
    MAX_ENTITIES_PER_DISCOVERY,
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_SERVER_TYPE,
    CONF_ALLOWED_IPS,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_PERSONALITY,
    DEFAULT_API_KEY,
    DEFAULT_ENABLE_MEMORY_TOOLS,
    DEFAULT_ENABLE_PERSONALITY,
    PERFORM_ACTION_DATA_DESCRIPTION,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_MODEL_NAME,
    GEMINI_BASE_URL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_HERMES,
    SERVER_TYPE_OPENCLAW,
)
from .memory_rank import (
    find_exact_text,
    format_forget_result,
    format_noop_result,
    format_remember_result,
    format_similarity_decision,
    format_store_decision_prompt,
    looks_like_personality_request,
    memory_compare_key,
    parse_llm_store_decision,
    parse_store_mode,
    plan_store,
    select_forget_matches,
)
from .store_decision import complete_store_decision, openai_chat_url
from .memory_settings import read_memory_settings
from .runtime import get_assist_runtime
from .discovery import EntityDiscovery
from .state_summary import format_display_name, summarize_state
from .domain_registry import (
    validate_domain_action,
    get_supported_domains,
    get_domain_info,
    get_domains_by_type,
    TYPE_CONTROLLABLE,
    TYPE_READ_ONLY,
    TYPE_SERVICE_ONLY,
)

_LOGGER = logging.getLogger(__name__)

# Sentinel marking values dropped by _strip_non_json_serializable
_OMIT = object()


def _sanitize_log_value(value: Any, max_len: int = 500) -> str:
    """Make a client-supplied value safe to log.

    Strips newlines/control characters (log-injection) and caps length.
    """
    text = str(value)
    text = "".join(ch for ch in text if ch >= " " or ch == "\t")
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text


def _strip_non_json_serializable(value: Any) -> Any:
    """Recursively drop values json.dumps cannot encode.

    Entity attributes may hold arbitrary Python objects or non-finite floats
    (NaN/Infinity), either of which makes json.dumps raise.
    """
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if not isinstance(key, str):
                key = str(key)
            item = _strip_non_json_serializable(item)
            if item is not _OMIT:
                cleaned[key] = item
        return cleaned
    if isinstance(value, (list, tuple, set)):
        return [
            item
            for item in (_strip_non_json_serializable(i) for i in value)
            if item is not _OMIT
        ]
    if isinstance(value, float):
        return value if math.isfinite(value) else _OMIT
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return _OMIT


class MCPServer:
    """MCP Server for entity discovery."""

    def __init__(self, hass: HomeAssistant, port: int, entry=None) -> None:
        """Initialize MCP server."""
        self.hass = hass
        self.port = port
        self.entry = entry
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.discovery = EntityDiscovery(hass)
        self.sse_clients = []  # Track SSE connections for notifications
        self.progress_queues = set()  # Track progress SSE clients

        # Extract allowed IPs from LM Studio URL
        self.allowed_ips = ["127.0.0.1", "::1"]  # Always allow localhost

        # Get LM Studio URL from config
        lmstudio_url = DEFAULT_LMSTUDIO_URL
        if entry:
            # Check options first, then data
            lmstudio_url = entry.options.get(
                CONF_LMSTUDIO_URL,
                entry.data.get(CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL),
            )

        # Extract hostname/IP from LM Studio URL
        try:
            parsed = urlparse(lmstudio_url)
            lmstudio_host = parsed.hostname or parsed.netloc.split(":")[0]
            if lmstudio_host and lmstudio_host not in self.allowed_ips:
                self.allowed_ips.append(lmstudio_host)
                _LOGGER.info(
                    "MCP server automatically whitelisted LM Studio IP: %s",
                    lmstudio_host,
                )
        except Exception as e:
            _LOGGER.warning("Could not parse LM Studio URL '%s': %s", lmstudio_url, e)

        # Add user-configured allowed IPs/CIDR ranges (shared setting)
        allowed_ips_str = self._get_shared_setting(
            CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS
        )
        if allowed_ips_str:
            # Parse comma-separated list
            additional_ips = [
                ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()
            ]
            for ip_entry in additional_ips:
                if ip_entry not in self.allowed_ips:
                    self.allowed_ips.append(ip_entry)
            if additional_ips:
                _LOGGER.info(
                    "MCP server added user-configured allowed IPs/ranges: %s",
                    additional_ips,
                )

        _LOGGER.info("MCP server allowed IPs/ranges: %s", self.allowed_ips)

        # Custom tools will be initialized in start() after system entry exists
        self.custom_tools = None
        # Per-conversation remember guards. Shown texts may be confirmed with
        # mode=add. A failed update blocks the following mode=add.
        self._store_guards: dict[str, dict[str, set[str]]] = {}
        # Speaker grant for a voice turn whose tools arrive without context
        # (OpenClaw / Hermes call the MCP server themselves).
        self._bound_grants: dict[str, Any] = {}
        self._tool_grant = None
        self._tool_config = None
        self._access_snap = None
        self._access_snap_failed = False

    def _get_shared_setting(self, key: str, default: Any) -> Any:
        """Get a shared setting from system entry with fallback to profile entry."""
        # Import here to avoid circular dependency
        from . import get_system_entry

        # Try to get from system entry first
        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        # Fallback to profile entry for backward compatibility
        if self.entry:
            value = self.entry.options.get(key, self.entry.data.get(key))
            if value is not None:
                return value

        # Return default
        return default

    def _get_search_provider(self) -> str:
        """Get search provider (shared setting) with backward compatibility."""
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER, None)
        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    def _external_custom_tools_enabled(self) -> bool:
        """Return whether user-defined external custom tool packages are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
            )
        )

    async def start(self) -> None:
        """Start the MCP server."""
        try:
            _LOGGER.info(
                "Starting MCP server on port %d, binding to all interfaces (0.0.0.0)",
                self.port,
            )

            # Create web application (IP checks are done per-handler, not via middleware)
            self.app = web.Application()
            self.app.router.add_post("/", self.handle_mcp_request)
            self.app.router.add_get("/sse", self.handle_sse)  # SSE endpoint
            self.app.router.add_get("/", self.handle_sse)  # Also handle root GET as SSE
            self.app.router.add_get("/ws", self.handle_websocket)
            self.app.router.add_get("/health", self.handle_health)
            self.app.router.add_get(
                "/external-tools/diagnostics",
                self.handle_external_tool_diagnostics,
            )
            self.app.router.add_get(
                "/progress", self.handle_progress_stream
            )  # Progress streaming

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            # Bind to all interfaces so external machines can connect
            self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
            await self.site.start()

            # Create and initialize built-in package wrappers and optional external tools
            # Done here (not in __init__) so system entry exists for reading settings
            try:
                from .custom_tools import CustomToolsLoader

                self.custom_tools = CustomToolsLoader(self.hass, self.entry)
                await self.custom_tools.initialize()
                _LOGGER.info(
                    "✅ Custom tools initialized (search provider: %s, external enabled: %s)",
                    self._get_search_provider(),
                    self._external_custom_tools_enabled(),
                )
            except Exception as e:
                _LOGGER.error(f"Failed to initialize custom tools: {e}")

            if self._memory_tools_enabled() or self._personality_enabled():
                runtime = get_assist_runtime(self.hass)
                try:
                    if self._memory_tools_enabled():
                        await runtime.memory.async_initialize()
                        _LOGGER.info("Memory tools initialized (persist until forgotten)")
                        runtime.embeddings.schedule_prepare()
                    if self._personality_enabled():
                        await runtime.personality.async_initialize()
                        _LOGGER.info("Personality block initialized")
                except Exception as err:
                    _LOGGER.error("Failed to initialize memory or personality: %s", err)

            _LOGGER.info(
                "✅ MCP server started successfully on http://0.0.0.0:%d", self.port
            )
            _LOGGER.info("🌐 MCP server is accessible from external machines")
            _LOGGER.info(
                "🔗 Health check available at: http://<your-ha-ip>:%d/health", self.port
            )
            _LOGGER.info("📡 WebSocket endpoint: ws://<your-ha-ip>:%d/ws", self.port)
            _LOGGER.info("📤 HTTP endpoint: http://<your-ha-ip>:%d/", self.port)

        except OSError as err:
            if err.errno == 98:  # Address already in use
                _LOGGER.error(
                    "❌ Port %d is already in use. Please choose a different port.",
                    self.port,
                )
                raise
            elif err.errno == 13:  # Permission denied
                _LOGGER.error(
                    "❌ Permission denied to bind to port %d. Try a port >= 1024.",
                    self.port,
                )
                raise
            else:
                _LOGGER.error(
                    "❌ Failed to bind MCP server to port %d: %s", self.port, err
                )
                raise
        except Exception as err:
            _LOGGER.error("❌ Failed to start MCP server: %s", err)
            raise

    async def stop(self) -> None:
        """Stop the MCP server."""
        _LOGGER.info("Stopping MCP server")

        if self.custom_tools:
            await self.custom_tools.shutdown()
            self.custom_tools = None

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    def _store_guard(self, conversation_id: str | None) -> dict[str, set[str]]:
        """Return the shown/failed sets for one conversation."""
        key = (conversation_id or "").strip() or "default"
        guard = self._store_guards.get(key)
        if guard is None:
            if len(self._store_guards) >= 40:
                self._store_guards.pop(next(iter(self._store_guards)))
            guard = {"shown": set(), "failed": set()}
            self._store_guards[key] = guard
        return guard

    def _chat_decision_target(self) -> tuple[str, str, str] | None:
        """OpenAI-compatible chat settings for one short store decision."""
        entry = self.entry
        if entry is None:
            return None
        server_type = entry.options.get(
            CONF_SERVER_TYPE, entry.data.get(CONF_SERVER_TYPE, "")
        )
        if server_type in {
            SERVER_TYPE_OLLAMA,
            SERVER_TYPE_ANTHROPIC,
            SERVER_TYPE_HERMES,
            SERVER_TYPE_OPENCLAW,
        }:
            return None
        model = entry.options.get(
            CONF_MODEL_NAME, entry.data.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME)
        )
        if not model:
            return None
        api_key = entry.options.get(
            CONF_API_KEY, entry.data.get(CONF_API_KEY, DEFAULT_API_KEY)
        ) or ""
        if server_type == SERVER_TYPE_GEMINI:
            base_url = GEMINI_BASE_URL
        elif server_type == SERVER_TYPE_OPENROUTER:
            base_url = OPENROUTER_BASE_URL
        elif server_type == SERVER_TYPE_OPENAI:
            base_url = entry.options.get(
                CONF_LMSTUDIO_URL, entry.data.get(CONF_LMSTUDIO_URL, OPENAI_BASE_URL)
            )
        else:
            base_url = entry.options.get(
                CONF_LMSTUDIO_URL, entry.data.get(CONF_LMSTUDIO_URL, "")
            )
        if not base_url:
            return None
        return openai_chat_url(str(base_url)), str(api_key), str(model)

    async def _llm_store_decision(
        self, kind: str, new_text: str, candidates: list
    ) -> dict[str, Any] | None:
        """Ask for add/update/noop. None means do not store."""
        target = self._chat_decision_target()
        if target is None or not candidates:
            return None
        url, api_key, model = target
        prompt = format_store_decision_prompt(kind, new_text, candidates)
        raw = await complete_store_decision(
            url=url, api_key=api_key, model=model, prompt=prompt
        )
        if not raw:
            return None
        allowed = {str(item.get("id")) for item in candidates if item.get("id")}
        decision = parse_llm_store_decision(raw, allowed)
        if decision is None:
            _LOGGER.warning("Store decision was not usable JSON")
        return decision

    async def _plan_store_call(
        self,
        *,
        kind: str,
        mode: str | None,
        new_text: str,
        items: list,
        requested_id: str | None,
        conversation_id: str | None,
    ) -> dict[str, Any]:
        """Score locally, then spend one short model call only when still ambiguous."""
        exact = find_exact_text(items, new_text)
        if exact is not None:
            _LOGGER.info("Store decision for %s: noop (exact)", kind)
            return {
                "action": "noop",
                "reason": "exact",
                "target_id": exact.get("id"),
                "target": exact,
                "candidates": [],
            }
        if mode == "noop":
            _LOGGER.info("Store decision for %s: noop (mode)", kind)
            return {
                "action": "noop",
                "reason": "mode",
                "target_id": None,
                "target": None,
                "candidates": [],
            }
        settings = read_memory_settings(self.hass)
        similarities = await self._score_items(new_text, items)
        guard = self._store_guard(conversation_id)
        token = f"{kind}:{memory_compare_key(new_text)}"
        shown = token in guard["shown"]
        failed = token in guard["failed"]
        preview = plan_store(
            mode=mode,
            new_text=new_text,
            items=items,
            similarities=similarities,
            requested_id=requested_id,
            candidate_threshold=settings.candidate_threshold,
            near_copy_threshold=settings.auto_merge_threshold,
            candidates_shown=shown,
            failed_update=failed,
        )
        needs_model = (
            preview["action"] == "candidates"
            and bool(preview["candidates"])
            and mode != "add"
            and not failed
        )
        decision = None
        if needs_model:
            decision = await self._llm_store_decision(
                kind, new_text, preview["candidates"]
            )
        plan = preview
        if decision is not None:
            plan = plan_store(
                mode=mode,
                new_text=new_text,
                items=items,
                similarities=similarities,
                requested_id=requested_id,
                candidate_threshold=settings.candidate_threshold,
                near_copy_threshold=settings.auto_merge_threshold,
                candidates_shown=shown,
                failed_update=failed,
                llm=decision,
            )
        if plan["action"] == "candidates":
            guard["shown"].add(token)
            if plan["reason"] == "unknown_id":
                guard["failed"].add(token)
        elif token in guard["failed"]:
            guard["failed"].discard(token)
        _LOGGER.info(
            "Store decision for %s: %s (%s)", kind, plan["action"], plan["reason"]
        )
        return plan

    def _candidates_result(self, kind: str, new_text: str, candidates: list) -> Dict[str, Any]:
        """Return real ids and store nothing."""
        if not candidates:
            id_field = "memory_id" if kind == "memory" else "instruction_id"
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"NOT STORED. No {kind} with that id. "
                            f"Nothing was written. Copy {id_field} from an existing row. "
                            "A category name is not an id."
                        ),
                    }
                ],
                "stored": False,
                "isError": True,
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": format_similarity_decision(kind, new_text, candidates),
                }
            ],
            "stored": False,
            "candidates": [
                {
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "category": item.get("category"),
                    "score": item.get("score"),
                }
                for item in candidates
            ],
        }

    def _noop_result(self, kind: str, plan: dict[str, Any]) -> Dict[str, Any]:
        target = plan.get("target") or {}
        if not target:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "NOOP. Nothing was stored. The fact was already known.",
                    }
                ],
                "stored": False,
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": format_noop_result(kind, target, plan.get("reason") or ""),
                }
            ],
            "stored": False,
            kind: target,
        }

    async def tool_remember_memory(
        self, args: Dict[str, Any], context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Store a persisted memory (kept until explicitly forgotten)."""
        memory_text = " ".join(str(args.get("memory") or "").split()).strip()
        if not memory_text:
            return {
                "content": [{"type": "text", "text": "Memory text is required."}],
                "isError": True,
            }

        category = args.get("category")
        if self._personality_enabled() and looks_like_personality_request(memory_text):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a speaking-style or persona request, so it was not stored as a memory. "
                            "Call set_personality_instruction now with that instruction. "
                            "Personality instructions are injected into every later reply."
                        ),
                    }
                ],
            }

        mode = parse_store_mode(args.get("mode"))
        if mode == "invalid":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "NOT STORED. mode must be add, update, or noop. "
                            "Nothing was written."
                        ),
                    }
                ],
                "isError": True,
            }
        memory_id = str(args.get("memory_id") or "").strip()
        category_text = str(category).strip() if category is not None else ""

        self.publish_progress(
            "tool_start",
            "Storing memory",
            tool="remember_memory",
        )

        conversation_id = (
            context.get("conversation_id") if isinstance(context, dict) else None
        )
        try:
            if mode == "update" and not memory_id:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "NOT STORED. mode=update requires memory_id. "
                                "Nothing was written. Copy a real id. A category is not an id."
                            ),
                        }
                    ],
                    "isError": True,
                    "stored": False,
                }
            existing = await self._memory().async_list_raw()
            plan = await self._plan_store_call(
                kind="memory",
                mode=mode,
                new_text=memory_text,
                items=existing,
                requested_id=memory_id or None,
                conversation_id=None if conversation_id is None else str(conversation_id),
            )
            if plan["action"] == "noop":
                return self._noop_result("memory", plan)
            if plan["action"] == "candidates":
                return self._candidates_result(
                    "memory", memory_text, plan["candidates"]
                )
            if plan["action"] == "update":
                target_id = str(plan["target_id"])
                previous = next(
                    (item for item in existing if str(item.get("id")) == target_id),
                    plan.get("target") or {},
                )
                stored = await self._memory().async_replace(
                    target_id,
                    memory_text,
                    category=category_text or None,
                    set_category=bool(category_text),
                )
                replaced_text = str(previous.get("text") or "")
            else:
                stored = await self._memory().remember(
                    memory_text,
                    category=category,
                )
                replaced_text = None
        except Exception as err:
            _LOGGER.error("Failed to store memory: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to store memory: {err}"}],
                "isError": True,
            }

        self.publish_progress(
            "tool_complete",
            "Memory updated" if replaced_text else "Memory stored",
            tool="remember_memory",
            memory_id=stored["id"],
        )
        self.hass.async_create_task(
            get_assist_runtime(self.hass).embeddings.async_embed_new_memory(
                stored["id"],
                stored["text"],
            )
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": format_remember_result(stored, replaced_text=replaced_text),
                }
            ],
            "memory": stored,
            "stored": True,
        }

    async def tool_recall_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Recall stored memories by query or category."""
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=5,
            minimum=1,
            maximum=50,
        )
        query = args.get("query")
        category = args.get("category")

        self.publish_progress(
            "tool_start",
            "Searching stored memories",
            tool="recall_memories",
        )

        try:
            result = await self._memory().recall(
                query=None if query is None else str(query),
                category=None if category is None else str(category),
                limit=limit,
            )
        except Exception as err:
            _LOGGER.error("Failed to recall memories: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to recall memories: {err}"}],
                "isError": True,
            }

        items = result["items"]
        self.publish_progress(
            "tool_complete",
            "Memory recall complete",
            tool="recall_memories",
            count=result["returned_count"],
            total=result["total_found"],
        )

        if not items:
            return {
                "content": [{"type": "text", "text": "No stored memories matched."}],
                "memories": [],
                "result_count": 0,
            }

        header = (
            f"Found {result['returned_count']} of {result['total_found']} stored memories:"
            if result["remaining_count"] > 0
            else f"Found {result['returned_count']} stored memories:"
        )
        lines = [header]
        for memory in items:
            category_text = (
                f" [{memory['category']}]" if memory.get("category") else ""
            )
            lines.append(f"- {memory['id']}{category_text}: {memory['text']}")
        if result["remaining_count"] > 0:
            lines.append(f"{result['remaining_count']} more memories matched but were not shown.")

        return {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            "memories": items,
            "result_count": result["total_found"],
        }

    async def _score_items(
        self,
        query: str,
        items: list,
    ) -> dict[str, float] | None:
        """Score items against a query. None means the embedding model is not ready."""
        if not query or not items:
            return {}
        try:
            return await get_assist_runtime(self.hass).embeddings.async_similarities(
                query, items
            )
        except Exception as err:
            _LOGGER.warning("Similarity scoring failed, using stem match only: %s", err)
            return None

    async def _forget_across_stores(
        self,
        *,
        memory_id: str | None,
        instruction_id: str | None,
        query: str | None,
        category: str | None,
        delete_all: bool,
        search_memories: bool,
        search_personality: bool,
    ) -> Dict[str, Any]:
        """Delete matching memories and, when the id is unknown, personality lines too."""
        settings = read_memory_settings(self.hass)
        deleted_memories: list = []
        deleted_personality: list = []
        searched_memories = search_memories and self._memory_tools_enabled()
        searched_personality = search_personality and self._personality_enabled()

        if memory_id:
            deleted_memories = await self._memory().async_delete_ids({memory_id})
            searched_personality = False
            searched_memories = True
        elif searched_memories and (query or category):
            memories = await self._memory().async_list_raw()
            similarities = await self._score_items(query or "", memories) if query else None
            matches = select_forget_matches(
                memories,
                query,
                category=category,
                delete_all=delete_all,
                similarities=similarities,
                threshold=settings.forget_threshold,
            )
            deleted_memories = await self._memory().async_delete_ids(
                {str(item["id"]) for item in matches}
            )

        if instruction_id:
            deleted_personality = await self._personality().async_delete_ids({instruction_id})
            searched_personality = True
            if not query:
                searched_memories = False
        elif searched_personality and query:
            instructions = await self._personality().async_view()
            similarities = await self._score_items(query, instructions)
            matches = select_forget_matches(
                instructions,
                query,
                delete_all=delete_all,
                similarities=similarities,
                threshold=settings.forget_threshold,
            )
            deleted_personality = await self._personality().async_delete_ids(
                {str(item["id"]) for item in matches}
            )

        message = format_forget_result(
            deleted_memories,
            deleted_personality,
            searched_personality=searched_personality,
            searched_memories=searched_memories,
        )
        deleted_count = len(deleted_memories) + len(deleted_personality)
        return {
            "content": [{"type": "text", "text": message}],
            "isError": deleted_count == 0,
            "deleted_count": deleted_count,
            "deleted": deleted_memories,
            "removed_personality": deleted_personality,
        }

    async def tool_forget_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete stored memories, and personality lines when the query is not an id."""
        memory_id = args.get("memory_id")
        query = args.get("query")
        category = args.get("category")
        forget_all_matches = bool(args.get("forget_all_matches", False))
        memory_id_text = None if memory_id is None else str(memory_id).strip()
        query_text = None if query is None else str(query)

        self.publish_progress(
            "tool_start",
            "Deleting stored memory",
            tool="forget_memory",
        )
        if not memory_id_text and not (query_text and query_text.strip()) and not (
            category and str(category).strip()
        ):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "DELETED NOTHING. memory_id, query, or category is required. "
                            "Tell the user that nothing was deleted."
                        ),
                    }
                ],
                "isError": True,
                "deleted_count": 0,
                "deleted": [],
            }

        try:
            result = await self._forget_across_stores(
                memory_id=memory_id_text or None,
                instruction_id=None,
                query=query_text,
                category=None if category is None else str(category),
                delete_all=forget_all_matches,
                search_memories=True,
                search_personality=not memory_id_text,
            )
        except Exception as err:
            _LOGGER.error("Failed to forget memory: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to forget memory: {err}"}],
                "isError": True,
            }

        self.publish_progress(
            "tool_complete",
            "Memory deletion complete",
            tool="forget_memory",
            deleted=result.get("deleted_count", 0),
        )
        return result

    def _format_personality_lines(self, instructions: list) -> str:
        """Format personality instructions for a tool result."""
        if not instructions:
            return "Personality block is empty."
        lines = ["Personality instructions:"]
        for item in instructions:
            lines.append(f"- {item['id']}: {item['text']}")
        return "\n".join(lines)

    async def tool_view_personality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return the persisted personality instructions."""
        del args
        try:
            instructions = await self._personality().async_view()
        except Exception as err:
            _LOGGER.error("Failed to read personality: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to read personality: {err}"}],
                "isError": True,
            }
        return {
            "content": [
                {"type": "text", "text": self._format_personality_lines(instructions)}
            ],
            "instructions": instructions,
        }

    async def tool_set_personality_instruction(
        self, args: Dict[str, Any], context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Add or replace one personality instruction."""
        text = " ".join(str(args.get("instruction") or "").split()).strip()
        instruction_id = args.get("instruction_id")
        instruction_id_text = None if instruction_id is None else str(instruction_id).strip()
        mode = parse_store_mode(args.get("mode"))
        if mode == "invalid":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "NOT STORED. mode must be add, update, or noop. Nothing was written.",
                    }
                ],
                "isError": True,
            }
        if mode is None and instruction_id_text:
            mode = "update"
        conversation_id = (
            context.get("conversation_id") if isinstance(context, dict) else None
        )
        try:
            if mode == "update" and not instruction_id_text:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "NOT STORED. mode=update requires instruction_id. "
                                "Nothing was written."
                            ),
                        }
                    ],
                    "isError": True,
                    "stored": False,
                }
            existing = await self._personality().async_view()
            plan = await self._plan_store_call(
                kind="personality",
                mode=mode,
                new_text=text,
                items=existing,
                requested_id=instruction_id_text,
                conversation_id=None if conversation_id is None else str(conversation_id),
            )
            if plan["action"] == "noop":
                return self._noop_result("personality", plan)
            if plan["action"] == "candidates":
                return self._candidates_result("personality", text, plan["candidates"])
            result = await self._personality().async_set(
                text,
                instruction_id=str(plan["target_id"]) if plan["action"] == "update" else None,
            )
        except Exception as err:
            _LOGGER.error("Failed to update personality: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to update personality: {err}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": result["message"]}],
            "isError": not result.get("ok", False),
            "instructions": result.get("instructions", []),
        }

    async def tool_remove_personality_instruction(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remove personality instructions, and memories about the same subject when the id is unknown."""
        instruction_id = args.get("instruction_id")
        query = args.get("query")
        instruction_id_text = None if instruction_id is None else str(instruction_id).strip()
        query_text = None if query is None else str(query)
        if not instruction_id_text and not (query_text and query_text.strip()):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "DELETED NOTHING. instruction_id or query is required. "
                            "Tell the user that nothing was removed."
                        ),
                    }
                ],
                "isError": True,
            }
        try:
            result = await self._forget_across_stores(
                memory_id=None,
                instruction_id=instruction_id_text or None,
                query=query_text,
                category=None,
                delete_all=False,
                search_memories=not instruction_id_text and self._memory_tools_enabled(),
                search_personality=True,
            )
        except Exception as err:
            _LOGGER.error("Failed to remove personality instruction: %s", err)
            return {
                "content": [
                    {"type": "text", "text": f"Failed to remove personality instruction: {err}"}
                ],
                "isError": True,
            }
        return result

    async def tool_merge_duplicate_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge near-duplicates into the newest entry of each group."""
        del args
        try:
            result = await get_assist_runtime(self.hass).embeddings.async_merge_duplicates()
        except Exception as err:
            _LOGGER.warning("Failed to merge duplicate memories: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to merge duplicates: {err}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": result.get("message") or "Nothing was merged."}],
            "isError": not result.get("ok", False),
            "memory_groups": result.get("memory_groups", []),
            "personality_groups": result.get("personality_groups", []),
        }

    async def tool_reset_personality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clear the personality block when confirm is true."""
        if not bool(args.get("confirm")):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Pass confirm=true to erase every personality instruction.",
                    }
                ],
                "isError": True,
            }
        try:
            result = await self._personality().async_reset()
        except Exception as err:
            _LOGGER.error("Failed to reset personality: %s", err)
            return {
                "content": [{"type": "text", "text": f"Failed to reset personality: {err}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": result["message"]}],
            "removed_count": result.get("removed_count", 0),
        }

    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if client IP is in the allowed list.

        Handles various formats:
        - IPv4: 192.168.1.7
        - IPv4 with port: 192.168.1.7:12345
        - IPv6: ::1 or 2001:db8::1
        - IPv6 with port: [2001:db8::1]:8080
        - CIDR ranges: 172.30.0.0/16, 192.168.1.0/24
        """
        if not self.allowed_ips:
            # If no IPs configured, allow all (backward compatible)
            return True

        if not client_ip:
            return False

        # Extract IP from various formats
        ip_only = client_ip

        # Handle IPv6 with port: [2001:db8::1]:8080 -> 2001:db8::1
        if ip_only.startswith("["):
            end_bracket = ip_only.find("]")
            if end_bracket > 0:
                ip_only = ip_only[1:end_bracket]
        # Handle IPv4 with port: 192.168.1.7:12345 -> 192.168.1.7
        # Only split on single colon (not IPv6 which has multiple colons)
        elif ip_only.count(":") == 1:
            ip_only = ip_only.split(":")[0]
        # Else: IPv6 without port (::1) or IPv4 without port - use as-is

        # Convert to IP address object for CIDR checking
        try:
            client_ip_obj = ipaddress.ip_address(ip_only)
        except ValueError:
            _LOGGER.warning("Invalid client IP format: %s", ip_only)
            return False

        # Check if client IP matches any allowed IP or CIDR range
        for allowed_entry in self.allowed_ips:
            # Check for exact IP match first (backward compatible)
            if ip_only == allowed_entry:
                return True

            # Check if it's a CIDR range
            if "/" in allowed_entry:
                try:
                    network = ipaddress.ip_network(allowed_entry, strict=False)
                    if client_ip_obj in network:
                        return True
                except ValueError:
                    # Invalid CIDR format, skip
                    _LOGGER.warning(
                        "Invalid CIDR format in allowed IPs: %s", allowed_entry
                    )
                    continue

        return False

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        client_ip = request.remote
        _LOGGER.info("🏥 Health check from %s", client_ip)

        # Same whitelist as every other route — the health endpoint otherwise
        # leaks server presence and metadata to unauthorized clients
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked health check from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        health_info = {
            "status": "healthy",
            "server": MCP_SERVER_NAME,
            "port": self.port,
            "version": "0.1.0",
            "endpoints": {
                "websocket": f"ws://<host>:{self.port}/ws",
                "http": f"http://<host>:{self.port}/",
                "health": f"http://<host>:{self.port}/health",
            },
            "tools_available": len(await self._get_tools_list()),
            "external_custom_tools_enabled": self._external_custom_tools_enabled(),
            "timestamp": dt_util.now().isoformat(),
        }
        if self.custom_tools:
            get_loaded_builtin_tool_info = getattr(
                self.custom_tools,
                "get_loaded_builtin_tool_info",
                None,
            )
            if callable(get_loaded_builtin_tool_info):
                health_info["built_in_tool_packages"] = get_loaded_builtin_tool_info()
            health_info["external_custom_tools_loaded"] = (
                self.custom_tools.get_loaded_external_tool_info()
            )
        return web.json_response(health_info)

    async def handle_external_tool_diagnostics(
        self,
        request: web.Request,
    ) -> web.Response:
        """Return diagnostics for manifest-based custom tool packages."""
        client_ip = request.remote
        _LOGGER.info("🧰 External tool diagnostics request from %s", client_ip)

        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked external tool diagnostics request from unauthorized IP: %s",
                client_ip,
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        diagnostics: dict[str, Any] = {
            "external_custom_tools_enabled": self._external_custom_tools_enabled(),
            "external_packages": [],
            "external_load_errors": [],
        }
        if self.custom_tools:
            get_package_diagnostics = getattr(
                self.custom_tools,
                "get_package_diagnostics",
                None,
            )
            if callable(get_package_diagnostics):
                diagnostics = get_package_diagnostics()

        return web.json_response(diagnostics)

    async def reload_external_custom_tools(self) -> dict[str, Any]:
        """Reload manifest-based custom tool packages and notify MCP clients."""
        if not self.custom_tools:
            return {
                "external_custom_tools_enabled": self._external_custom_tools_enabled(),
                "external_packages": [],
                "external_load_errors": ["Custom tool loader is not initialized"],
            }

        reload_tool_packages = getattr(self.custom_tools, "reload_tool_packages", None)
        if callable(reload_tool_packages):
            diagnostics = await reload_tool_packages()
        else:
            reload_external_tools = getattr(self.custom_tools, "reload_external_tools", None)
            if not callable(reload_external_tools):
                return {
                    "external_custom_tools_enabled": self._external_custom_tools_enabled(),
                    "external_packages": [],
                    "external_load_errors": ["Custom tool reload is not available"],
                }
            diagnostics = await reload_external_tools()

        await self.broadcast_notification("notifications/tools/list_changed")
        return diagnostics

    async def handle_progress_stream(self, request: web.Request) -> web.StreamResponse:
        """SSE endpoint for progress updates during tool execution."""
        client_ip = request.remote
        _LOGGER.info("📊 Progress stream request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked progress stream request from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        # Create a queue for this client
        queue = asyncio.Queue()
        self.progress_queues.add(queue)

        try:
            # Send initial connection message
            data = f"data: {json.dumps({'type': 'connected', 'message': 'Progress stream connected'})}\n\n"
            await response.write(data.encode())

            # Stream progress updates
            while True:
                msg = await queue.get()
                data = f"data: {json.dumps(msg)}\n\n"
                await response.write(data.encode())

        except Exception as e:
            _LOGGER.debug(f"Progress stream closed: {e}")
        finally:
            self.progress_queues.discard(queue)

        return response

    def publish_progress(self, event_type: str, message: str, **kwargs):
        """Publish progress update to all progress SSE clients."""
        import time

        msg = {
            "type": event_type,
            "message": message,
            "timestamp": time.time(),
            **kwargs,
        }

        # Send to all progress clients
        for queue in list(self.progress_queues):
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                _LOGGER.debug("Progress queue full, skipping")

    async def handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle Server-Sent Events for MCP notifications."""
        client_ip = request.remote
        _LOGGER.info("🌊 SSE connection request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked SSE connection from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering
        response.headers["Access-Control-Allow-Origin"] = "*"

        await response.prepare(request)

        # Store this client for notifications
        self.sse_clients.append(response)
        _LOGGER.info("✅ SSE client connected. Total clients: %d", len(self.sse_clients))

        try:
            # Send initial connection confirmation
            await response.write(b": connected\n\n")

            # Send tools list changed notification immediately
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed",
            }
            await response.write(f"data: {json.dumps(notification)}\n\n".encode())
            _LOGGER.info("📤 Sent initial tools/list_changed notification")

            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                await response.write(b": keepalive\n\n")

        except Exception as err:
            _LOGGER.info("📤 SSE client disconnected: %s", err)
        finally:
            if response in self.sse_clients:
                self.sse_clients.remove(response)
            _LOGGER.info("SSE clients remaining: %d", len(self.sse_clients))

        return response

    async def _get_tools_list(self) -> List[Dict[str, Any]]:
        """Get the tools list for health check."""
        tools_result = await self.handle_tools_list()
        return tools_result.get("tools", [])

    async def handle_websocket(self, request: web.Request) -> WebSocketResponse:
        """Handle WebSocket connections for MCP protocol."""
        client_ip = request.remote
        _LOGGER.info("🔌 New MCP WebSocket connection from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning(
                "🚫 Blocked WebSocket connection from unauthorized IP: %s", client_ip
            )
            return web.Response(status=403, text="Forbidden: IP not authorized")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        _LOGGER.info("✅ MCP WebSocket connection established with %s", client_ip)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        if not isinstance(data, dict):
                            await ws.send_str(
                                json.dumps(
                                    {
                                        "jsonrpc": "2.0",
                                        "error": {
                                            "code": -32600,
                                            "message": "Invalid Request: expected a JSON-RPC 2.0 object",
                                        },
                                        "id": None,
                                    }
                                )
                            )
                            continue

                        # Check if it's a notification (no id field)
                        if "id" not in data:
                            await self.process_mcp_notification(data)
                            # No response for notifications
                        else:
                            response = await self.process_mcp_message(data)
                            await ws.send_str(json.dumps(response))
                    except json.JSONDecodeError:
                        await ws.send_str(
                            json.dumps(
                                {"error": {"code": -32700, "message": "Parse error"}}
                            )
                        )
                    except Exception as err:
                        _LOGGER.exception("Error processing MCP message")
                        await ws.send_str(
                            json.dumps(
                                {
                                    "error": {
                                        "code": -32000,
                                        "message": f"Server error: {err}",
                                    }
                                }
                            )
                        )
                elif msg.type == WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error: %s", ws.exception())
                    break

        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("WebSocket handler error")

        return ws

    async def handle_mcp_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP requests with proper JSON-RPC 2.0 protocol."""
        client_ip = request.remote
        _LOGGER.info("📨 MCP HTTP JSON-RPC request from %s", client_ip)

        # Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            _LOGGER.warning("🚫 Blocked MCP request from unauthorized IP: %s", client_ip)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": "Forbidden: IP not authorized",
                    },
                    "id": None,
                },
                status=403,
            )

        request_id = None
        try:
            data = await request.json()

            # A JSON-RPC 2.0 request must be an object; arrays/scalars would
            # otherwise raise AttributeError below and surface as HTTP 500
            if not isinstance(data, dict):
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: body must be a JSON object",
                        },
                        "id": None,
                    },
                    status=400,
                )

            request_id = data.get("id")

            # Validate JSON-RPC 2.0 format
            if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing or invalid jsonrpc field",
                        },
                        "id": request_id,
                    },
                    status=400,
                )

            if "method" not in data:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing method field",
                        },
                        "id": request_id,
                    },
                    status=400,
                )

            # Check if this is a notification (no id field)
            is_notification = "id" not in data

            if is_notification:
                _LOGGER.debug("📮 MCP notification: %s", data.get("method"))
                # Process the notification but don't expect a response
                await self.process_mcp_notification(data)
                # Return 204 No Content for notifications
                return web.Response(status=204)
            else:
                _LOGGER.debug(
                    "📋 MCP method: %s (id: %s)", data.get("method"), request_id
                )
                response = await self.process_mcp_message(data)
                return web.json_response(response)

        except json.JSONDecodeError:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error: invalid JSON"},
                    "id": None,
                },
                status=400,
            )
        except Exception as err:
            _LOGGER.exception("Error processing MCP request")
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal error: {str(err)}"},
                    "id": request_id,
                },
                status=500,
            )

    async def process_mcp_notification(self, data: Dict[str, Any]) -> None:
        """Process MCP notification (no response expected)."""
        method = data.get("method")
        params = data.get("params", {})

        _LOGGER.info("Processing MCP notification: %s", method)

        try:
            # Handle both old and new MCP notification formats
            # Old format: "initialized"
            # New format: "notifications/initialized"
            if method in ("initialized", "notifications/initialized"):
                _LOGGER.info("✅ MCP client initialized successfully")
                # Send tools/list_changed to all SSE clients
                await self.broadcast_notification("notifications/tools/list_changed")
            elif method == "notifications/cancelled":
                # Client cancelled a pending request
                _LOGGER.debug("MCP client cancelled a request")
            else:
                _LOGGER.warning("Unknown notification method: %s", method)
        except Exception as err:
            _LOGGER.exception("Error processing notification %s: %s", method, err)

    async def broadcast_notification(
        self, method: str, params: Dict[str, Any] | None = None
    ) -> None:
        """Send notification to all SSE clients."""
        if not self.sse_clients:
            _LOGGER.debug("No SSE clients to notify for %s", method)
            return

        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params

        data = f"data: {json.dumps(notification)}\n\n".encode()

        # Send to all clients, removing dead ones
        dead_clients = []
        for client in self.sse_clients:
            try:
                await client.write(data)
            except Exception as err:
                _LOGGER.debug("Failed to send to client: %s", err)
                dead_clients.append(client)

        # Remove dead clients
        for client in dead_clients:
            self.sse_clients.remove(client)

        if dead_clients:
            _LOGGER.info("Removed %d dead SSE clients", len(dead_clients))

    async def process_mcp_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP message according to JSON-RPC 2.0 protocol."""
        method = data.get("method")
        params = data.get("params", {})
        msg_id = data.get("id")

        _LOGGER.debug("Processing MCP method: %s (id: %s)", method, msg_id)

        try:
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list()
            elif method == "tools/call":
                result = await self.handle_tool_call(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": msg_id,
                }

            # Always include jsonrpc and id in successful responses
            response = {"jsonrpc": "2.0", "result": result, "id": msg_id}

            return response

        except Exception as err:
            _LOGGER.exception("Error in MCP method %s", method)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error in {method}: {str(err)}",
                },
                "id": msg_id,
            }

    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        _LOGGER.info("🔌 MCP initialize request received")
        return {
            "protocolVersion": "2024-11-05",  # MCP uses date-based versioning
            "capabilities": {
                "tools": {
                    "listChanged": True  # Tell client that tools can change dynamically
                }
            },
            "serverInfo": {"name": MCP_SERVER_NAME, "version": "0.1.0"},
        }

    def _memory_tools_enabled(self) -> bool:
        """Return whether persistent memory tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_MEMORY_TOOLS,
                DEFAULT_ENABLE_MEMORY_TOOLS,
            )
        )

    def _personality_enabled(self) -> bool:
        """Return whether the personality block and its tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_PERSONALITY,
                DEFAULT_ENABLE_PERSONALITY,
            )
        )

    def _memory(self):
        """Shared memory store."""
        return get_assist_runtime(self.hass).memory

    def _personality(self):
        """Shared personality store."""
        return get_assist_runtime(self.hass).personality

    def _personality_tool_definitions(self) -> List[Dict[str, Any]]:
        """Tool schemas for the personality block."""
        return [
            {
                "name": "view_personality",
                "description": "Read the persisted personality and behaviour instructions. These are injected into every reply.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "set_personality_instruction",
                "description": "Add a speaking-style, persona, tone, or form-of-address instruction. Call it once with the instruction only. The server adds it, updates the same instruction, or reports that it is already stored. Do not pass mode or invent an instruction_id. Use this for 'talk to me like a pirate' or 'говори со мной как пират'. Do not store those with remember_memory.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "One short behaviour instruction, under 240 characters.",
                        },
                        "instruction_id": {
                            "type": "string",
                            "description": "Omit this. A real instruction id copied from a tool result. A category name is not an id.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["add", "update", "noop"],
                            "description": "Omit this. The server decides. mode=add does not bypass duplicate checks.",
                        },
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "remove_personality_instruction",
                "description": "Remove personality instructions by id or by a query. Russian inflections match (пиратов matches пират). When the id is unknown, this also deletes stored memories about the same subject. If the user says to forget everything about a subject, put that phrase in query. If the result says DELETED NOTHING, tell the user nothing was removed. Do not claim you forgot it.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "instruction_id": {
                            "type": "string",
                            "description": "Instruction id to remove.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Text contained by the instruction to remove, used when the id is not known.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "reset_personality",
                "description": "Erase every personality instruction. Pass confirm=true. Use this when the user wants the default voice back.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "description": "Must be true to clear the personality block.",
                        }
                    },
                    "required": ["confirm"],
                    "additionalProperties": False,
                },
            },
        ]

    def _coerce_int_arg(
        self, value: Any, *, default: int, minimum: int, maximum: int
    ) -> int:
        """Coerce an integer-like tool argument safely."""
        if value is None:
            parsed = default
        elif isinstance(value, bool):
            parsed = default
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        else:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                parsed = default

        return max(minimum, min(parsed, maximum))

    async def handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        _LOGGER.info("MCP tools/list request received")

        # Get configured max entities limit from system entry
        from .const import DOMAIN, CONF_MAX_ENTITIES_PER_DISCOVERY, DEFAULT_MAX_ENTITIES_PER_DISCOVERY
        max_limit = DEFAULT_MAX_ENTITIES_PER_DISCOVERY
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if entry.source == "system":
                max_limit = entry.data.get(CONF_MAX_ENTITIES_PER_DISCOVERY, DEFAULT_MAX_ENTITIES_PER_DISCOVERY)
                break

        tools = [
            {
                "name": "discover_entities",
                "description": "Find and list Home Assistant entities by criteria like area, floor, label, type, domain, device_class, or current state. Use this to discover what devices are available before trying to control them. Each result already includes the current state with its unit and key attributes (brightness, color temperature, temperatures, position, volume, battery), so you usually do not need get_entity_details afterwards.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": "Type of entity to find (e.g., 'light', 'switch', 'sensor', 'climate')",
                        },
                        "area": {
                            "type": "string",
                            "description": "Area/room name or alias to search in - use names from the areas list provided in your system context (e.g., 'Kitchen', 'Back Garden', 'Living Room'). If the value matches a floor name or alias instead, it will search that floor.",
                        },
                        "floor": {
                            "type": "string",
                            "description": "Floor name or alias to search in (e.g., 'Upstairs', 'Basement', 'Ground Floor'). Check get_index() to see available floors.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Label name to filter by (matches labels assigned directly to entities, their devices, or their areas). Check get_index() to see available labels.",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Home Assistant domain to filter by (e.g., 'light', 'switch', 'climate', 'sensor')",
                        },
                        "state": {
                            "type": "string",
                            "description": "Current state to filter by (e.g., 'on', 'off', 'unavailable')",
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Text that entity name should contain (case-insensitive)",
                        },
                        "device_class": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "Device class to filter by (e.g., 'temperature', 'motion', 'door', 'moisture'). Can be a single string or array of strings for OR logic. Check the index for available device classes per domain.",
                        },
                        "name_pattern": {
                            "type": "string",
                            "description": "Wildcard pattern to match entity IDs (e.g., '*_person_detected', 'sensor.*_ble_area'). Supports * for any characters.",
                        },
                        "inferred_type": {
                            "type": "string",
                            "description": "Inferred entity type from the index (e.g., 'person_detection', 'location_tracking'). The pattern will be looked up from the index's inferred_types. Check get_index() to see available inferred types.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": f"Maximum number of entities to return (default: 20, max: {max_limit})",
                            "default": 20,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_entity_details",
                "description": "Get the full state, all attributes, area, floor, and labels of specific entities. Only needed when discover_entities results do not already contain what you need.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity IDs to get details for",
                        }
                    },
                    "required": ["entity_ids"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_areas",
                "description": "List all areas in the home with their entity counts, floors, and area labels",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_domains",
                "description": "List all available domains with entity counts",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_index",
                "description": "Get the pre-generated system structure index. This index provides a lightweight overview of the Home Assistant system including areas, floors, labels, domains, device classes, people, pets, calendars, zones, automations, and scripts. Call this ONCE at the start of a conversation to understand what exists in the system, then use discover_entities to query specific entities. The index is ~400-800 tokens vs ~15k tokens for a full entity dump.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "perform_action",
                "description": "Control Home Assistant devices by calling services. Use after discovering entities to turn on/off lights, set temperatures, open/close covers, etc.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The domain of the service to call (e.g., 'light', 'switch', 'climate', 'vacuum', 'media_player', etc.)",
                        },
                        "action": {
                            "type": "string",
                            "description": "The service action (e.g., 'turn_on', 'turn_off', 'toggle', 'set_temperature')",
                        },
                        "target": {
                            "type": "object",
                            "description": "Target entities, areas, or devices",
                            "properties": {
                                "entity_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single entity ID or list of entity IDs",
                                },
                                "area_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single area ID or list of area IDs",
                                },
                                "device_id": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Single device ID or list of device IDs",
                                },
                            },
                            "minProperties": 1,
                            "additionalProperties": False,
                        },
                        "data": {
                            "type": "object",
                            "description": PERFORM_ACTION_DATA_DESCRIPTION,
                            "additionalProperties": True,
                        },
                    },
                    "required": ["domain", "action", "target"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "set_conversation_state",
                "description": "Indicate whether you expect a response from the user after your message",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "expecting_response": {
                            "type": "boolean",
                            "description": "true if expecting user response, false if task is complete",
                        }
                    },
                    "required": ["expecting_response"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_script",
                "description": "Execute a Home Assistant script and return its response variables. Use this for scripts that return data (e.g., camera analysis, calculations). Returns the script's response variables.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "script_id": {
                            "type": "string",
                            "description": "The script entity ID (e.g., 'script.llm_camera_analysis' or just 'llm_camera_analysis')",
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables to pass to the script",
                            "additionalProperties": True,
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)",
                            "default": 60,
                        },
                    },
                    "required": ["script_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_automation",
                "description": "Trigger a Home Assistant automation with optional variables. Use this to manually trigger automations.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "automation_id": {
                            "type": "string",
                            "description": "The automation entity ID (e.g., 'automation.notify_on_motion' or just 'notify_on_motion')",
                        },
                        "variables": {
                            "type": "object",
                            "description": "Variables to pass to the automation (available as trigger.variables)",
                            "additionalProperties": True,
                        },
                        "skip_conditions": {
                            "type": "boolean",
                            "description": "Whether to skip the automation's conditions (default: false)",
                            "default": False,
                        },
                    },
                    "required": ["automation_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_entity_history",
                "description": "Get historical state changes for a specific entity over a time period. Shows when the entity changed state with timestamps. Useful for answering questions like 'when did the front door open?' or 'what time did the temperature change?'",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID to get history for (e.g., 'binary_sensor.front_door', 'sensor.temperature')",
                        },
                        "hours": {
                            "type": "integer",
                            "description": "Number of hours of history to retrieve (default: 24, max: 168 for 1 week)",
                            "default": 24,
                            "minimum": 1,
                            "maximum": 168,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of state changes to return (default: 50, max: 100). Most recent changes shown first.",
                            "default": 50,
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["entity_id"],
                    "additionalProperties": False,
                },
            },
        ]

        # Add persistent memory tool definitions if enabled
        if self._memory_tools_enabled():
            tools.extend(
                [
                    {
                        "name": "remember_memory",
                        "description": "Store a fact or preference. Call it once with the fact only. The server stores a new row, updates the same fact, or reports that it is already stored. Do not pass mode or memory_id. Do not invent an id or pass a category as an id. 'Олега зовут хозяином' does not replace 'Меня зовут Олег'. Use this only when the user explicitly asks you to remember something. Do not use it for speaking style; use set_personality_instruction for that.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "memory": {
                                    "type": "string",
                                    "description": "The fact, preference, or instruction to store.",
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Optional short category such as 'preference', 'household', or 'schedule'.",
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["add", "update", "noop"],
                                    "description": "Omit this. The server decides. mode=add does not bypass duplicate checks. After a rejected update it stores nothing.",
                                },
                                "memory_id": {
                                    "type": "string",
                                    "description": "Omit this. A real memory id, copied from a tool result. A category name is not an id.",
                                },
                            },
                            "required": ["memory"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "recall_memories",
                        "description": "Search stored facts by query or category, or list recent facts when no query is given. Relevant facts are already injected into the prompt when they match the current request. Use this when the user asks what you remember, or when those injected facts are not enough.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Optional search text to match against stored memory text.",
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Optional category filter.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of memories to return (default: 5).",
                                    "minimum": 1,
                                    "maximum": 50,
                                    "default": 5,
                                },
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "forget_memory",
                        "description": "Delete stored memories about a subject. Also deletes a matching personality instruction when you pass a query instead of a memory_id, so a style saved in either store is removed. Russian inflections match: query пиратов deletes a memory that says пират. Set forget_all_matches true when the user says to forget everything about a subject (забудь всё про ...). If the result says DELETED NOTHING, tell the user nothing was deleted. Do not claim success.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "memory_id": {
                                    "type": "string",
                                    "description": "Specific memory id to delete.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search text to find a memory to delete when the id is not known.",
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Optional category filter when deleting by query.",
                                },
                                "forget_all_matches": {
                                    "type": "boolean",
                                    "description": "Delete every matching memory and personality instruction, not only the best match. Set true for 'forget everything about X' or 'забудь всё про ...'.",
                                    "default": False,
                                },
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "name": "merge_duplicate_memories",
                        "description": "Merge only near-identical memories and personality instructions (automatic merge threshold, default 0.91) into the newest entry of each group. Related but different facts are left untouched. Use remember_memory mode=update when you decide two related facts should become one. The result lists every removed line.",
                        "inputSchema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                ]
            )

        if self._personality_enabled():
            tools.extend(self._personality_tool_definitions())

        # Add custom tool definitions if enabled
        if self.custom_tools:
            try:
                custom_tool_defs = self.custom_tools.get_tool_definitions()
                tools.extend(custom_tool_defs)
            except Exception as e:
                _LOGGER.error(f"Failed to get custom tool definitions: {e}")

        # nextCursor is optional - omit if not paginating
        return {"tools": tools}

    def bind_speaker(self, conversation_id: str, grant) -> None:
        """Remember the speaker for tool calls that do not carry a context."""
        self._bound_grants[conversation_id or ""] = grant

    def unbind_speaker(self, conversation_id: str) -> None:
        """Drop the speaker bound for one conversation."""
        self._bound_grants.pop(conversation_id or "", None)

    def _load_access_config(self):
        from . import get_system_entry

        source: dict[str, Any] = {}
        system_entry = get_system_entry(self.hass)
        entry = system_entry or self.entry
        if entry is not None:
            source.update(getattr(entry, "data", {}) or {})
            source.update(getattr(entry, "options", {}) or {})
        return load_access_config(source)

    def _grant_for_call(self, context: Any):
        """Grant for this tool call. Unknown when access control is on and no speaker was sent."""
        config = self._tool_config
        if config is None or not config.enabled:
            return None
        payload = context.get("access") if isinstance(context, dict) else None
        if payload:
            return grant_from_payload(payload)
        bound = list(self._bound_grants.values())
        if len(bound) == 1:
            return bound[0]
        if len(bound) > 1:
            grant = bound[0]
            for other in bound[1:]:
                grant = grant.intersect(other)
            return grant
        return grant_for_person(config, "")

    def _begin_tool_access(self, context: Any) -> None:
        self._tool_config = self._load_access_config()
        self._tool_grant = (
            self._grant_for_call(context) if self._tool_config.enabled else None
        )
        self._access_snap = None
        self._access_snap_failed = False

    def _end_tool_access(self) -> None:
        self._tool_grant = None
        self._tool_config = None
        self._access_snap = None
        self._access_snap_failed = False

    def _access_restricted(self) -> bool:
        return self._tool_grant is not None and not is_unrestricted(self._tool_grant)

    def _access_snapshot(self):
        if self._access_snap_failed:
            return None
        if self._access_snap is None:
            try:
                self._access_snap = snapshot_from_hass(self.hass)
            except Exception:
                _LOGGER.exception("Access control could not read entities")
                self._access_snap_failed = True
                return None
        return self._access_snap

    def _prepare_service_call(self, domain: str, service: str, target: Any, data: Any, *, broad_data: bool):
        """Return (target, data, refusal or None). Refusal means do not call the service."""
        if not self._access_restricted():
            return target, data, None
        snap = self._access_snapshot()
        if snap is None or self._tool_config is None or self._tool_grant is None:
            return target, data, refusal_tool_result()
        decision = authorize_action(
            domain,
            service,
            target or {},
            data or {},
            self._tool_grant,
            self._tool_config,
            snap,
            broad_data=broad_data,
        )
        if not decision.allowed:
            return target, data, refusal_tool_result()
        if decision.passthrough or broad_data:
            return target, data, None
        return {"entity_id": list(decision.entity_ids)}, strip_targeting(data if isinstance(data, dict) else {}), None

    async def handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request, enforcing the speaker grant when access control is on."""
        context = params.get("context")
        tool_name = params.get("name")
        self._begin_tool_access(context)
        try:
            if (
                self._access_restricted()
                and self.custom_tools
                and self.custom_tools.is_custom_tool(tool_name)
                and not external_tool_allowed(
                    self._tool_grant,
                    self.custom_tools.is_external_custom_tool(tool_name),
                )
            ):
                return refusal_tool_result()
            return await self._dispatch_tool_call(params)
        finally:
            self._end_tool_access()

    async def _dispatch_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        context = params.get("context")

        _LOGGER.debug(
            "Calling tool: %s with args: %s",
            _sanitize_log_value(tool_name, 100),
            _sanitize_log_value(arguments),
        )

        if tool_name == "discover_entities":
            return await self.tool_discover_entities(arguments)
        elif tool_name == "get_entity_details":
            return await self.tool_get_entity_details(arguments)
        elif tool_name == "list_areas":
            return await self.tool_list_areas()
        elif tool_name == "list_domains":
            return await self.tool_list_domains()
        elif tool_name == "get_index":
            return await self.tool_get_index()
        elif tool_name == "perform_action":
            return await self.tool_perform_action(arguments, context)
        elif tool_name == "set_conversation_state":
            return await self.tool_set_conversation_state(arguments)
        elif tool_name == "run_script":
            return await self.tool_run_script(arguments)
        elif tool_name == "run_automation":
            return await self.tool_run_automation(arguments)
        elif tool_name == "remember_memory":
            return await self.tool_remember_memory(arguments, context)
        elif tool_name == "recall_memories":
            return await self.tool_recall_memories(arguments)
        elif tool_name == "forget_memory":
            return await self.tool_forget_memory(arguments)
        elif tool_name == "merge_duplicate_memories":
            return await self.tool_merge_duplicate_memories(arguments)
        elif tool_name == "view_personality":
            return await self.tool_view_personality(arguments)
        elif tool_name == "set_personality_instruction":
            return await self.tool_set_personality_instruction(arguments, context)
        elif tool_name == "remove_personality_instruction":
            return await self.tool_remove_personality_instruction(arguments)
        elif tool_name == "reset_personality":
            return await self.tool_reset_personality(arguments)
        elif tool_name == "get_entity_history":
            return await self.tool_get_entity_history(arguments)
        else:
            # Check if it's a custom tool
            if self.custom_tools and self.custom_tools.is_custom_tool(tool_name):
                return await self.custom_tools.handle_tool_call(
                    tool_name,
                    arguments,
                    context=context if isinstance(context, dict) else None,
                )
            else:
                # Per MCP spec an unknown tool is a tool-level error result,
                # not a protocol error (raising here surfaced as -32603/HTTP 500)
                return {
                    "content": [
                        {"type": "text", "text": f"❌ Unknown tool: {tool_name}"}
                    ],
                    "isError": True,
                }

    async def tool_discover_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Discover entities based on criteria with progress notifications."""
        # Notify start
        self.publish_progress(
            "tool_start",
            "Starting entity discovery",
            tool="discover_entities",
            args=args,
        )

        self.discovery.begin_resolution_hints()
        entities = await self.discovery.discover_entities(
            entity_type=args.get("entity_type"),
            area=args.get("area"),
            floor=args.get("floor"),
            label=args.get("label"),
            domain=args.get("domain"),
            state=args.get("state"),
            name_contains=args.get("name_contains"),
            limit=args.get("limit", 20),
            device_class=args.get("device_class"),
            name_pattern=args.get("name_pattern"),
            inferred_type=args.get("inferred_type"),
        )
        if self._access_restricted():
            snap = self._access_snapshot()
            entities = (
                []
                if snap is None or self._tool_grant is None or self._tool_config is None
                else filter_discovery_rows(entities, self._tool_grant, self._tool_config, snap)
            )

        # Notify completion
        self.publish_progress(
            "tool_complete",
            f"Discovery complete: found {len(entities)} entities",
            tool="discover_entities",
            count=len(entities),
        )

        # Format results based on whether it's smart discovery or general
        result = self._format_discovery_results(entities, args)
        return self._with_resolution_hints(result)

    def _with_resolution_hints(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Prefix typo-tolerance notes (fuzzy match used, or ambiguity) to a result."""
        hints = self.discovery.resolution_hints()
        content = result.get("content") or []
        if hints and content and content[0].get("type") == "text":
            note = "\n".join(f"Note: {hint}" for hint in hints)
            content[0]["text"] = f"{note}\n{content[0]['text']}"
        return result

    def _display_name(self, entity: Dict[str, Any]) -> str:
        """Entity name plus its aliases, so the model can tell entities apart."""
        state_obj = self.hass.states.get(entity.get("entity_id", ""))
        name = entity.get("name") or (state_obj.name if state_obj else "") or entity.get("entity_id", "")
        return format_display_name(name, entity.get("aliases"))

    def _live_state_text(self, entity: Dict[str, Any]) -> str:
        """Current state with unit and key attributes, read live from HA."""
        state_obj = self.hass.states.get(entity.get("entity_id", ""))
        if state_obj is None:
            return summarize_state(entity.get("state"), entity.get("attributes"))
        return summarize_state(state_obj.state, state_obj.attributes)

    def _format_discovery_results(
        self, entities: List[Dict[str, Any]], args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format discovery results for the LLM, handling both smart and general discovery."""
        if not entities:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No entities found matching the search criteria.",
                    }
                ]
            }

        # Check if this is a smart discovery result (has summary metadata)
        has_summary = entities and entities[0].get("entity_id") == "_summary"

        if has_summary:
            # Smart discovery with grouping
            summary = entities[0]
            actual_entities = entities[1:]

            # Build formatted text
            text_parts = []

            # Add summary header
            query_type = summary.get("query_type", "general")
            query = summary.get("query", "")

            if query_type == "person":
                text_parts.append(f"🧑 Person Discovery: '{query}'")
            elif query_type == "pet":
                text_parts.append(f"🐾 Pet Discovery: '{query}'")
            elif query_type == "area":
                text_parts.append(f"🏠 Area Discovery: '{query}'")
            elif query_type == "aggregate":
                text_parts.append(f"📊 Aggregate Discovery")
            else:
                text_parts.append(f"🔍 Discovery Results")

            text_parts.append(f"Found {summary.get('total_found', 0)} total entities")

            # Group entities by relationship
            primary = [e for e in actual_entities if e.get("relationship") == "primary"]
            related = [e for e in actual_entities if e.get("relationship") != "primary"]

            # Add primary entities
            if primary:
                text_parts.append("\n📍 Primary Entities:")
                for entity in primary:
                    type_desc = (
                        f" ({entity.get('type', '')})" if entity.get("type") else ""
                    )
                    location = []
                    if entity.get("area"):
                        location.append(entity["area"])
                    if entity.get("floor"):
                        location.append(entity["floor"])
                    location_text = f" @ {' / '.join(location)}" if location else ""
                    labels = (
                        f" [Labels: {', '.join(entity['labels'])}]"
                        if entity.get("labels")
                        else ""
                    )
                    text_parts.append(
                        f"  • {entity['entity_id']}: {self._display_name(entity)} - {self._live_state_text(entity)}{type_desc}{location_text}{labels}"
                    )

            # Group related entities by category
            if related:
                categories = {}
                for entity in related:
                    cat = entity.get("relationship", "other")
                    categories.setdefault(cat, []).append(entity)

                text_parts.append("\n🔗 Related Entities:")
                for category, cat_entities in categories.items():
                    # Format category name
                    cat_name = category.replace("_", " ").title()
                    text_parts.append(f"\n  {cat_name}:")
                    for entity in cat_entities:
                        location = []
                        if entity.get("area"):
                            location.append(entity["area"])
                        if entity.get("floor"):
                            location.append(entity["floor"])
                        location_text = f" @ {' / '.join(location)}" if location else ""
                        labels = (
                            f" [Labels: {', '.join(entity['labels'])}]"
                            if entity.get("labels")
                            else ""
                        )
                        text_parts.append(
                            f"    • {entity['entity_id']}: {self._display_name(entity)} - {self._live_state_text(entity)}{location_text}{labels}"
                        )

            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}
        else:
            # General discovery - simple list
            text_parts = [f"Found {len(entities)} entities:"]

            for entity in entities:
                detail_parts = [f"State: {self._live_state_text(entity)}"]
                detail_parts.append(f"Area: {entity.get('area', 'None')}")
                if entity.get("floor"):
                    detail_parts.append(f"Floor: {entity['floor']}")
                if entity.get("labels"):
                    detail_parts.append(f"Labels: {', '.join(entity['labels'])}")
                text_parts.append(
                    f"- {entity['entity_id']}: {self._display_name(entity)} ({', '.join(detail_parts)})"
                )

            return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_entity_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about specific entities."""
        entity_ids = args.get("entity_ids", [])
        if self._access_restricted():
            snap = self._access_snapshot()
            if snap is None or self._tool_grant is None or self._tool_config is None:
                entity_ids = []
            else:
                entity_ids = [
                    entity_id
                    for entity_id in entity_ids
                    if entity_visible(entity_id, self._tool_grant, self._tool_config, snap)
                ]
            if not entity_ids:
                return {"content": [{"type": "text", "text": "No entities found."}]}
        details = await self.discovery.get_entity_details(entity_ids)

        details = _strip_non_json_serializable(details)
        return {"content": [{"type": "text", "text": json.dumps(details, indent=2)}]}

    async def tool_list_areas(self) -> Dict[str, Any]:
        """List all areas."""
        areas = await self.discovery.list_areas()
        if self._access_restricted():
            snap = self._access_snapshot()
            visible: dict[str, int] = {}
            if snap is not None and self._tool_grant is not None and self._tool_config is not None:
                for entity in snap.entities.values():
                    if entity.area_id and entity_visible(
                        entity.entity_id, self._tool_grant, self._tool_config, snap
                    ):
                        visible[entity.area_id] = visible.get(entity.area_id, 0) + 1
            filtered = []
            for area in areas:
                if area.get("id") in visible:
                    copied = dict(area)
                    copied["entity_count"] = visible[area["id"]]
                    filtered.append(copied)
            areas = filtered

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Available areas ({len(areas)}):\n"
                    + "\n".join(
                        [
                            (
                                f"- {area['name']}"
                                + (
                                    f" (Aliases: {', '.join(area['aliases'])})"
                                    if area.get("aliases")
                                    else ""
                                )
                                + (
                                    f" (Floor: {area['floor']})"
                                    if area.get("floor")
                                    else ""
                                )
                                + (
                                    f" [Labels: {', '.join(area['labels'])}]"
                                    if area.get("labels")
                                    else ""
                                )
                                + f": {area['entity_count']} entities"
                            )
                            for area in areas
                        ]
                    ),
                }
            ]
        }

    async def tool_list_domains(self) -> Dict[str, Any]:
        """List all domains with entity counts and support status."""
        # Get domains that have entities in this HA instance
        entity_domains = await self.discovery.list_domains()
        if self._access_restricted():
            snap = self._access_snapshot()
            counts: dict[str, int] = {}
            if snap is not None and self._tool_grant is not None and self._tool_config is not None:
                for entity in snap.entities.values():
                    if entity_visible(entity.entity_id, self._tool_grant, self._tool_config, snap):
                        counts[entity.domain] = counts.get(entity.domain, 0) + 1
            entity_domains = [
                {"domain": domain, "count": count}
                for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ]
        entity_domain_map = {d["domain"]: d["count"] for d in entity_domains}

        # Get all supported domains from registry
        supported_domains = get_supported_domains()
        controllable_domains = get_domains_by_type(TYPE_CONTROLLABLE)
        read_only_domains = get_domains_by_type(TYPE_READ_ONLY)

        # Build comprehensive list
        result_text = f"Home Assistant Domains (Entities: {len(entity_domains)}, Supported: {len(supported_domains)}):\n\n"

        # Show domains with entities
        result_text += "📊 Domains with entities in your system:\n"
        for domain in entity_domains:
            support_status = "✅" if domain["domain"] in supported_domains else "⚠️"
            result_text += (
                f"  {support_status} {domain['domain']}: {domain['count']} entities\n"
            )

        # Show supported domains without entities
        result_text += "\n🔧 Additional supported domains (no entities found):\n"
        for domain in supported_domains:
            if domain not in entity_domain_map:
                domain_type = (
                    "controllable"
                    if domain in controllable_domains
                    else "read-only"
                    if domain in read_only_domains
                    else "service"
                )
                result_text += f"  ✅ {domain} ({domain_type})\n"

        result_text += f"\n📈 Summary:\n"
        result_text += f"  - Total entity domains: {len(entity_domains)}\n"
        result_text += f"  - Supported domains: {len(supported_domains)}\n"
        result_text += f"  - Controllable: {len(controllable_domains)}\n"
        result_text += f"  - Read-only: {len(read_only_domains)}\n"

        return {"content": [{"type": "text", "text": result_text}]}

    async def tool_get_index(self) -> Dict[str, Any]:
        """Get the pre-generated system structure index."""
        from .const import DOMAIN

        # Get index manager from hass.data
        index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")

        if not index_manager:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Index manager not available. This feature requires MCP Assist 0.5.0 or later.",
                    }
                ]
            }

        # Get the index
        index = await index_manager.get_index()
        if self._access_restricted():
            snap = self._access_snapshot()
            if snap is None or self._tool_grant is None or self._tool_config is None:
                index = {}
            else:
                index = filter_index(index, self._tool_grant, self._tool_config, snap)

        # Format as JSON for structured consumption
        return {"content": [{"type": "text", "text": json.dumps(index, indent=2)}]}

    async def tool_perform_action(
        self, args: Dict[str, Any], context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Perform an action and report whether entity state actually changed."""
        domain = args.get("domain")
        action = args.get("action")
        target = args.get("target", {})
        data = args.get("data", {})
        if not isinstance(data, dict):
            data = {}

        # Validate required parameters
        if not domain:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'domain'. Use discover_entities to find the correct domain.",
                    }
                ]
            }

        if not action:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Error: Missing required parameter 'action'. Common actions: turn_on, turn_off, toggle.",
                    }
                ]
            }

        _LOGGER.info(
            "🎯 Performing action: %s.%s on %s",
            _sanitize_log_value(domain, 100),
            _sanitize_log_value(action, 100),
            _sanitize_log_value(target, 200),
        )

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Performing action: {domain}.{action}",
            tool="perform_action",
            domain=domain,
            action=action,
        )

        # Validate the service and get the correct service name
        try:
            service = self.validate_service(domain, action)
        except ValueError as err:
            error_msg = str(err)
            _LOGGER.error(f"Service validation error: {error_msg}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        target, data, refusal = self._prepare_service_call(
            domain, service, target, data, broad_data=False
        )
        if refusal is not None:
            return refusal

        # Resolve target (convert areas to entity_ids if needed)
        try:
            resolved_target = await self.resolve_target(target)
            _LOGGER.debug(f"Resolved target: {resolved_target}")
        except Exception as err:
            error_msg = f"Failed to resolve target: {err}"
            _LOGGER.error(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

        # Deprecated color_temp (mireds) was removed from light.turn_on in HA
        # 2026.3. Auto-convert instead of rejecting so the LLM doesn't need a
        # retry round-trip: 100-1000 is a plausible mired value; above that the
        # model used kelvin under the wrong key.
        if domain == "light" and "color_temp" in data:
            raw_value = data.pop("color_temp")
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                numeric = None

            if numeric and 100 <= numeric <= 1000:
                data.setdefault("color_temp_kelvin", round(1_000_000 / numeric))
                _LOGGER.debug(
                    "🔍 Converted deprecated color_temp %s mireds to %s K",
                    raw_value,
                    data["color_temp_kelvin"],
                )
            elif numeric and numeric > 1000:
                data.setdefault("color_temp_kelvin", round(numeric))
                _LOGGER.debug(
                    "🔍 Treated color_temp value %s as kelvin", raw_value
                )
            else:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "❌ Error: color_temp is deprecated. Use color_temp_kelvin instead. "
                            "Examples: 2700 (warm white), 4000 (neutral white), 6500 (cool white). "
                            "Lower Kelvin values = warmer light, higher Kelvin values = cooler light.",
                        }
                    ]
                }

        try:
            return await self._execute_verified_action(
                domain=domain,
                action=action,
                service=service,
                data=data,
                resolved_target=resolved_target,
                context=context,
            )
        except Exception as err:
            error_msg = f"Service call failed: {err}"
            _LOGGER.exception(error_msg)
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

    def _target_entity_ids(self, resolved_target: Dict[str, Any]) -> List[str]:
        """Entity ids named by the target, including entities of a device id."""
        found: List[str] = []
        raw_ids = resolved_target.get("entity_id")
        if isinstance(raw_ids, str):
            found.append(raw_ids)
        elif isinstance(raw_ids, list):
            found.extend(str(entity_id) for entity_id in raw_ids if entity_id)

        device_ids = resolved_target.get("device_id")
        if isinstance(device_ids, str):
            device_ids = [device_ids]
        if isinstance(device_ids, list):
            try:
                registry = er.async_get(self.hass)
                wanted = {str(device_id) for device_id in device_ids if device_id}
                for entry in registry.entities.values():
                    if getattr(entry, "device_id", None) in wanted:
                        found.append(entry.entity_id)
            except Exception:
                _LOGGER.debug("Could not expand device_id to entities", exc_info=True)

        unique: List[str] = []
        for entity_id in found:
            if entity_id not in unique:
                unique.append(entity_id)
        return unique

    def _snapshot_entity(self, entity_id: str) -> Dict[str, Any]:
        """Read one entity, including name and area when the registry has them."""
        state_obj = self.hass.states.get(entity_id)
        if state_obj is None:
            return snapshot_from_mapping(
                {"entity_id": entity_id, "missing": True, "state": "unknown"}
            )
        area = ""
        aliases: List[str] = []
        try:
            entry = er.async_get(self.hass).async_get(entity_id)
            if entry is not None:
                aliases = [
                    str(alias)
                    for alias in (getattr(entry, "aliases", None) or [])
                    if alias
                ]
                area_id = getattr(entry, "area_id", None)
                if not area_id and getattr(entry, "device_id", None):
                    device = dr.async_get(self.hass).async_get(entry.device_id)
                    area_id = getattr(device, "area_id", None) if device else None
                if area_id:
                    area_entry = ar.async_get(self.hass).async_get_area(area_id)
                    area = area_entry.name if area_entry else ""
        except Exception:
            _LOGGER.debug(
                "Could not read registry context for %s", entity_id, exc_info=True
            )
        attributes = dict(getattr(state_obj, "attributes", None) or {})
        return snapshot_from_mapping(
            {
                "entity_id": entity_id,
                "state": state_obj.state,
                "name": getattr(state_obj, "name", None)
                or attributes.get("friendly_name")
                or entity_id,
                "friendly_name": attributes.get("friendly_name"),
                "area": area,
                "aliases": aliases,
                "attributes": attributes,
                "last_updated": getattr(state_obj, "last_updated", None),
                "last_reported": getattr(state_obj, "last_reported", None),
                "missing": False,
            }
        )

    async def _wait_for_state_reports(
        self,
        entity_ids: List[str],
        before: Dict[str, Dict[str, Any]],
        timeout: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Poll until a snapshot changes or the wait expires."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        latest = {entity_id: before[entity_id] for entity_id in entity_ids}
        while True:
            latest = {
                entity_id: self._snapshot_entity(entity_id) for entity_id in entity_ids
            }
            if any(
                observation_changed(before[entity_id], latest[entity_id])
                for entity_id in entity_ids
            ):
                return latest
            remaining = deadline - loop.time()
            if remaining <= 0:
                return latest
            await asyncio.sleep(min(POLL_SECONDS, remaining))

    async def _execute_verified_action(
        self,
        *,
        domain: str,
        action: str,
        service: str,
        data: Dict[str, Any],
        resolved_target: Dict[str, Any],
        context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        """Call the service only when we can still confirm the outcome."""
        entity_ids = self._target_entity_ids(resolved_target)[:10]
        before = {
            entity_id: self._snapshot_entity(entity_id) for entity_id in entity_ids
        }
        live_ids = [
            entity_id
            for entity_id, snapshot in before.items()
            if not is_dead_snapshot(snapshot)
        ]
        if entity_ids and not live_ids:
            rows = [
                {"status": "not_done", "before": snapshot, "after": snapshot}
                for snapshot in before.values()
            ]
            self._publish_action_finished(domain, service, success=False)
            return self._verification_result(rows)

        guard = guard_empty_activation(
            service,
            data,
            [before[entity_id] for entity_id in live_ids],
            facts_from_context(context),
        )
        if guard:
            self._publish_action_finished(domain, service, success=False)
            return {
                "content": [{"type": "text", "text": guard}],
                "isError": True,
                "action_result": {"status": "not_done", "entities": []},
            }

        call_target = dict(resolved_target)
        if "entity_id" in call_target and live_ids:
            call_target["entity_id"] = live_ids
        service_data = {**call_target, **data}

        supports = self.hass.services.supports_response(domain, service)
        wants_response = supports is not SupportsResponse.NONE
        response = await self.hass.services.async_call(
            domain=domain,
            service=service,
            service_data=service_data,
            blocking=True,
            return_response=wants_response,
        )

        if not entity_ids:
            self._publish_action_finished(
                domain, service, success=bool(wants_response and response)
            )
            if wants_response and response:
                response = _strip_non_json_serializable(response)
                text = (
                    "SERVICE RESPONSE. This is data from the service, "
                    "not a confirmed device change.\n"
                    + json.dumps(response, indent=2)
                )
                return {"content": [{"type": "text", "text": text}]}
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "NOT CONFIRMED\n"
                            "No entity state was available to confirm this call. "
                            "Do not say the action was done."
                        ),
                    }
                ],
                "isError": True,
                "action_result": {"status": "not_confirmed", "entities": []},
            }

        timeout = (
            CONFIRM_WAIT_SECONDS
            if any(
                expects_change(service, data, before[entity_id])
                for entity_id in live_ids
            )
            else SETTLED_WAIT_SECONDS
        )
        after = await self._wait_for_state_reports(live_ids, before, timeout)
        rows = []
        for entity_id, snapshot in before.items():
            if is_dead_snapshot(snapshot):
                rows.append(
                    {"status": "not_done", "before": snapshot, "after": snapshot}
                )
                continue
            later = after.get(entity_id, snapshot)
            rows.append(
                {
                    "status": judge_entity(service, data, snapshot, later),
                    "before": snapshot,
                    "after": later,
                }
            )
        result = self._verification_result(rows)
        if wants_response and response:
            response = _strip_non_json_serializable(response)
            result["content"][0]["text"] += "\n\nResponse:\n" + json.dumps(
                response, indent=2
            )
        self._publish_action_finished(domain, service, success=not result.get("isError"))
        return result

    def _publish_action_finished(self, domain: str, service: str, *, success: bool) -> None:
        self.publish_progress(
            "tool_complete",
            f"Action completed: {domain}.{service}",
            tool="perform_action",
            success=success,
        )

    def _verification_result(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        text, is_error, status = render_verification(rows)
        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "action_result": {
                "status": status,
                "entities": [
                    {
                        "entity_id": (row.get("before") or {}).get("entity_id"),
                        "status": row.get("status"),
                        "before": public_snapshot(row.get("before") or {}),
                        "after": public_snapshot(row.get("after") or {}),
                    }
                    for row in rows
                ],
            },
        }
        if is_error:
            result["isError"] = True
        return result

    async def tool_set_conversation_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set whether the assistant expects a user response."""
        expecting_response = args.get("expecting_response", False)

        # Log the state for debugging
        _LOGGER.info(
            f"🔄 Conversation state set: expecting_response={expecting_response}"
        )

        # Return a marker that the agent can detect
        return {
            "content": [
                {"type": "text", "text": f"conversation_state:{expecting_response}"}
            ]
        }

    def _conversation_exposure_error(
        self, entity_id: str, kind: str
    ) -> Dict[str, Any] | None:
        """Return an error result if entity_id is not exposed to conversation.

        Discovery only surfaces exposed entities; without the same gate here a
        whitelisted MCP client could run non-exposed privileged scripts or
        automations by guessing their IDs.
        """
        if async_should_expose(self.hass, "conversation", entity_id):
            return None

        _LOGGER.warning(
            "🚫 Blocked %s run for non-exposed entity: %s", kind, entity_id
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"❌ Error: {entity_id} is not exposed to the "
                    "conversation assistant. Expose it in Home Assistant "
                    "(Settings → Voice assistants → Expose) to allow running it.",
                }
            ]
        }

    async def tool_run_script(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Home Assistant script and return its response variables."""
        script_id = args.get("script_id")
        variables = args.get("variables", {})
        timeout = args.get("timeout", 60)

        # Extract script name (remove script. prefix if present)
        script_name = script_id.replace("script.", "")
        full_script_id = f"script.{script_name}"

        # Scripts must be exposed to conversation, same as discovery — without
        # this check any whitelisted MCP client can run privileged scripts
        exposure_error = self._conversation_exposure_error(full_script_id, "script")
        if exposure_error:
            return exposure_error

        _ignored_target, _ignored_data, refusal = self._prepare_service_call(
            "script",
            script_name,
            {"entity_id": full_script_id},
            variables if isinstance(variables, dict) else {},
            broad_data=True,
        )
        if refusal is not None:
            return refusal

        _LOGGER.info(
            "📜 Running script: %s with variables: %s",
            _sanitize_log_value(full_script_id, 100),
            _sanitize_log_value(variables),
        )

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Running script: {full_script_id}",
            tool="run_script",
            script_id=full_script_id,
        )

        try:
            # Call the script directly as a service (not script.turn_on)
            # Variables go directly in service_data, not nested
            response = await asyncio.wait_for(
                self.hass.services.async_call(
                    domain="script",
                    service=script_name,  # Call script directly
                    service_data=variables,  # Variables go directly here
                    blocking=True,
                    return_response=True,
                ),
                timeout=timeout,
            )

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Script completed: {full_script_id}",
                tool="run_script",
                success=True,
            )

            # Format the response
            result_text = f"✅ Script {full_script_id} completed successfully"

            # If the script returned response variables, include them
            if response:
                result_text += f"\n\nResponse:\n{json.dumps(response, indent=2)}"
                return {
                    "content": [{"type": "text", "text": result_text}],
                    "response": response,
                }
            else:
                result_text += "\n\nNo response variables returned (script may not have response_variable defined)"
                return {"content": [{"type": "text", "text": result_text}]}

        except asyncio.TimeoutError:
            error_msg = f"Script execution timed out after {timeout} seconds"
            _LOGGER.error(f"❌ {error_msg}: {full_script_id}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}
        except Exception as err:
            error_msg = f"Script execution failed: {err}"
            _LOGGER.exception(f"❌ {error_msg}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

    async def tool_run_automation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a Home Assistant automation with optional variables."""
        automation_id = args.get("automation_id")
        variables = args.get("variables", {})
        skip_conditions = args.get("skip_conditions", False)

        # Normalize automation_id (add automation. prefix if missing)
        if not automation_id.startswith("automation."):
            automation_id = f"automation.{automation_id}"

        # Automations must be exposed to conversation, same as discovery
        exposure_error = self._conversation_exposure_error(
            automation_id, "automation"
        )
        if exposure_error:
            return exposure_error

        _ignored_target, _ignored_data, refusal = self._prepare_service_call(
            "automation",
            "trigger",
            {"entity_id": automation_id},
            variables if isinstance(variables, dict) else {},
            broad_data=True,
        )
        if refusal is not None:
            return refusal

        _LOGGER.info(
            "🤖 Triggering automation: %s with variables: %s, skip_conditions: %s",
            _sanitize_log_value(automation_id, 100),
            _sanitize_log_value(variables),
            skip_conditions,
        )

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Triggering automation: {automation_id}",
            tool="run_automation",
            automation_id=automation_id,
        )

        try:
            # Trigger the automation
            await self.hass.services.async_call(
                domain="automation",
                service="trigger",
                service_data={
                    "entity_id": automation_id,
                    "variables": variables,
                    "skip_condition": skip_conditions,
                },
                blocking=True,
            )

            # Notify completion
            self.publish_progress(
                "tool_complete",
                f"Automation triggered: {automation_id}",
                tool="run_automation",
                success=True,
            )

            result_text = f"✅ Automation {automation_id} triggered successfully"
            if skip_conditions:
                result_text += " (conditions skipped)"

            return {"content": [{"type": "text", "text": result_text}]}

        except Exception as err:
            error_msg = f"Automation trigger failed: {err}"
            _LOGGER.exception(f"❌ {error_msg}")
            return {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}

    async def tool_get_entity_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get entity history with human-readable formatting."""
        entity_id = args.get("entity_id")
        hours = min(args.get("hours", 24), 168)  # Max 1 week
        limit = min(args.get("limit", 50), 100)  # Max 100 changes

        _LOGGER.info(f"📜 Getting history for {entity_id}: {hours} hours, limit {limit}")
        if self._access_restricted():
            snap = self._access_snapshot()
            visible = (
                snap is not None
                and self._tool_grant is not None
                and self._tool_config is not None
                and entity_visible(entity_id, self._tool_grant, self._tool_config, snap)
            )
            if not visible:
                return {
                    "content": [
                        {"type": "text", "text": f"Entity '{entity_id}' not found."}
                    ]
                }

        # Notify start
        self.publish_progress(
            "tool_start",
            f"Retrieving history for {entity_id}",
            tool="get_entity_history",
            entity_id=entity_id,
        )

        # 1. Get current state
        current_state = self.hass.states.get(entity_id)
        if not current_state:
            return {
                "content": [
                    {"type": "text", "text": f"Entity '{entity_id}' not found."}
                ]
            }

        friendly_name = current_state.attributes.get("friendly_name", entity_id)

        # 2. Calculate time range (UTC)
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # 3. Query history on the recorder's own executor. The previous
        # positional call put True into no_attributes AND descending, so
        # results were unintentionally newest-first.
        try:
            states = await get_instance(self.hass).async_add_executor_job(
                partial(
                    history.state_changes_during_period,
                    self.hass,
                    start_time,
                    end_time,
                    entity_id,
                    no_attributes=True,  # performance; attributes unused below
                    include_start_time_state=True,
                )
            )
            entity_states = states.get(entity_id, [])
        except Exception as e:
            _LOGGER.error(f"Failed to get history for {entity_id}: {e}")
            return {
                "content": [
                    {"type": "text", "text": f"Failed to retrieve history: {str(e)}"}
                ]
            }

        # Notify completion
        self.publish_progress(
            "tool_complete",
            f"History retrieved: {len(entity_states)} changes",
            tool="get_entity_history",
            success=True,
        )

        # 4. Format history (most recent first, limited)
        if not entity_states:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"{friendly_name} ({entity_id})\nCurrent state: {current_state.state}\n\nNo history available for the last {hours} hours.",
                    }
                ]
            }

        # Reverse to get most recent first, apply limit
        recent_states = list(reversed(entity_states[-limit:]))

        # Build formatted text
        text_parts = [
            f"{friendly_name} ({entity_id})",
            f"Current state: {current_state.state}",
            "",
            f"Recent history (last {hours} hours):",
        ]

        now = dt_util.utcnow()
        for state in recent_states:
            # Calculate relative time
            time_diff = now - state.last_changed
            seconds = time_diff.total_seconds()

            if seconds < 60:
                relative = "just now"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                relative = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            elif seconds < 86400:
                hours_ago = int(seconds / 3600)
                relative = f"{hours_ago} hour{'s' if hours_ago != 1 else ''} ago"
            else:
                days = int(seconds / 86400)
                relative = f"{days} day{'s' if days != 1 else ''} ago"

            # Absolute timestamp
            absolute = dt_util.as_local(state.last_changed).strftime("%H:%M:%S")

            # Format line
            text_parts.append(f"• {relative} ({absolute}) → {state.state}")

        text_parts.append("")
        text_parts.append(
            f"Showing {len(recent_states)} change{'s' if len(recent_states) != 1 else ''}"
        )

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    def validate_service(self, domain: str, action: str) -> str:
        """Validate that a domain/action combination is allowed.

        Returns:
            The correct service name to use

        Raises:
            ValueError: If domain or action is invalid
        """
        valid, result = validate_domain_action(domain, action)
        if valid:
            _LOGGER.debug(
                f"Validated service: {domain}.{result} (from action: {action})"
            )
            return result  # Returns the correct service name
        else:
            _LOGGER.warning(f"Service validation failed: {result}")
            raise ValueError(result)  # Returns error message

    async def resolve_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve areas and devices to entity_ids."""
        resolved = {}

        # Pass through entity_id as-is
        if "entity_id" in target:
            entity_id = target["entity_id"]
            # Ensure it's a list for consistency
            if isinstance(entity_id, str):
                entity_id = [entity_id]
            resolved["entity_id"] = entity_id
            _LOGGER.debug(f"Using entity_ids: {entity_id}")

        # Resolve area_id to entities
        if "area_id" in target:
            area_ids = target["area_id"]
            if isinstance(area_ids, str):
                area_ids = [area_ids]

            area_entities = []
            for area_id in area_ids:
                # Get entities in this area
                self.discovery.begin_resolution_hints()
                entities = await self.discovery.get_entities_by_area(area_id)
                if not entities:
                    hints = self.discovery.resolution_hints()
                    if hints:
                        # Ambiguous typo: never act on a guessed area
                        raise ValueError(" ".join(hints))
                area_entities.extend([e["entity_id"] for e in entities])
                _LOGGER.debug(f"Found {len(entities)} entities in area '{area_id}'")

            if area_entities:
                if "entity_id" in resolved:
                    # Merge with existing entity_ids
                    existing = resolved["entity_id"]
                    resolved["entity_id"] = list(set(existing + area_entities))
                else:
                    resolved["entity_id"] = area_entities

        # Pass through device_id if present (HA will handle it)
        if "device_id" in target:
            resolved["device_id"] = target["device_id"]
            _LOGGER.debug(f"Using device_id: {target['device_id']}")

        return resolved if resolved else target
