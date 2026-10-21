"""Smart Entity Discovery System for MCP Server.

This module provides intelligent entity discovery with relationship understanding,
pattern recognition, and structured results optimized for LLM interaction.
"""

import asyncio
import fnmatch
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er, device_registry as dr
from homeassistant.components.homeassistant import async_should_expose

try:
    from homeassistant.helpers import floor_registry as fr
except ImportError:  # pragma: no cover - older Home Assistant versions
    fr = None

try:
    from homeassistant.helpers import label_registry as lr
except ImportError:  # pragma: no cover - older Home Assistant versions
    lr = None

from .action_verify import listing_disposition, prepare_listed_entity
from .const import MAX_ENTITIES_PER_DISCOVERY
from .ha_compat import devices_in_area
from contextvars import ContextVar

from .text_match import (
    any_key_equal,
    best_typo_match,
    key_contains,
    keys_equal,
    typo_contains,
)

# Notes about typo-tolerant lookups made while serving the current tool call
# (fuzzy matches taken, or ambiguous names that were not guessed). Awaiting a
# coroutine keeps the caller's context, so the tool handler reads them back.
_RESOLUTION_HINTS: ContextVar[list[str] | None] = ContextVar(
    "mcp_assist_se_resolution_hints", default=None
)

_LOGGER = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of discovery queries."""
    PERSON = "person"
    PET = "pet"
    DEVICE = "device"
    AREA = "area"
    STATE = "state"
    AGGREGATE = "aggregate"
    GENERAL = "general"


class EntityPattern:
    """Entity pattern definitions for relationship detection."""

    # Person-related patterns
    PERSON_PATTERNS = [
        (r"^person\.{name}$", "primary", "Person entity"),
        (r"^device_tracker\..*{name}.*", "device_tracking", "Device tracker"),
        (r"^sensor\..*{name}.*_ble_area$", "ble_tracking", "BLE area sensor"),
        (r"^sensor\..*{name}.*_ble_room_presence$", "ble_tracking", "BLE room presence"),
        (r"^input_text\.room_{name}$", "room_tracking", "Room tracker"),
        (r"^input_text\.{name}_room$", "room_tracking", "Room tracker"),
        (r"^input_boolean\.{name}_home$", "presence", "Home presence"),
        (r"^input_boolean\.{name}_inside$", "presence", "Inside tracker"),
        (r"^binary_sensor\.{name}_home$", "presence", "Home sensor"),
    ]

    # Pet-related patterns (similar to person but without device_tracker)
    PET_PATTERNS = [
        (r"^person\.{name}$", "primary", "Pet as person entity"),
        (r"^binary_sensor\.{name}$", "primary", "Pet sensor"),
        (r"^sensor\..*{name}.*_ble_area$", "ble_tracking", "BLE area sensor"),
        (r"^input_text\.room_{name}$", "room_tracking", "Room tracker"),
        (r"^input_text\.{name}_room$", "room_tracking", "Room tracker"),
        (r"^input_boolean\.{name}_inside$", "presence", "Inside tracker"),
        (r"^input_boolean\.{name}_home$", "presence", "Home presence"),
    ]


class SmartDiscovery:
    """Smart entity discovery with relationship understanding."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize smart discovery."""
        self.hass = hass
        self._entity_cache = None
        self._cache_time = None

    async def discover_entities(
        self,
        entity_type: Optional[str] = None,
        area: Optional[str] = None,
        floor: Optional[str] = None,
        label: Optional[str] = None,
        domain: Optional[str] = None,
        state: Optional[str] = None,
        name_contains: Optional[str] = None,
        limit: int = 20,
        device_class: Optional[Union[str, List[str]]] = None,
        name_pattern: Optional[str] = None,
        inferred_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Smart entity discovery with relationship understanding.

        This is the main entry point that routes to appropriate discovery strategy.

        Args:
            entity_type: Entity type (deprecated, use domain)
            area: Area name to filter by
            floor: Floor name to filter by
            label: Label name to filter by
            domain: Domain to filter by (e.g., 'sensor', 'light')
            state: State to filter by (e.g., 'on', 'off')
            name_contains: Substring to search for in entity names
            limit: Maximum number of entities to return
            device_class: Device class to filter by (e.g., 'temperature', 'motion')
                         Can be a single string or list of strings for OR logic
            name_pattern: Wildcard pattern to match entity IDs (e.g., '*_person_detected')
            inferred_type: Inferred entity type from the index (e.g., 'person_detection')
                          Looks up the pattern from the index and applies it
        """
        if _RESOLUTION_HINTS.get() is None:
            self.begin_resolution_hints()

        # Detect query type and intent
        query_type = self._detect_query_type(
            entity_type, area or floor, domain, state, name_contains
        )

        # Log the detected query type
        _LOGGER.debug(f"Detected query type: {query_type}, name_contains: {name_contains}")

        # Route to appropriate discovery method
        if query_type == QueryType.PERSON and name_contains and not device_class and not name_pattern and not inferred_type:
            return await self._discover_person_entities(name_contains, limit)
        elif query_type == QueryType.PET and name_contains and not device_class and not name_pattern and not inferred_type:
            return await self._discover_pet_entities(name_contains, limit)
        elif query_type == QueryType.AGGREGATE and not device_class and not name_pattern and not inferred_type and not floor and not label:
            return await self._discover_aggregate_entities(domain, state, limit)
        elif area and not floor and not label and not device_class and not name_pattern and not inferred_type:
            return await self._discover_area_entities(area, domain, state, limit)
        else:
            # Fall back to general discovery (handles device_class, name_pattern, and inferred_type)
            return await self._discover_general_entities(
                entity_type,
                area,
                floor,
                label,
                domain,
                state,
                name_contains,
                limit,
                device_class,
                name_pattern,
                inferred_type,
            )

    def _detect_query_type(
        self,
        entity_type: Optional[str],
        area: Optional[str],
        domain: Optional[str],
        state: Optional[str],
        name_contains: Optional[str]
    ) -> QueryType:
        """Detect the type of query based on parameters."""
        # Check for person/pet queries
        if name_contains:
            name_lower = name_contains.lower()

            # Check if it's likely a person name
            if self._is_likely_person_name(name_lower):
                # Check if it's actually a pet
                if self._is_likely_pet_name(name_lower):
                    return QueryType.PET
                return QueryType.PERSON

            # Check for aggregate queries
            if any(word in name_lower for word in ["who", "anyone", "everyone", "all"]):
                return QueryType.AGGREGATE

        # Area-based query
        if area and not name_contains:
            return QueryType.AREA

        # State-based query
        if state and not name_contains and not area:
            return QueryType.STATE

        # Device type query
        if domain and not name_contains:
            return QueryType.DEVICE

        return QueryType.GENERAL

    def _is_likely_person_name(self, name: str) -> bool:
        """Check if the name is likely a person's name."""
        # Check if we have a person entity with this name
        person_entity = f"person.{name}"
        if self.hass.states.get(person_entity):
            return True

        # Check for device_tracker patterns (often indicates a person)
        for entity_id in self.hass.states.async_entity_ids():
            if f"device_tracker" in entity_id and name in entity_id.lower():
                return True

        # Check for common person-related patterns
        escaped_name = re.escape(name)
        for entity_id in self.hass.states.async_entity_ids():
            if re.match(rf"input_text\.room_{escaped_name}", entity_id):
                return True
            if re.match(rf"sensor\..*{escaped_name}.*_ble_area", entity_id):
                return True

        return False

    def _is_likely_pet_name(self, name: str) -> bool:
        """Check if the name is likely a pet's name.

        Detection logic:
        - Pets typically have tracking entities but no device_tracker
        - Pets often have 'inside' or room tracking but no person entity
        - Generic keywords like 'cat', 'dog', 'pet' indicate pet queries
        """
        name_lower = name.lower()

        # Check for generic pet keywords
        if name_lower in ["cat", "dog", "pet", "puppy", "kitten", "bird", "fish"]:
            return True

        # Check if we have pet-specific entities without person entities
        has_pet_entities = False
        has_person_entity = self.hass.states.get(f"person.{name_lower}") is not None
        has_device_tracker = False

        # Look for pet-specific patterns
        for entity_id in self.hass.states.async_entity_ids():
            entity_lower = entity_id.lower()
            if name_lower in entity_lower:
                # Pet indicators
                if "inside" in entity_lower or "room" in entity_lower:
                    has_pet_entities = True
                # Person indicators
                if "device_tracker" in entity_lower:
                    has_device_tracker = True

        # It's likely a pet if:
        # - Has tracking entities but no device_tracker (pets don't have phones)
        # - Has no person entity
        return has_pet_entities and not has_person_entity and not has_device_tracker

    def _resolve_area_entry(
        self, area_name: str, area_registry: Any, fuzzy: bool = True
    ) -> Any:
        """Resolve an area by name or alias."""
        if not area_name:
            return None

        get_by_name = getattr(area_registry, "async_get_area_by_name", None)
        if get_by_name:
            area_entry = get_by_name(area_name)
            if area_entry:
                return area_entry

        get_by_alias = getattr(area_registry, "async_get_areas_by_alias", None)
        if get_by_alias:
            area_matches = get_by_alias(area_name)
            if area_matches:
                return area_matches[0]

        for area_entry in area_registry.async_list_areas():
            if keys_equal(area_entry.name, area_name):
                return area_entry
            aliases = getattr(area_entry, "aliases", set()) or set()
            if any_key_equal(area_name, aliases):
                return area_entry

        if not fuzzy:
            return None
        return self._typo_fallback(
            "area",
            area_name,
            area_registry.async_list_areas(),
            lambda entry: [entry.name, *sorted(getattr(entry, "aliases", set()) or set())],
        )

    def _resolve_floor_entry(
        self, floor_name: str, floor_registry: Any, fuzzy: bool = True
    ) -> Any:
        """Resolve a floor by name or alias."""
        if not floor_name or floor_registry is None:
            return None

        get_by_name = getattr(floor_registry, "async_get_floor_by_name", None)
        if get_by_name:
            floor_entry = get_by_name(floor_name)
            if floor_entry:
                return floor_entry

        get_by_alias = getattr(floor_registry, "async_get_floors_by_alias", None)
        if get_by_alias:
            floor_matches = get_by_alias(floor_name)
            if floor_matches:
                return floor_matches[0]

        for floor_entry in floor_registry.async_list_floors():
            if keys_equal(floor_entry.name, floor_name):
                return floor_entry
            aliases = getattr(floor_entry, "aliases", set()) or set()
            if any_key_equal(floor_name, aliases):
                return floor_entry

        if not fuzzy:
            return None
        return self._typo_fallback(
            "floor",
            floor_name,
            floor_registry.async_list_floors(),
            lambda entry: [entry.name, *sorted(getattr(entry, "aliases", set()) or set())],
        )

    def _resolve_label_entry(self, label_name: str, label_registry: Any) -> Any:
        """Resolve a label by name."""
        if not label_name or label_registry is None:
            return None

        get_by_name = getattr(label_registry, "async_get_label_by_name", None)
        if get_by_name:
            label_entry = get_by_name(label_name)
            if label_entry:
                return label_entry

        for label_entry in label_registry.async_list_labels():
            if keys_equal(label_entry.name, label_name):
                return label_entry

        return self._typo_fallback(
            "label",
            label_name,
            label_registry.async_list_labels(),
            lambda entry: [entry.name],
        )

    def _resolve_area_or_floor(
        self, name: str, area_registry: Any, floor_registry: Any
    ) -> Tuple[Any, Any]:
        """Resolve a place name to (area, None) or (None, floor).

        Exact (normalized) matches on areas and floors are tried before any
        typo-tolerant match, so a typo never overrides an exact hit.
        """
        area_entry = self._resolve_area_entry(name, area_registry, fuzzy=False)
        if area_entry:
            return area_entry, None
        floor_entry = self._resolve_floor_entry(name, floor_registry, fuzzy=False)
        if floor_entry:
            return None, floor_entry
        area_entry = self._resolve_area_entry(name, area_registry)
        if area_entry:
            return area_entry, None
        return None, self._resolve_floor_entry(name, floor_registry)

    @staticmethod
    def begin_resolution_hints() -> None:
        """Start collecting typo-tolerance notes for the current tool call."""
        _RESOLUTION_HINTS.set([])

    @staticmethod
    def resolution_hints() -> List[str]:
        """Notes collected since begin_resolution_hints()."""
        return list(_RESOLUTION_HINTS.get() or [])

    @staticmethod
    def _add_resolution_hint(hint: str) -> None:
        hints = _RESOLUTION_HINTS.get()
        if hints is not None and hint not in hints:
            hints.append(hint)

    def _typo_fallback(
        self, kind: str, query: str, entries: Any, names: Any
    ) -> Any:
        """Typo-tolerant lookup used only after exact matching found nothing."""
        result = best_typo_match(query, list(entries or []), names)
        if result.match is not None:
            matched_name = result.candidates[0]
            _LOGGER.debug("Typo-tolerant %s match: %r -> %r", kind, query, matched_name)
            self._add_resolution_hint(
                f"No {kind} is named '{query}'; used the closest {kind} '{matched_name}'."
            )
            return result.match
        if result.ambiguous:
            _LOGGER.debug(
                "Ambiguous typo-tolerant %s match for %r: %s", kind, query, result.candidates
            )
            self._add_resolution_hint(
                f"No {kind} is named '{query}'. Did you mean one of: "
                + ", ".join(f"'{name}'" for name in result.candidates)
                + "? Retry with the exact name."
            )
        return None

    def _get_label_names(self, label_ids: Set[str], label_registry: Any) -> List[str]:
        """Resolve label IDs to friendly label names."""
        if not label_ids:
            return []

        label_names = set()
        for label_id in label_ids:
            label_name = label_id
            if label_registry is not None:
                label_entry = label_registry.async_get_label(label_id)
                if label_entry:
                    label_name = label_entry.name
            label_names.add(label_name)

        return sorted(label_names, key=str.casefold)

    def _get_entry_aliases(self, entry: Any) -> List[str]:
        """Get aliases from a registry entry if supported."""
        aliases = getattr(entry, "aliases", set()) if entry is not None else set()
        return sorted(set(aliases or set()), key=str.casefold)

    def _get_entity_aliases(self, entity_entry: Any) -> List[str]:
        """Get resolved entity aliases."""
        if entity_entry is None:
            return []

        get_entity_aliases = getattr(er, "async_get_entity_aliases", None)
        if get_entity_aliases is not None:
            try:
                return get_entity_aliases(self.hass, entity_entry)
            except Exception:  # pragma: no cover - defensive fallback
                _LOGGER.debug("Falling back to raw entity aliases", exc_info=True)

        aliases = getattr(entity_entry, "aliases", []) or []
        return sorted(
            {alias for alias in aliases if isinstance(alias, str)},
            key=str.casefold,
        )

    def _get_entity_context(
        self,
        entity_entry: Any,
        device_registry: Any,
        area_registry: Any,
        floor_registry: Any = None,
        label_registry: Any = None,
        include_label_sources: bool = False,
    ) -> Dict[str, Any]:
        """Resolve area, floor, and label context for an entity."""
        device_entry = None
        if entity_entry and entity_entry.device_id:
            device_entry = device_registry.async_get(entity_entry.device_id)

        area_id = None
        if entity_entry and entity_entry.area_id:
            area_id = entity_entry.area_id
        elif device_entry and device_entry.area_id:
            area_id = device_entry.area_id

        area_entry = area_registry.async_get_area(area_id) if area_id else None

        floor_id = getattr(area_entry, "floor_id", None) if area_entry else None
        floor_name = None
        floor_entry = None
        if floor_id and floor_registry is not None:
            floor_entry = floor_registry.async_get_floor(floor_id)
            if floor_entry:
                floor_name = floor_entry.name

        label_sources = {
            "entity": set(getattr(entity_entry, "labels", set()) or set()),
            "device": set(getattr(device_entry, "labels", set()) or set()),
            "area": set(getattr(area_entry, "labels", set()) or set()),
        }
        label_ids = set().union(*label_sources.values())
        area_aliases = self._get_entry_aliases(area_entry)
        floor_aliases = self._get_entry_aliases(floor_entry)
        device_aliases = self._get_entry_aliases(device_entry)

        context = {
            "area": area_entry.name if area_entry else None,
            "area_id": area_id,
            "area_aliases": area_aliases,
            "floor": floor_name,
            "floor_id": floor_id,
            "floor_aliases": floor_aliases,
            "labels": self._get_label_names(label_ids, label_registry),
            "label_ids": sorted(label_ids),
            "device": (
                (device_entry.name_by_user or device_entry.name)
                if device_entry
                else None
            ),
            "device_name": device_entry.name if device_entry else None,
            "device_name_by_user": device_entry.name_by_user if device_entry else None,
            "device_aliases": device_aliases,
        }

        if include_label_sources:
            context["label_sources"] = {
                source: self._get_label_names(source_ids, label_registry)
                for source, source_ids in label_sources.items()
                if source_ids
            }

        return context

    def _entity_matches_search_term(
        self,
        search_term: str,
        state_obj: Any,
        entity_entry: Any,
        entity_context: Dict[str, Any],
    ) -> bool:
        """Check whether a search term matches an entity or its related context."""
        search_values = {
            state_obj.entity_id,
            state_obj.name,
            state_obj.attributes.get("friendly_name", ""),
            entity_context.get("area"),
            entity_context.get("floor"),
            entity_context.get("device"),
            entity_context.get("device_name"),
            entity_context.get("device_name_by_user"),
        }

        entity_aliases = self._get_entity_aliases(entity_entry)
        if entity_aliases:
            search_values.update(entity_aliases)

        search_values.update(entity_context.get("area_aliases", []))
        search_values.update(entity_context.get("floor_aliases", []))
        search_values.update(entity_context.get("device_aliases", []))
        search_values.update(entity_context.get("labels", []))

        return any(
            value and key_contains(value, search_term)
            for value in search_values
        )

    def _entity_typo_distance(
        self,
        search_term: str,
        state_obj: Any,
        entity_entry: Any,
        entity_context: Dict[str, Any],
    ) -> Optional[int]:
        """Best typo distance between the search term and the entity's names."""
        search_values = {
            state_obj.name,
            state_obj.attributes.get("friendly_name", ""),
            entity_context.get("area"),
            entity_context.get("floor"),
            entity_context.get("device"),
            entity_context.get("device_name"),
            entity_context.get("device_name_by_user"),
        }
        search_values.update(self._get_entity_aliases(entity_entry))
        search_values.update(entity_context.get("area_aliases", []))
        search_values.update(entity_context.get("floor_aliases", []))
        search_values.update(entity_context.get("device_aliases", []))
        search_values.update(entity_context.get("labels", []))
        distances = [
            distance
            for distance in (
                typo_contains(value, search_term) for value in search_values if value
            )
            if distance is not None
        ]
        return min(distances) if distances else None

    async def _discover_person_entities(
        self, name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities related to a person with smart grouping."""
        name_lower = name.lower()
        results = {
            "query": name,
            "query_type": "person",
            "primary_entities": [],
            "related_entities": {
                "device_tracking": [],
                "ble_tracking": [],
                "room_tracking": [],
                "presence": [],
                "other": []
            }
        }

        # Find all entities related to this person
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            entity_id_lower = entity_id.lower()

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Check against patterns
            matched = False
            for pattern, category, description in EntityPattern.PERSON_PATTERNS:
                pattern_regex = pattern.replace("{name}", re.escape(name_lower))
                if re.match(pattern_regex, entity_id_lower):
                    entity_info = self._create_entity_info(state_obj, description)
                    if entity_info is None:
                        continue

                    if category == "primary":
                        results["primary_entities"].append(entity_info)
                    else:
                        results["related_entities"][category].append(entity_info)
                    matched = True
                    break

            # Also check friendly name and aliases
            if not matched:
                # Check friendly name
                if key_contains(state_obj.name, name_lower):
                    entity_info = self._create_entity_info(state_obj)
                    if entity_info is not None:
                        results["related_entities"]["other"].append(entity_info)
                else:
                    # Check entity aliases
                    entity_registry = er.async_get(self.hass)
                    entity_entry = entity_registry.async_get(entity_id)
                    entity_aliases = self._get_entity_aliases(entity_entry)
                    if entity_aliases:
                        for alias in entity_aliases:
                            if key_contains(alias, name_lower):
                                entity_info = self._create_entity_info(state_obj)
                                if entity_info is None:
                                    break
                                entity_info["matched_alias"] = alias
                                results["related_entities"]["other"].append(entity_info)
                                break

        # Format results for return
        return self._format_smart_results(results, limit)

    async def _discover_pet_entities(
        self, name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities related to a pet with smart grouping."""
        name_lower = name.lower()
        results = {
            "query": name,
            "query_type": "pet",
            "primary_entities": [],
            "related_entities": {
                "ble_tracking": [],
                "room_tracking": [],
                "presence": [],
                "other": []
            }
        }

        # Find all entities related to this pet
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            entity_id_lower = entity_id.lower()

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Check against patterns
            matched = False
            for pattern, category, description in EntityPattern.PET_PATTERNS:
                pattern_regex = pattern.replace("{name}", re.escape(name_lower))
                if re.match(pattern_regex, entity_id_lower):
                    entity_info = self._create_entity_info(state_obj, description)
                    if entity_info is None:
                        continue

                    if category == "primary":
                        results["primary_entities"].append(entity_info)
                    else:
                        results["related_entities"][category].append(entity_info)
                    matched = True
                    break

            # Also check friendly name and aliases
            if not matched:
                # Check friendly name
                if key_contains(state_obj.name, name_lower):
                    entity_info = self._create_entity_info(state_obj)
                    if entity_info is not None:
                        results["related_entities"]["other"].append(entity_info)
                else:
                    # Check entity aliases
                    entity_registry = er.async_get(self.hass)
                    entity_entry = entity_registry.async_get(entity_id)
                    entity_aliases = self._get_entity_aliases(entity_entry)
                    if entity_aliases:
                        for alias in entity_aliases:
                            if key_contains(alias, name_lower):
                                entity_info = self._create_entity_info(state_obj)
                                if entity_info is None:
                                    break
                                entity_info["matched_alias"] = alias
                                results["related_entities"]["other"].append(entity_info)
                                break

        # Format results for return
        return self._format_smart_results(results, limit)

    async def _discover_aggregate_entities(
        self, domain: Optional[str], state: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities for aggregate queries like 'who is home'."""
        results = {
            "query": f"{'all ' + domain if domain else 'entities'} {'with state ' + state if state else ''}",
            "query_type": "aggregate",
            "primary_entities": [],
            "related_entities": {}
        }

        # Focus on person and presence entities for "who is home" type queries
        target_domains = ["person", "device_tracker", "binary_sensor"]
        if domain:
            target_domains = [domain]

        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id
            entity_domain = entity_id.split(".")[0]

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Filter by domain
            if entity_domain not in target_domains:
                continue

            # Filter by state if specified
            if state and state_obj.state.lower() != state.lower():
                continue

            # Add to appropriate category
            entity_info = self._create_entity_info(state_obj)
            if entity_info is None:
                continue

            if entity_domain == "person":
                results["primary_entities"].append(entity_info)
            else:
                category = entity_domain.replace("_", " ").title()
                results["related_entities"].setdefault(category, []).append(entity_info)

        return self._format_smart_results(results, limit)

    async def _discover_area_entities(
        self, area: str, domain: Optional[str], state: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Discover entities in a specific area with smart grouping."""
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        # Handle if area is passed as list (defensive)
        if isinstance(area, list):
            if not area:
                return []
            area = area[0]  # Use first area

        # Find area by name (or a floor with that name)
        area_entry, floor_entry = self._resolve_area_or_floor(
            area, area_registry, floor_registry
        )
        area_id = area_entry.id if area_entry else None

        if not area_id:
            if floor_entry:
                return await self._discover_general_entities(
                    entity_type=None,
                    area=None,
                    floor=floor_entry.name,
                    label=None,
                    domain=domain,
                    state=state,
                    name_contains=None,
                    limit=limit,
                )
            return []

        results = {
            "query": f"{area} area",
            "query_type": "area",
            "area_name": area_entry.name,
            "primary_entities": [],
            "related_entities": {}
        }

        # Group entities by domain
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id

            # Check if entity should be exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Get entity's area
            entity_entry = entity_registry.async_get(entity_id)
            entity_area_id = None

            if entity_entry:
                if entity_entry.area_id:
                    entity_area_id = entity_entry.area_id
                elif entity_entry.device_id:
                    device_entry = device_registry.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id:
                        entity_area_id = device_entry.area_id

            # Check if entity is in the target area
            if entity_area_id != area_id:
                continue

            # Apply domain filter
            entity_domain = entity_id.split(".")[0]
            if domain and entity_domain != domain:
                continue

            # Apply state filter
            if state and state_obj.state.lower() != state.lower():
                continue

            # Add to results grouped by domain
            entity_info = self._create_entity_info(
                state_obj,
                entity_entry=entity_entry,
                entity_context=self._get_entity_context(
                    entity_entry,
                    device_registry,
                    area_registry,
                    floor_registry,
                    label_registry,
                ),
            )
            if entity_info is None:
                continue
            domain_key = entity_domain.replace("_", " ").title()
            results["related_entities"].setdefault(domain_key, []).append(entity_info)

        return self._format_smart_results(results, limit)

    async def _discover_general_entities(
        self,
        entity_type: Optional[str],
        area: Optional[str],
        floor: Optional[str],
        label: Optional[str],
        domain: Optional[str],
        state: Optional[str],
        name_contains: Optional[str],
        limit: int,
        device_class: Optional[Union[str, List[str]]] = None,
        name_pattern: Optional[str] = None,
        inferred_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """General entity discovery with enhanced search."""
        # Handle inferred_type by looking up pattern from index
        if inferred_type and not name_pattern:
            from .const import DOMAIN
            index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")
            if index_manager:
                index = await index_manager.get_index()
                inferred_types = index.get("inferred_types", {})
                if inferred_type in inferred_types:
                    pattern_data = inferred_types[inferred_type]
                    name_pattern = pattern_data.get("pattern")
                    _LOGGER.debug("Inferred type '%s' mapped to pattern '%s'",
                                 inferred_type, name_pattern)
                else:
                    _LOGGER.warning("Inferred type '%s' not found in index", inferred_type)
            else:
                _LOGGER.warning("Index manager not available for inferred_type lookup")

        entities = []
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        # Get area ID if area name provided
        area_id = None
        floor_id = None
        label_id = None
        if area:
            if floor:
                area_entry = self._resolve_area_entry(area, area_registry)
                floor_entry = None
            else:
                area_entry, floor_entry = self._resolve_area_or_floor(
                    area, area_registry, floor_registry
                )
            if area_entry:
                area_id = area_entry.id
            elif not floor:
                if floor_entry:
                    floor_id = floor_entry.floor_id
                else:
                    return []
            else:
                return []

        if floor:
            floor_entry = self._resolve_floor_entry(floor, floor_registry)
            if not floor_entry:
                return []
            floor_id = floor_entry.floor_id

        if label:
            label_entry = self._resolve_label_entry(label, label_registry)
            if not label_entry:
                return []
            label_id = label_entry.label_id

        # Limit to prevent excessive results
        # Read max limit from system entry config, fallback to constant
        from .const import DOMAIN, CONF_MAX_ENTITIES_PER_DISCOVERY
        max_limit = MAX_ENTITIES_PER_DISCOVERY  # Default fallback

        # Try to get configured limit from system entry
        domain_data = self.hass.data.get(DOMAIN, {})
        system_entry = None
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            if entry.source == "system":
                system_entry = entry
                break

        if system_entry:
            max_limit = system_entry.data.get(CONF_MAX_ENTITIES_PER_DISCOVERY, MAX_ENTITIES_PER_DISCOVERY)

        limit = min(limit, max_limit)

        entities = self._scan_general_entities(
            entity_type, domain, state, name_contains, limit, device_class,
            name_pattern, area_id, floor_id, label_id, entity_registry,
            device_registry, area_registry, floor_registry, label_registry,
            fuzzy=False,
        )
        if not entities and name_contains:
            entities = self._scan_general_entities(
                entity_type, domain, state, name_contains, limit, device_class,
                name_pattern, area_id, floor_id, label_id, entity_registry,
                device_registry, area_registry, floor_registry, label_registry,
                fuzzy=True,
            )
            if entities:
                self._add_resolution_hint(
                    f"Nothing contains '{name_contains}' exactly; showing the "
                    "closest names (typo-tolerant match)."
                )

        _LOGGER.debug(
            f"General discovery found {len(entities)} entities with filters: "
            f"type={entity_type}, area={area}, floor={floor}, label={label}, "
            f"domain={domain}, state={state}, name_contains={name_contains}"
        )

        return entities

    def _scan_general_entities(
        self,
        entity_type: Optional[str],
        domain: Optional[str],
        state: Optional[str],
        name_contains: Optional[str],
        limit: int,
        device_class: Optional[Union[str, List[str]]],
        name_pattern: Optional[str],
        area_id: Optional[str],
        floor_id: Optional[str],
        label_id: Optional[str],
        entity_registry: Any,
        device_registry: Any,
        area_registry: Any,
        floor_registry: Any,
        label_registry: Any,
        fuzzy: bool,
    ) -> List[Dict[str, Any]]:
        """One filtering pass; with fuzzy=True, name_contains is typo-tolerant
        and only the entities with the smallest distance are kept."""
        entities: List[Dict[str, Any]] = []
        distances: List[int] = []
        for state_obj in self.hass.states.async_all():
            entity_id = state_obj.entity_id

            # Check if entity is exposed to conversation
            if not async_should_expose(self.hass, "conversation", entity_id):
                continue

            # Apply filters
            entity_domain = entity_id.split(".")[0]

            # Domain filter
            if domain and entity_domain != domain:
                continue

            # Entity type filter (same as domain for backward compatibility)
            if entity_type and entity_domain != entity_type:
                continue

            # State filter
            if state and state_obj.state.lower() != state.lower():
                continue

            # Get entity registry entry for aliases and area info
            entity_entry = entity_registry.async_get(entity_id)
            entity_context = self._get_entity_context(
                entity_entry,
                device_registry,
                area_registry,
                floor_registry,
                label_registry,
            )

            # Enhanced name search - search entity_id, friendly name, AND aliases
            distance = 0
            if name_contains:
                search_term = name_contains.lower()
                if fuzzy:
                    typo = self._entity_typo_distance(
                        search_term, state_obj, entity_entry, entity_context
                    )
                    if typo is None:
                        continue
                    distance = typo
                elif not self._entity_matches_search_term(
                    search_term,
                    state_obj,
                    entity_entry,
                    entity_context,
                ):
                    continue

            # Device class filter
            if device_class:
                entity_device_class = state_obj.attributes.get('device_class')

                # Convert single device_class to list for uniform handling
                device_class_list = [device_class] if isinstance(device_class, str) else device_class

                # Check if entity's device_class matches any in the list (OR logic)
                if entity_device_class not in device_class_list:
                    continue

            # Name pattern filter (wildcard matching)
            if name_pattern:
                # Auto-wrap with wildcards if none are present (defensive for LLM usage)
                pattern = name_pattern
                if '*' not in pattern and '?' not in pattern:
                    pattern = f"*{pattern}*"
                if not fnmatch.fnmatch(entity_id, pattern):
                    continue

            # Apply area filter
            if area_id and entity_context["area_id"] != area_id:
                continue

            # Apply floor filter
            if floor_id and entity_context["floor_id"] != floor_id:
                continue

            # Apply label filter
            if label_id and label_id not in entity_context["label_ids"]:
                continue

            # Create entity info
            entity_info = self._create_entity_info(
                state_obj,
                entity_entry=entity_entry,
                entity_context=entity_context,
            )
            if entity_info is None:
                continue

            entities.append(entity_info)
            distances.append(distance)

            # Stop if we hit the limit (a fuzzy pass ranks everything first)
            if not fuzzy and len(entities) >= limit:
                break

        if fuzzy and entities:
            best = min(distances)
            entities = [
                entity for entity, d in zip(entities, distances) if d == best
            ][:limit]
        return entities

    def _create_entity_info(
        self,
        state_obj: Any,
        description: Optional[str] = None,
        entity_entry: Any = None,
        entity_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create entity information dictionary."""
        entity_info = {
            "entity_id": state_obj.entity_id,
            "name": state_obj.name,
            "domain": state_obj.domain,
            "state": state_obj.state,
        }

        if description:
            entity_info["type"] = description

        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        entity_entry = entity_entry or entity_registry.async_get(state_obj.entity_id)
        if entity_context is None:
            entity_context = self._get_entity_context(
                entity_entry,
                device_registry,
                area_registry,
                floor_registry,
                label_registry,
            )

        for key in (
            "area",
            "area_id",
            "area_aliases",
            "floor",
            "floor_id",
            "floor_aliases",
            "device",
            "device_name",
            "device_name_by_user",
            "device_aliases",
        ):
            if entity_context.get(key):
                entity_info[key] = entity_context[key]

        if entity_context.get("labels"):
            entity_info["labels"] = entity_context["labels"]
            entity_info["label_ids"] = entity_context["label_ids"]

        # Add aliases if present
        entity_aliases = self._get_entity_aliases(entity_entry)
        if entity_aliases:
            entity_info["aliases"] = entity_aliases

        # Add useful attributes
        if state_obj.attributes:
            useful_attrs = {}
            for attr in ["brightness", "temperature", "humidity", "unit_of_measurement",
                        "device_class", "friendly_name"]:
                if attr in state_obj.attributes:
                    useful_attrs[attr] = state_obj.attributes[attr]
            if useful_attrs:
                entity_info["attributes"] = useful_attrs

        return prepare_listed_entity(entity_info)

    def _format_smart_results(
        self, results: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Format smart discovery results for the LLM."""
        # Flatten the results into a list while preserving structure information
        formatted = []

        # Add primary entities first
        for entity in results.get("primary_entities", [])[:limit]:
            entity["relationship"] = "primary"
            formatted.append(entity)

        # Add related entities with their categories
        remaining_limit = limit - len(formatted)
        for category, entities in results.get("related_entities", {}).items():
            if remaining_limit <= 0:
                break
            for entity in entities[:remaining_limit]:
                entity["relationship"] = category
                formatted.append(entity)
                remaining_limit -= 1
                if remaining_limit <= 0:
                    break

        # Add metadata about the query
        if formatted:
            # Add a summary as the first item
            summary = {
                "entity_id": "_summary",
                "query_type": results.get("query_type", "general"),
                "query": results.get("query", ""),
                "total_found": len(results.get("primary_entities", [])) +
                              sum(len(v) for v in results.get("related_entities", {}).values()),
                "primary_count": len(results.get("primary_entities", [])),
                "related_count": sum(len(v) for v in results.get("related_entities", {}).values()),
            }
            formatted.insert(0, summary)

        return formatted

    @staticmethod
    def _serialize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize entity attributes, converting datetime objects to ISO strings."""
        serialized = {}
        for key, value in attributes.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = EntityDiscovery._serialize_attributes(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    item.isoformat() if isinstance(item, datetime) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    # Legacy methods for backward compatibility
    async def get_entity_details(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Get detailed information about specific entities."""
        details = {}
        entity_registry = er.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        for entity_id in entity_ids:
            # Check if entity is exposed
            if not async_should_expose(self.hass, "conversation", entity_id):
                details[entity_id] = {"error": "Entity not exposed to conversation"}
                continue

            state_obj = self.hass.states.get(entity_id)
            if not state_obj:
                details[entity_id] = {"error": "Entity not found"}
                continue

            entity_entry = entity_registry.async_get(entity_id)
            entity_context = self._get_entity_context(
                entity_entry,
                device_registry,
                area_registry,
                floor_registry,
                label_registry,
                include_label_sources=True,
            )

            entity_details = {
                "entity_id": entity_id,
                "name": state_obj.name,
                "domain": state_obj.domain,
                "state": state_obj.state,
                "attributes": self._serialize_attributes(dict(state_obj.attributes)),
                "area": entity_context["area"],
                "area_id": entity_context["area_id"],
                "area_aliases": entity_context["area_aliases"],
                "floor": entity_context["floor"],
                "floor_id": entity_context["floor_id"],
                "floor_aliases": entity_context["floor_aliases"],
                "labels": entity_context["labels"],
                "label_ids": entity_context["label_ids"],
                "label_sources": entity_context.get("label_sources", {}),
                "device": entity_context["device"],
                "device_name": entity_context["device_name"],
                "device_name_by_user": entity_context["device_name_by_user"],
                "device_aliases": entity_context["device_aliases"],
                "last_changed": state_obj.last_changed.isoformat(),
                "last_updated": state_obj.last_updated.isoformat(),
            }

            if entity_entry:
                entity_details.update({
                    "unique_id": entity_entry.unique_id,
                    "entity_category": entity_entry.entity_category,
                    "disabled": entity_entry.disabled_by is not None,
                })

            details[entity_id] = entity_details

        return details

    def _area_name_for_entry(self, entity_entry: Any) -> str:
        """Area name for an entity, following its device when needed."""
        if entity_entry is None:
            return ""
        area_id = getattr(entity_entry, "area_id", None)
        if not area_id and getattr(entity_entry, "device_id", None):
            device = dr.async_get(self.hass).async_get(entity_entry.device_id)
            area_id = getattr(device, "area_id", None) if device else None
        if not area_id:
            return ""
        area = ar.async_get(self.hass).async_get_area(area_id)
        return area.name if area else ""

    def _listable(
        self, entity_id: str, entity_entry: Any = None, area_name: str = ""
    ) -> bool:
        """False for unavailable entities that have no human name, alias, or area."""
        if not async_should_expose(self.hass, "conversation", entity_id):
            return False
        state_obj = self.hass.states.get(entity_id)
        attributes = getattr(state_obj, "attributes", None) or {}
        if entity_entry is None:
            entity_entry = er.async_get(self.hass).async_get(entity_id)
        aliases = (
            self._get_entity_aliases(entity_entry) if entity_entry is not None else []
        )
        record = {
            "entity_id": entity_id,
            "state": getattr(state_obj, "state", None) or "unknown",
            "name": getattr(state_obj, "name", None) or entity_id,
            "friendly_name": attributes.get("friendly_name"),
            "aliases": aliases,
            "area": area_name,
        }
        return listing_disposition(record) != "skip"

    async def list_areas(self) -> List[Dict[str, Any]]:
        """List all areas with entity counts."""
        area_registry = ar.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        areas = []
        for area_entry in area_registry.areas.values():
            # Count entities in this area
            entity_count = 0

            # Count entities directly assigned to area
            for entity_entry in entity_registry.entities.values():
                if entity_entry.area_id == area_entry.id:
                    if self._listable(entity_entry.entity_id, entity_entry, area_entry.name):
                        entity_count += 1

            # Count entities via devices in this area. async_entries_for_area
            # avoids deprecated mapping access on device_registry.devices.
            for device_entry in devices_in_area(device_registry, area_entry.id):
                for entity_entry in entity_registry.entities.values():
                    if (entity_entry.device_id == device_entry.id
                        and not entity_entry.area_id  # Not already counted
                        and self._listable(entity_entry.entity_id, entity_entry, area_entry.name)):
                        entity_count += 1

            floor_id = getattr(area_entry, "floor_id", None)
            floor_name = None
            if floor_id and floor_registry is not None:
                floor_entry = floor_registry.async_get_floor(floor_id)
                if floor_entry:
                    floor_name = floor_entry.name

            label_ids = set(getattr(area_entry, "labels", set()) or set())

            areas.append({
                "id": area_entry.id,
                "name": area_entry.name,
                "aliases": self._get_entry_aliases(area_entry),
                "entity_count": entity_count,
                "floor": floor_name,
                "floor_id": floor_id,
                "labels": self._get_label_names(label_ids, label_registry),
                "label_ids": sorted(label_ids),
            })

        # Sort by name
        areas.sort(key=lambda x: x["name"])
        return areas

    async def list_domains(self) -> List[Dict[str, Any]]:
        """List all domains with entity counts."""
        domain_counts = {}
        entity_registry = er.async_get(self.hass)

        for state_obj in self.hass.states.async_all():
            entity_entry = entity_registry.async_get(state_obj.entity_id)
            if self._listable(
                state_obj.entity_id,
                entity_entry,
                self._area_name_for_entry(entity_entry),
            ):
                domain = state_obj.domain
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        domains = [
            {"domain": domain, "count": count}
            for domain, count in domain_counts.items()
        ]

        # Sort by count (descending) then by name
        domains.sort(key=lambda x: (-x["count"], x["domain"]))
        return domains

    async def get_entities_by_area(self, area_id: str) -> List[Dict[str, Any]]:
        """Get all entities in a specific area by area ID."""
        entities = []
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        area_entry = area_registry.async_get_area(area_id)
        if area_entry is None:
            # Models sometimes pass an area name or alias instead of its id
            area_entry = self._resolve_area_entry(area_id, area_registry)
            if area_entry is not None:
                area_id = area_entry.id
        area_name = area_entry.name if area_entry else ""

        for state in self.hass.states.async_all():
            entity_entry = entity_registry.async_get(state.entity_id)
            entity_area_id = None

            if entity_entry:
                if entity_entry.area_id:
                    entity_area_id = entity_entry.area_id
                elif entity_entry.device_id:
                    device_entry = device_registry.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id:
                        entity_area_id = device_entry.area_id

            if entity_area_id == area_id:
                # Skip unavailable entities that have no human name.
                if self._listable(state.entity_id, entity_entry, area_name):
                    entities.append({
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                        "domain": state.entity_id.split(".")[0]
                    })

        _LOGGER.debug(f"Found {len(entities)} entities in area '{area_id}'")
        return entities


# For backward compatibility, keep the old class name as an alias
EntityDiscovery = SmartDiscovery
