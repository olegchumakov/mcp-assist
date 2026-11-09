"""Small discovery rules that do not need Home Assistant.

"all" is a whole word, so a hallway search is not treated as "everything".
A room plus a name stays a name search inside that room. A domain of all,
any, or * means no domain filter.
"""

from __future__ import annotations

import re
from typing import Any

_AGGREGATE_WORDS = re.compile(r"\b(who|anyone|everyone|all)\b", re.IGNORECASE)
_ANY_DOMAIN = {"all", "any", "*"}
_FILTER_KEYS = (
    "area",
    "floor",
    "label",
    "domain",
    "device_class",
    "state",
    "name_contains",
    "name_pattern",
    "inferred_type",
)


def name_is_aggregate_query(name: str) -> bool:
    """True when the name asks for everyone, as whole words only."""
    return _AGGREGATE_WORDS.search(name or "") is not None


def domain_means_any(domain: Any) -> bool:
    """True when the model asked for every domain with a catch-all value."""
    return isinstance(domain, str) and domain.strip().lower() in _ANY_DOMAIN


def area_listing_applies(
    *,
    area: Any,
    name_contains: Any,
    floor: Any,
    label: Any,
    device_class: Any,
    name_pattern: Any,
    inferred_type: Any,
) -> bool:
    """True when the search is a room list and not also a name search."""
    return bool(
        area
        and not name_contains
        and not floor
        and not label
        and not device_class
        and not name_pattern
        and not inferred_type
    )


def empty_discovery_message(
    args: dict[str, Any] | None,
    *,
    area_is_known: bool | None = None,
    known_area_names: list[str] | None = None,
) -> str:
    """Explain an empty search so the model can try a simpler one.

    known_area_names is omitted when the caller must not list every room,
    for example while per-person limits are on.
    """
    raw = args if isinstance(args, dict) else {}
    message = "No entities found matching the search criteria."
    area = raw.get("area")
    if isinstance(area, str) and area and area_is_known is False:
        if known_area_names:
            return (
                f"{message} '{area}' is not a known area. "
                f"Valid areas: {', '.join(known_area_names)}."
            )
        return (
            f"{message} '{area}' is not a known area. "
            "Use a room name this assistant can see."
        )

    filters = [key for key in _FILTER_KEYS if raw.get(key)]
    if len(filters) > 1:
        return (
            f"{message} Try again with fewer filters (used: {', '.join(filters)})."
        )
    if raw.get("name_contains"):
        return (
            f"{message} Try a shorter part of the name, or search by domain "
            "or device_class from the index instead."
        )
    return message
