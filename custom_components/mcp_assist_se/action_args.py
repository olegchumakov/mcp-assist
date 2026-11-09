"""Normalize perform_action arguments before access control and verification.

A plain entity_id is accepted, and target fields the model put in data are
moved onto the target so they go through the same checks. This module does
not decide who may act: the caller's access control and action verification
still run on the normalized arguments.
"""

from __future__ import annotations

from typing import Any, Callable

_TARGET_MOVE_KEYS = ("entity_id", "area_id", "device_id")
_BROAD_ENTITY_IDS = {"all", "none"}


def merge_ids(existing: Any, extra: Any) -> list[str]:
    """Combine one id or a list of ids, keeping the first occurrence."""
    merged: list[str] = []
    for value in (existing, extra):
        if isinstance(value, str):
            items: list[Any] = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            continue
        for item in items:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def normalize_action_arguments(
    args: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (target, data) with a flat entity_id and data targets folded in."""
    raw = args if isinstance(args, dict) else {}
    target = raw.get("target") or {}
    target = dict(target) if isinstance(target, dict) else {}
    data = raw.get("data") or {}
    data = dict(data) if isinstance(data, dict) else {}

    if raw.get("entity_id"):
        target["entity_id"] = merge_ids(target.get("entity_id"), raw.get("entity_id"))
    for key in _TARGET_MOVE_KEYS:
        if key in data:
            target[key] = merge_ids(target.get(key), data.pop(key))
    return target, data


def explicit_entity_ids(target: dict[str, Any] | None) -> list[str]:
    """Entity ids the caller named, before areas or devices are expanded."""
    if not isinstance(target, dict):
        return []
    return merge_ids(None, target.get("entity_id"))


def unexposed_entity_ids(
    entity_ids: Any, is_exposed: Callable[[str], bool]
) -> list[str]:
    """Ids for which is_exposed is false, in the original order."""
    if isinstance(entity_ids, str):
        values: list[Any] = [entity_ids]
    elif isinstance(entity_ids, (list, tuple)):
        values = list(entity_ids)
    else:
        return []
    refused: list[str] = []
    for entity_id in values:
        if not isinstance(entity_id, str) or not entity_id:
            continue
        if entity_id not in refused and not is_exposed(entity_id):
            refused.append(entity_id)
    return refused


def entity_domain(entity_id: str) -> str:
    """Domain prefix of an entity id, or empty when it has none."""
    if not isinstance(entity_id, str) or "." not in entity_id:
        return ""
    return entity_id.split(".", 1)[0]


def matches_service_domain(entity_id: str, service_domain: str) -> bool:
    """True when this entity is the kind of device the action controls.

    ``homeassistant`` actions, such as turn_on, apply to every domain.
    """
    if service_domain == "homeassistant":
        return True
    return entity_domain(entity_id) == service_domain


def group_target_refusal_text(service_domain: str) -> str:
    """Refusal when a device or room has no exposed entity of this kind."""
    kind = service_domain if service_domain and service_domain != "homeassistant" else "matching"
    return (
        "❌ Error: no exposed "
        f"{kind} entities were found for that device or room, "
        "so the command was not sent."
    )


def finalize_action_entities(
    *,
    explicit_ids: list[str],
    group_ids: list[str],
    is_exposed: Callable[[str], bool],
    service_domain: str,
    had_group_target: bool,
) -> tuple[list[str], str]:
    """Choose the entities a perform_action call may control.

    An entity id the caller named must itself be exposed. ``all`` is refused.
    A device or room contributes only entities that are exposed and match the
    action's domain. Hidden entities are never returned. When the device or
    room was the target and nothing exposed matches, the call is refused.
    """
    if any(
        isinstance(entity_id, str) and entity_id.lower() in _BROAD_ENTITY_IDS
        for entity_id in explicit_ids
    ):
        broad = [
            entity_id
            for entity_id in explicit_ids
            if isinstance(entity_id, str) and entity_id.lower() in _BROAD_ENTITY_IDS
        ]
        return [], exposure_refusal_text(broad or ["all"], explicit_ids)

    hidden_named = unexposed_entity_ids(explicit_ids, is_exposed)
    if hidden_named:
        return [], exposure_refusal_text(hidden_named, explicit_ids)

    selected_group: list[str] = []
    for entity_id in group_ids:
        if not isinstance(entity_id, str) or not entity_id:
            continue
        if entity_id.lower() in _BROAD_ENTITY_IDS:
            continue
        if not matches_service_domain(entity_id, service_domain):
            continue
        if not is_exposed(entity_id):
            continue
        if entity_id not in selected_group:
            selected_group.append(entity_id)

    # A named id stays ahead of the same id found on the device or in the room.
    chosen = merge_ids(explicit_ids, selected_group)
    if had_group_target and not chosen:
        return [], group_target_refusal_text(service_domain)
    return chosen, ""


def exposure_refusal_text(refused: list[str], requested: list[str]) -> str:
    """Tell the model the command was not sent.

    Only ids the caller actually named are repeated. Entities found by
    expanding a device stay unnamed, so a hidden sensor id is not introduced.
    """
    if not refused:
        return ""
    if any(entity_id.lower() in _BROAD_ENTITY_IDS for entity_id in refused):
        return (
            "❌ Error: name specific devices. 'all' is not allowed, because it "
            "would include devices that are not exposed to the assistant."
        )
    requested_set = set(requested)
    named = [entity_id for entity_id in refused if entity_id in requested_set]
    if named and len(named) == len(refused):
        return (
            "❌ Error: these entities are not exposed to Assist and cannot be "
            f"controlled: {', '.join(named)}"
        )
    if named:
        return (
            "❌ Error: these entities are not exposed to Assist and cannot be "
            f"controlled: {', '.join(named)}. The command was not sent."
        )
    return (
        "❌ Error: the target includes entities that are not exposed to Assist, "
        "so the command was not sent."
    )
