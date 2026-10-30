"""Build an access snapshot from Home Assistant registries.

Local lookups only. This module does not call the Speaker ID add-on.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    label_registry as lr,
)

from .access_control import EntityRef, HomeSnapshot
from .ha_compat import entity_area_id


def _member_ids(attributes: dict) -> tuple[str, ...]:
    raw = attributes.get("entity_id")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    members: list[str] = []
    for item in raw:
        if isinstance(item, str) and "." in item and item not in members:
            members.append(item)
    return tuple(members)


def snapshot_from_hass(hass: HomeAssistant) -> HomeSnapshot:
    """Every entity the checker can see, including ones not exposed to Assist.

    Area-wide service calls reach hidden entities too, so the allow list has
    to include them or a room action could slip past the check.
    """
    entity_reg = er.async_get(hass)
    device_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)
    label_reg = lr.async_get(hass)

    areas: dict = {}
    area_floors: dict[str, str] = {}
    for area in area_reg.async_list_areas():
        areas[area.id] = area
        if area.floor_id:
            area_floors[area.id] = area.floor_id

    label_names: dict[str, str] = {
        label.label_id: label.name
        for label in label_reg.async_list_labels()
        if label.name
    }

    def resolved_labels(label_ids: set[str]) -> tuple[frozenset[str], frozenset[str]]:
        names = frozenset(label_names[item] for item in label_ids if item in label_names)
        return frozenset(label_ids), names

    entities: dict[str, EntityRef] = {}

    def add(entity_id: str, state, entry) -> None:
        if not entity_id or entity_id in entities:
            return
        device_id = entry.device_id if entry is not None else None
        device = device_reg.async_get(device_id) if device_id else None
        area_id = entity_area_id(hass, entry) if entry is not None else None
        area = areas.get(area_id) if area_id else None
        label_ids: set[str] = set()
        if entry is not None:
            label_ids.update(entry.labels)
        if device is not None:
            label_ids.update(device.labels)
        if area is not None:
            label_ids.update(area.labels)
        attributes = dict(state.attributes) if state is not None else {}
        friendly = attributes.get("friendly_name") or ""
        if not isinstance(friendly, str):
            friendly = str(friendly)
        name = (state.name if state is not None else None) or friendly or entity_id
        ids, names = resolved_labels(label_ids)
        domain = entity_id.split(".", 1)[0]
        entities[entity_id] = EntityRef(
            entity_id=entity_id,
            domain=domain,
            device_id=device_id,
            area_id=area_id,
            area_name=area.name if area is not None else "",
            label_ids=ids,
            label_names=names,
            member_ids=_member_ids(attributes),
            name=str(name),
            friendly_name=friendly,
        )

    for state in hass.states.async_all():
        add(state.entity_id, state, entity_reg.async_get(state.entity_id))

    for entry in entity_reg.entities.values():
        if entry.entity_id not in entities:
            add(entry.entity_id, None, entry)

    return HomeSnapshot(entities=entities, area_floors=area_floors)
