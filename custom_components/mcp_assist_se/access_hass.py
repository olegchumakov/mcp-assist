"""Build an access snapshot from Home Assistant registries.

Local lookups only. This module does not call the Speaker ID add-on.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er

try:
    from homeassistant.helpers import label_registry as lr
except ImportError:  # pragma: no cover - older Home Assistant cores
    lr = None

from .access_control import EntityRef, HomeSnapshot
from .ha_compat import iter_device_entries


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
    label_reg = lr.async_get(hass) if lr is not None else None

    devices: dict = {}
    for device in iter_device_entries(device_reg):
        device_id = getattr(device, "id", None)
        if device_id:
            devices[device_id] = device

    areas: dict = {}
    area_floors: dict[str, str] = {}
    try:
        area_list = list(area_reg.async_list_areas())
    except Exception:
        area_list = list(getattr(area_reg, "areas", {}).values())
    for area in area_list:
        area_id = getattr(area, "id", None)
        if not area_id:
            continue
        areas[area_id] = area
        floor_id = getattr(area, "floor_id", None)
        if floor_id:
            area_floors[area_id] = floor_id

    label_names: dict[str, str] = {}
    if label_reg is not None:
        try:
            for label in label_reg.async_list_labels():
                label_id = getattr(label, "label_id", None)
                name = getattr(label, "name", None)
                if label_id and name:
                    label_names[label_id] = name
        except Exception:
            label_names = {}

    def resolved_labels(label_ids: set[str]) -> tuple[frozenset[str], frozenset[str]]:
        names = frozenset(label_names[item] for item in label_ids if item in label_names)
        return frozenset(label_ids), names

    entities: dict[str, EntityRef] = {}

    def add(entity_id: str, state, entry) -> None:
        if not entity_id or entity_id in entities:
            return
        device_id = getattr(entry, "device_id", None) if entry is not None else None
        device = devices.get(device_id) if device_id else None
        area_id = getattr(entry, "area_id", None) if entry is not None else None
        if not area_id and device is not None:
            area_id = getattr(device, "area_id", None)
        area = areas.get(area_id) if area_id else None
        label_ids: set[str] = set()
        if entry is not None:
            label_ids.update(getattr(entry, "labels", ()) or ())
        if device is not None:
            label_ids.update(getattr(device, "labels", ()) or ())
        if area is not None:
            label_ids.update(getattr(area, "labels", ()) or ())
        attributes = dict(getattr(state, "attributes", None) or {})
        friendly = attributes.get("friendly_name") or ""
        if not isinstance(friendly, str):
            friendly = str(friendly)
        name = getattr(state, "name", None) or friendly or entity_id
        ids, names = resolved_labels(label_ids)
        domain = entity_id.split(".", 1)[0]
        entities[entity_id] = EntityRef(
            entity_id=entity_id,
            domain=domain,
            device_id=device_id,
            area_id=area_id,
            area_name=getattr(area, "name", "") or "",
            label_ids=ids,
            label_names=names,
            member_ids=_member_ids(attributes),
            name=str(name),
            friendly_name=friendly,
        )

    for state in hass.states.async_all():
        entity_id = getattr(state, "entity_id", "")
        entry = entity_reg.async_get(entity_id) if entity_id else None
        add(entity_id, state, entry)

    for entry in entity_reg.entities.values():
        entity_id = getattr(entry, "entity_id", "")
        if entity_id and entity_id not in entities:
            add(entity_id, None, entry)

    return HomeSnapshot(entities=entities, area_floors=area_floors)
