"""Small adapters for Home Assistant APIs that differ between supported cores.

The integration supports Home Assistant 2025.10 and newer. Only public
helpers that were added after 2025.10 are feature-detected here; everything
else is called directly.

- 2026.10 replaced ``ToolResultContent.tool_result`` with
  ``ToolResultContent.result`` (an ``llm.ToolResult``).
- 2026.4 added ``entity_registry.async_get_entity_aliases``, which resolves
  computed alias placeholders; before that ``RegistryEntry.aliases`` is a
  plain ``set[str]``.
- 2026.9 added child devices, which inherit their parent's area, together
  with ``async_get_effective_area_id`` in the device and entity registries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from homeassistant.components.conversation import chat_log
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import llm

# Public helpers that only exist on newer cores (None on older ones).
_LLM_TOOL_RESULT = getattr(llm, "ToolResult", None)  # 2026.10+
_DEVICE_EFFECTIVE_AREA = getattr(dr, "async_get_effective_area_id", None)  # 2026.9+
_ENTITY_EFFECTIVE_AREA = getattr(er, "async_get_effective_area_id", None)  # 2026.9+
_ENTITY_ALIASES = getattr(er, "async_get_entity_aliases", None)  # 2026.4+


def _as_json_object(tool_result: Any) -> dict[str, Any]:
    """Coerce a tool payload into the JSON object ``llm.ToolResult`` stores."""
    if isinstance(tool_result, dict):
        return tool_result
    return {"result": tool_result}


def _tool_result_failed(tool_result: dict[str, Any]) -> bool:
    """Best-effort failure flag for chat-log tool results."""
    if tool_result.get("isError") is True:
        return True
    if tool_result.get("error"):
        return True
    result = tool_result.get("result")
    return isinstance(result, str) and result.startswith("ERROR:")


def build_tool_result_content(
    agent_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_result: Any,
) -> chat_log.ToolResultContent:
    """Build a ``ToolResultContent`` for the running Home Assistant version."""
    payload = _as_json_object(tool_result)
    if _LLM_TOOL_RESULT is not None:
        return chat_log.ToolResultContent(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=_LLM_TOOL_RESULT(
                data=payload, error=_tool_result_failed(payload)
            ),
        )
    return chat_log.ToolResultContent(
        agent_id=agent_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_result=payload,
    )


def device_area_id(hass: HomeAssistant, device: dr.DeviceEntry) -> str | None:
    """Return the area of a device, inheriting the parent's area if needed."""
    if _DEVICE_EFFECTIVE_AREA is not None:
        return _DEVICE_EFFECTIVE_AREA(hass, device)
    return device.area_id


def entity_area_id(hass: HomeAssistant, entry: er.RegistryEntry) -> str | None:
    """Return the area of an entity, falling back to its device's area."""
    if _ENTITY_EFFECTIVE_AREA is not None:
        return _ENTITY_EFFECTIVE_AREA(hass, entry)
    if entry.area_id:
        return entry.area_id
    if entry.device_id and (device := dr.async_get(hass).async_get(entry.device_id)):
        return device.area_id
    return None


def entity_aliases(hass: HomeAssistant, entry: er.RegistryEntry) -> list[str]:
    """Return an entity's aliases as plain, non-empty strings."""
    if _ENTITY_ALIASES is not None:
        # A computed alias resolves to an empty string for an unnamed entity.
        return [alias for alias in _ENTITY_ALIASES(hass, entry) if alias]
    return sorted((alias for alias in entry.aliases if alias), key=str.casefold)


def devices_in_area(device_registry: dr.DeviceRegistry, area_id: str) -> list[Any]:
    """Return devices in ``area_id`` (child devices included on newer cores)."""
    return list(dr.async_entries_for_area(device_registry, area_id))


def devices_by_label(
    device_registry: dr.DeviceRegistry, label_registry: lr.LabelRegistry
) -> dict[str, set[str]]:
    """Map label id to the device ids that carry that label."""
    label_devices: dict[str, set[str]] = defaultdict(set)
    for label_entry in label_registry.async_list_labels():
        for device_entry in dr.async_entries_for_label(
            device_registry, label_entry.label_id
        ):
            label_devices[label_entry.label_id].add(device_entry.id)
    return label_devices
