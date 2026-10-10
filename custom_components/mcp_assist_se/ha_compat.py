"""Small adapters for Home Assistant conversation and device-registry APIs.

Home Assistant 2026.10 replaced ``ToolResultContent.tool_result`` with
``ToolResultContent.result`` (an ``llm.ToolResult``). Passing the old keyword
raises ``TypeError``. 2026.9 and older still take ``tool_result``.

The same release warns when custom integrations use ``device_registry.devices``
as a mapping (``.values()``, ``.get()``, ...). Iterating the collection is
still supported; area and label helpers are the accessors to prefer.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from homeassistant.components.conversation import chat_log
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm


def _dataclass_field_names(cls: type) -> set[str]:
    """Return field names for a dataclass, or constructor parameter names."""
    fields = getattr(cls, "__dataclass_fields__", None)
    if isinstance(fields, dict) and fields:
        return set(fields)
    try:
        return set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        return set()


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
) -> Any:
    """Build a ``ToolResultContent`` for the running Home Assistant version."""
    payload = _as_json_object(tool_result)
    content_fields = _dataclass_field_names(chat_log.ToolResultContent)
    if "result" in content_fields:
        tool_result_cls = getattr(llm, "ToolResult", None)
        if tool_result_cls is None:
            raise RuntimeError(
                "This Home Assistant version expects ToolResultContent.result "
                "but homeassistant.helpers.llm.ToolResult is missing"
            )
        result_kwargs: dict[str, Any] = {"data": payload}
        if "error" in _dataclass_field_names(tool_result_cls):
            result_kwargs["error"] = _tool_result_failed(payload)
        return chat_log.ToolResultContent(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=tool_result_cls(**result_kwargs),
        )

    return chat_log.ToolResultContent(
        agent_id=agent_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_result=payload,
    )


def iter_device_entries(device_registry: Any) -> Iterator[Any]:
    """Yield device entries without deprecated mapping access.

    - Very old registries store a real ``dict`` (iteration yields device ids).
    - Home Assistant 2026.8 exposes a ``UserDict`` whose ``.values()`` is a
      real method and whose iteration also yields ids.
    - Home Assistant 2026.9+ iterates entries directly. On 2026.10, calling
      ``.values()`` on that collection logs a deprecation warning because the
      method is provided by ``__getattr__``. Look the method up on the class
      so the warning is not triggered just by checking for it.
    """
    devices = device_registry.devices
    if isinstance(devices, dict):
        return iter(devices.values())

    values = getattr(type(devices), "values", None)
    if callable(values) and not _iteration_yields_entries(devices):
        return iter(values(devices))
    return iter(devices)


def _iteration_yields_entries(devices: Any) -> bool:
    """Return whether iterating ``devices`` yields entries rather than ids."""
    iterator = iter(devices)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return not isinstance(first, str)


def devices_in_area(device_registry: Any, area_id: str) -> list[Any]:
    """Return devices whose area matches ``area_id``.

    Uses ``async_entries_for_area`` when Home Assistant provides it, including
    child devices that belong to the area. Older cores fall back to a scan
    that does not call deprecated mapping methods.
    """
    entries_for_area = getattr(dr, "async_entries_for_area", None)
    if entries_for_area is not None:
        return list(entries_for_area(device_registry, area_id))
    return [
        device
        for device in iter_device_entries(device_registry)
        if getattr(device, "area_id", None) == area_id
    ]


def devices_by_label(
    device_registry: Any, label_registry: Any
) -> dict[str, set[str]]:
    """Map label id to the device ids that carry that label."""
    label_devices: dict[str, set[str]] = defaultdict(set)
    entries_for_label = getattr(dr, "async_entries_for_label", None)
    list_labels = (
        getattr(label_registry, "async_list_labels", None)
        if label_registry is not None
        else None
    )
    if entries_for_label is not None and list_labels is not None:
        for label_entry in list_labels():
            label_id = getattr(label_entry, "label_id", None)
            if not label_id:
                continue
            for device_entry in entries_for_label(device_registry, label_id):
                device_id = getattr(device_entry, "id", None)
                if device_id:
                    label_devices[label_id].add(device_id)
        return label_devices

    for device_entry in iter_device_entries(device_registry):
        device_id = getattr(device_entry, "id", None)
        if not device_id:
            continue
        for label_id in getattr(device_entry, "labels", set()) or set():
            label_devices[label_id].add(device_id)
    return label_devices
