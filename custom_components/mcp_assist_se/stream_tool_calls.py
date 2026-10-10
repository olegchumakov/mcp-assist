"""Accumulate streamed tool calls without dropping parallel calls.

OpenAI streams one call per index and then appends argument fragments.
Gemini's OpenAI-compatible stream often reuses index 0 for every parallel
call and omits ids, or repeats one id. Keying only on that index keeps the
last call. A new name after a slot has started its arguments is a new call.
Later argument fragments for that index follow the newest call.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

_STREAM_INDEX = "_stream_index"


def _function(payload: dict[str, Any]) -> dict[str, Any]:
    function = payload.get("function")
    return function if isinstance(function, dict) else {}


def _name(payload: dict[str, Any]) -> str:
    name = _function(payload).get("name")
    return name if isinstance(name, str) else ""


def _call_id(payload: dict[str, Any]) -> str:
    call_id = payload.get("id")
    return call_id if isinstance(call_id, str) else ""


def _provider_index(fragment: dict[str, Any]) -> int | None:
    raw_index = fragment.get("index")
    if raw_index is None or raw_index == "":
        return None
    try:
        return max(0, int(raw_index))
    except (TypeError, ValueError):
        return 0


def _arguments_text(payload: dict[str, Any]) -> str:
    raw = _function(payload).get("arguments")
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if raw is None:
        return ""
    return str(raw)


def _has_started_arguments(slot: dict[str, Any]) -> bool:
    function = _function(slot)
    partial = function.get("_partial")
    if isinstance(partial, str) and partial.strip():
        return True
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return True
    return isinstance(arguments, str) and bool(arguments.strip())


def _arguments_complete(slot: dict[str, Any]) -> bool:
    raw = _function(slot).get("arguments")
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def _opens_new_payload(fragment: dict[str, Any]) -> bool:
    """True when this fragment looks like the start of a call, not a JSON tail."""
    function = _function(fragment)
    if "arguments" not in function:
        return True
    raw = function.get("arguments")
    if isinstance(raw, dict) or raw is None:
        return True
    text = str(raw).strip()
    return text == "" or text[:1] in "{["


def _is_new_call(existing: dict[str, Any], fragment: dict[str, Any]) -> bool:
    """True when this fragment must not be merged into ``existing``."""
    if not existing:
        return False
    fragment_id = _call_id(fragment)
    existing_id = _call_id(existing)
    fragment_name = _name(fragment)
    existing_name = _name(existing)
    if fragment_id and existing_id and fragment_id != existing_id:
        return True
    if fragment_name and existing_name and fragment_name != existing_name:
        return True
    if fragment_id and existing_id and fragment_id == existing_id:
        return False
    if (
        fragment_name
        and existing_name
        and _has_started_arguments(existing)
        and _opens_new_payload(fragment)
    ):
        return True
    if (
        not fragment_name
        and _arguments_complete(existing)
        and _opens_new_payload(fragment)
        and _arguments_text(fragment).strip()[:1] in "{["
    ):
        return True
    return False


def _slots_for_provider_index(tool_calls: list[dict[str, Any]], provider_index: int) -> list[int]:
    return [
        index
        for index, slot in enumerate(tool_calls)
        if slot.get(_STREAM_INDEX) == provider_index
    ]


def resolve_streamed_tool_slot(tool_calls: list[dict[str, Any]], fragment: dict[str, Any]) -> int:
    """Return the slot this fragment belongs to, appending when it is a new call."""
    fragment_id = _call_id(fragment)
    if fragment_id:
        matches = [
            index
            for index, existing in enumerate(tool_calls)
            if _call_id(existing) == fragment_id and not _is_new_call(existing, fragment)
        ]
        if matches:
            # A reused id's argument tail belongs to the call that is still open.
            open_slots = [
                index for index in matches if not _arguments_complete(tool_calls[index])
            ]
            return open_slots[-1] if open_slots else matches[-1]

    provider_index = _provider_index(fragment)
    if provider_index is not None:
        aliased = _slots_for_provider_index(tool_calls, provider_index)
        target = aliased[-1] if aliased else provider_index
        if target < len(tool_calls) and _is_new_call(tool_calls[target], fragment):
            return len(tool_calls)
        return target

    if _call_id(fragment) or _name(fragment):
        if not tool_calls:
            return 0
        if _is_new_call(tool_calls[-1], fragment):
            return len(tool_calls)
        return len(tool_calls) - 1
    return max(0, len(tool_calls) - 1)


def merge_streamed_tool_fragment(
    tool_calls: list[dict[str, Any]], fragment: dict[str, Any]
) -> int:
    """Merge one streamed fragment. Argument text is kept until it is valid JSON."""
    index = resolve_streamed_tool_slot(tool_calls, fragment)
    while index >= len(tool_calls):
        tool_calls.append({})
    slot = tool_calls[index]
    provider_index = _provider_index(fragment)
    if provider_index is not None:
        slot[_STREAM_INDEX] = provider_index
    fragment_id = _call_id(fragment)
    if fragment_id:
        slot["id"] = fragment_id
        slot["type"] = "function"
    if "extra_content" in fragment and "extra_content" not in slot:
        slot["extra_content"] = fragment["extra_content"]

    function = _function(fragment)
    if "name" in function or "arguments" in function:
        slot.setdefault("function", {})
    if isinstance(function.get("name"), str) and function["name"]:
        slot["function"]["name"] = function["name"]
    if "arguments" not in function:
        return index

    raw_arguments = function["arguments"]
    if isinstance(raw_arguments, dict):
        slot["function"]["arguments"] = json.dumps(raw_arguments, ensure_ascii=False)
        slot["function"].pop("_partial", None)
        return index

    partial = str(slot["function"].get("_partial") or "") + str(raw_arguments)
    slot["function"]["_partial"] = partial
    try:
        json.loads(partial)
    except json.JSONDecodeError:
        return index
    slot["function"]["arguments"] = partial
    return index


def finalize_streamed_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop empty slots, finish partial arguments, and give every call its own id."""
    finished: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for slot in tool_calls:
        if not slot:
            continue
        function = dict(_function(slot))
        partial = function.pop("_partial", None)
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        if "arguments" not in function and partial:
            function["arguments"] = partial
        call = {
            "type": slot.get("type") or "function",
            "function": function,
        }
        call_id = _call_id(slot)
        if not call_id or call_id in seen_ids:
            call_id = f"call_{len(finished)}_{uuid.uuid4().hex[:8]}"
        seen_ids.add(call_id)
        call["id"] = call_id
        if "extra_content" in slot:
            call["extra_content"] = slot["extra_content"]
        finished.append(call)
    return finished
