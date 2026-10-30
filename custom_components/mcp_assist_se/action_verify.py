"""Generic post-state checks for perform_action.

Nothing here is specific to a home, a device, or a language. Callers pass
entity snapshots and whatever remembered facts were injected on this turn.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .memory_rank import content_stems, fold_token, stems_match

# Quick wait for actions whose outcome we cannot predict (volume_up, play_media).
CONFIRM_WAIT_SECONDS = 2.0
# Short wait when nothing should change, to catch a late report.
SETTLED_WAIT_SECONDS = 0.25
# Longest wait for an expected change. The shared option overrides it.
EXPECTED_CHANGE_WAIT_SECONDS = 10.0
MIN_EXPECTED_CHANGE_WAIT_SECONDS = 2.0
MAX_EXPECTED_CHANGE_WAIT_SECONDS = 30.0

WAIT_EXPECT = "expect"
WAIT_OBSERVE = "observe"
WAIT_SETTLE = "settle"

_DEAD_STATES = frozenset({"unavailable", "unknown"})
_ACTIVATION_ACTIONS = frozenset({"turn_on", "open", "unlock", "start", "play"})
_IMPLIED_STATE = {
    "turn_on": "on",
    "turn_off": "off",
    "open": "open",
    "close": "closed",
    "lock": "locked",
    "unlock": "unlocked",
    "start": "on",
    "stop": "off",
}
_HEX_ID = re.compile(r"^(?:0x)?[0-9a-f]{8,}$", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_SKIP_ATTRS = frozenset(
    {
        "friendly_name",
        "icon",
        "entity_picture",
        "supported_features",
        "supported_color_modes",
        "attribution",
    }
)
_SUMMARY_ATTRS = (
    "brightness_pct",
    "color_temp_kelvin",
    "rgb_color",
    "hs_color",
    "xy_color",
    "temperature",
    "target_temp_high",
    "target_temp_low",
    "humidity",
    "current_position",
    "position",
    "percentage",
    "volume_level",
)

_ERROR_STATUSES = frozenset({"not_done", "no_change", "not_confirmed"})
_STATUS_LABEL = {
    "changed": "CHANGED",
    "sent": "SENT",
    "already": "ALREADY",
    "no_change": "NO CHANGE",
    "not_done": "NOT DONE",
    "not_confirmed": "NOT CONFIRMED",
}


def name_is_machine_id(name: str, entity_id: str = "") -> bool:
    """True when a label is empty, the raw entity id, or an IEEE/hex address."""
    text = (name or "").strip()
    if not text:
        return True
    if entity_id and text.casefold() == str(entity_id).strip().casefold():
        return True
    compact = re.sub(r"[\s_\-.]", "", text)
    if _HEX_ID.match(compact):
        return True
    object_id = str(entity_id).split(".", 1)[-1] if entity_id else ""
    object_compact = re.sub(r"[\s_\-.]", "", object_id)
    folded = text.casefold().replace(" ", "_")
    if (
        object_id
        and folded == object_id.casefold()
        and _HEX_ID.match(object_compact)
    ):
        return True
    return False


def _human_labels(record: dict[str, Any], *, include_area: bool) -> list[str]:
    entity_id = str(record.get("entity_id") or "")
    labels: list[str] = []
    for key in ("friendly_name", "name"):
        value = str(record.get(key) or "").strip()
        if value and not name_is_machine_id(value, entity_id):
            labels.append(value)
    for alias in record.get("aliases") or []:
        text = str(alias).strip()
        if text and not name_is_machine_id(text, entity_id):
            labels.append(text)
    if include_area or not labels:
        area = str(record.get("area") or "").strip()
        if area and not name_is_machine_id(area, entity_id):
            if not labels or include_area:
                labels.append(area)
    return labels


def has_human_identity(record: dict[str, Any]) -> bool:
    """True when the entity has a friendly name, alias, or area a person set."""
    return bool(_human_labels(record, include_area=True))


def listing_disposition(record: dict[str, Any]) -> str:
    """keep, mark, or skip.

    Unavailable or unknown entities with no human name are skipped.
    Other unavailable entities are kept and marked.
    """
    state = str(record.get("state") or "").casefold()
    if state not in _DEAD_STATES:
        return "keep"
    if has_human_identity(record):
        return "mark"
    return "skip"


def prepare_listed_entity(info: dict[str, Any]) -> dict[str, Any] | None:
    """Drop unnamed unavailable entities and mark the rest."""
    if not isinstance(info, dict):
        return None
    if info.get("entity_id") == "_summary":
        return info
    attributes = info.get("attributes") if isinstance(info.get("attributes"), dict) else {}
    record = {
        "entity_id": info.get("entity_id") or "",
        "state": info.get("state"),
        "name": info.get("name"),
        "friendly_name": info.get("friendly_name") or attributes.get("friendly_name"),
        "aliases": info.get("aliases") or [],
        "area": info.get("area") or "",
    }
    disposition = listing_disposition(record)
    if disposition == "skip":
        return None
    if disposition != "mark":
        return info
    state = str(info.get("state") or "")
    if "unavailable" in state.casefold():
        return info
    marked = dict(info)
    marked["state"] = f"{state} (unavailable)"
    return marked


def index_display_name(record: dict[str, Any]) -> str:
    """Name for an index list, with a mark when the entity is unavailable."""
    entity_id = str(record.get("entity_id") or "")
    raw = (
        str(record.get("friendly_name") or "").strip()
        or str(record.get("name") or "").strip()
        or entity_id
    )
    if listing_disposition(record) == "mark" and "unavailable" not in raw.casefold():
        return f"{raw} (unavailable)"
    return raw


def _copy_attribute(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        if isinstance(value, str) and len(value) > 80:
            return None
        return value
    if (
        isinstance(value, (list, tuple))
        and len(value) <= 8
        and all(isinstance(item, (int, float, str)) for item in value)
    ):
        return [item for item in value]
    return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return str(value)
    return str(value)


def snapshot_from_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one entity state into a comparable snapshot."""
    attributes_in = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    attributes: dict[str, Any] = {}
    for key, value in attributes_in.items():
        if key in _SKIP_ATTRS:
            continue
        copied = _copy_attribute(value)
        if copied is not None:
            attributes[key] = copied
    brightness = attributes_in.get("brightness")
    brightness_pct = attributes_in.get("brightness_pct")
    if brightness_pct is None and isinstance(brightness, (int, float)) and not isinstance(
        brightness, bool
    ):
        brightness_pct = round(float(brightness) * 100 / 255)
        attributes["brightness_pct"] = brightness_pct
    elif brightness_pct is not None:
        attributes["brightness_pct"] = brightness_pct
    missing = bool(raw.get("missing")) or raw.get("state") is None
    state = "unknown" if missing else raw.get("state")
    return {
        "entity_id": raw.get("entity_id") or "",
        "state": state,
        "name": raw.get("friendly_name") or raw.get("name") or raw.get("entity_id") or "",
        "friendly_name": raw.get("friendly_name") or attributes_in.get("friendly_name"),
        "area": raw.get("area") or "",
        "aliases": list(raw.get("aliases") or []),
        "last_updated": _as_text(raw.get("last_updated")),
        "last_reported": _as_text(raw.get("last_reported")),
        "brightness_pct": attributes.get("brightness_pct"),
        "attributes": attributes,
        "missing": missing,
    }


def is_dead_snapshot(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("missing"):
        return True
    return str(snapshot.get("state") or "").casefold() in _DEAD_STATES


def _numbers_close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(1.0, abs(b) * 0.02)


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_match(a, b) for a, b in zip(left, right))
    if _numbers_close(left, right):
        return True
    return left == right


def _attribute(snapshot: dict[str, Any], key: str) -> Any:
    attributes = snapshot.get("attributes") or {}
    if key in attributes:
        return attributes[key]
    return snapshot.get(key)


def observable_settings(
    data: dict[str, Any] | None, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Service fields that correspond to a value we can read back.

    Parameters that are not on the entity, such as a transition time, are
    not treated as a requested device setting.
    """
    if not data:
        return {}
    kept: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"brightness", "brightness_pct"} or _attribute(snapshot, key) is not None:
            kept[key] = value
    return kept


def requested_matches(data: dict[str, Any] | None, snapshot: dict[str, Any]) -> bool:
    """True when every observable value in data is already the current value."""
    payload = observable_settings(data, snapshot)
    if not payload:
        return True
    for key, value in payload.items():
        if key in {"brightness_pct", "brightness"} and _numbers_close(
            value, snapshot.get("brightness_pct")
        ):
            continue
        current = _attribute(snapshot, key)
        if current is None and key == "brightness":
            current = snapshot.get("brightness_pct")
        if current is None or not _values_match(value, current):
            return False
    return True


def implied_state(action: str) -> str | None:
    return _IMPLIED_STATE.get(str(action or ""))


def _state_is(snapshot: dict[str, Any], expected: str | None) -> bool | None:
    if expected is None:
        return None
    return str(snapshot.get("state") or "").casefold() == expected


def values_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """True when state or comparable attributes changed. Timestamps do not count."""
    if str(before.get("state")) != str(after.get("state")):
        return True
    if before.get("attributes") != after.get("attributes"):
        return True
    return before.get("brightness_pct") != after.get("brightness_pct")


def observation_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """True when anything we wait on, including a new state report, changed."""
    if values_changed(before, after):
        return True
    if before.get("last_reported") != after.get("last_reported"):
        return True
    return before.get("last_updated") != after.get("last_updated")


def expects_change(action: str, data: dict[str, Any] | None, before: dict[str, Any]) -> bool:
    """True when this call should move state or a requested setting."""
    if is_dead_snapshot(before):
        return False
    payload = observable_settings(data, before)
    if str(action or "") == "toggle":
        return True
    if payload and not requested_matches(payload, before):
        return True
    expected = implied_state(action)
    if expected is None:
        return False
    return str(before.get("state") or "").casefold() != expected


def judge_entity(
    action: str,
    data: dict[str, Any] | None,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    """One of changed, already, no_change, not_done, not_confirmed.

    not_confirmed only means the state did not show the change yet. The
    caller turns it into sent when the service call itself succeeded.
    """
    if is_dead_snapshot(before) or is_dead_snapshot(after):
        return "not_done"
    payload = observable_settings(data, before)
    expected = implied_state(action)
    before_state_ok = _state_is(before, expected)
    after_state_ok = _state_is(after, expected)
    settings = bool(payload)
    settings_before = requested_matches(payload, before) if settings else True
    settings_after = requested_matches(payload, after) if settings else True

    if str(action or "") == "toggle":
        if values_changed(before, after) and str(before.get("state")) != str(after.get("state")):
            return "changed"
        return "not_confirmed"

    if not settings and before_state_ok is True:
        if values_changed(before, after) and after_state_ok is not True:
            return "changed"
        return "no_change"

    request_met_after = settings_after and after_state_ok is not False
    request_met_before = settings_before and before_state_ok is not False
    if settings and request_met_before and request_met_after and not (
        expected is not None and before_state_ok is False
    ):
        return "already"
    if not settings and expected is None:
        if values_changed(before, after):
            return "changed"
        return "not_confirmed"
    if request_met_after and (values_changed(before, after) or request_met_before):
        if request_met_before and not values_changed(before, after):
            return "already"
        return "changed"
    return "not_confirmed"


def expected_label(action: str, data: dict[str, Any] | None) -> str:
    """Short text for the debug log: what the check waits for."""
    if str(action or "") == "toggle":
        return "toggled"
    expected = implied_state(action)
    parts = [expected] if expected else []
    if data:
        parts.append("settings " + ",".join(sorted(str(key) for key in data)))
    return " + ".join(parts) or "any change"


def delivered_status(status: str) -> str:
    """Status after a service call that returned without an error.

    A change that did not show up in time is reported as sent, not as a
    failure: slow devices (IR, cloud, devices behind a power meter) often
    report their new state seconds later.
    """
    return "sent" if status == "not_confirmed" else status


def confirm_wait_seconds(value: Any) -> float:
    """The configured longest wait, clamped to a sane range."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return EXPECTED_CHANGE_WAIT_SECONDS
    if math.isnan(seconds):
        return EXPECTED_CHANGE_WAIT_SECONDS
    return max(
        MIN_EXPECTED_CHANGE_WAIT_SECONDS,
        min(MAX_EXPECTED_CHANGE_WAIT_SECONDS, seconds),
    )


def _has_known_outcome(action: str, data: dict[str, Any] | None, before: dict[str, Any]) -> bool:
    if str(action or "") == "toggle":
        return True
    if implied_state(action) is not None:
        return True
    return bool(observable_settings(data, before))


def wait_plan(
    action: str,
    data: dict[str, Any] | None,
    before: dict[str, Any],
    expected_wait: float = EXPECTED_CHANGE_WAIT_SECONDS,
) -> tuple[str, float]:
    """How to wait for one target: (mode, longest wait in seconds).

    expect: a change is expected (for example turn_on of an off device).
        Wait until it shows, up to expected_wait.
    observe: the outcome cannot be predicted from the call. Wait a short
        time for any value to change.
    settle: nothing should change. Wait briefly for a late report.
    """
    if is_dead_snapshot(before):
        return WAIT_SETTLE, 0.0
    if expects_change(action, data, before):
        return WAIT_EXPECT, max(CONFIRM_WAIT_SECONDS, float(expected_wait))
    if _has_known_outcome(action, data, before):
        return WAIT_SETTLE, SETTLED_WAIT_SECONDS
    return WAIT_OBSERVE, CONFIRM_WAIT_SECONDS


def wait_satisfied(
    mode: str,
    action: str,
    data: dict[str, Any] | None,
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    """True when waiting for this target can stop early."""
    if mode == WAIT_EXPECT:
        return judge_entity(action, data, before, after) in {"changed", "already"}
    if mode == WAIT_OBSERVE:
        return values_changed(before, after)
    return False


def _summary(snapshot: dict[str, Any]) -> str:
    parts = [str(snapshot.get("state") or "unknown")]
    attributes = snapshot.get("attributes") or {}
    brightness = snapshot.get("brightness_pct")
    if brightness is not None:
        parts.append(f"{brightness}%")
    for key in _SUMMARY_ATTRS:
        if key == "brightness_pct":
            continue
        if key not in attributes:
            continue
        value = attributes[key]
        if key == "color_temp_kelvin":
            parts.append(f"{value} K")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _label(snapshot: dict[str, Any]) -> str:
    return str(snapshot.get("name") or snapshot.get("entity_id") or "entity")


def _overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_confirmed"
    if any(status == "not_done" for status in statuses):
        return "not_done"
    if any(status == "not_confirmed" for status in statuses):
        return "not_confirmed"
    if any(status == "sent" for status in statuses):
        return "sent"
    if any(status == "changed" for status in statuses):
        return "changed"
    if any(status == "no_change" for status in statuses):
        return "no_change"
    return "already"


def render_verification(
    rows: list[dict[str, Any]],
) -> tuple[str, bool, str]:
    """Turn per-entity judgements into text, isError, and a status token."""
    statuses = [str(row.get("status") or "not_confirmed") for row in rows]
    status = _overall_status(statuses)
    lines = [_STATUS_LABEL[status]]
    if status == "no_change":
        lines.append(
            "No setting values were sent in data. If the user wanted a setting, "
            "call perform_action again with those values in data. "
            "Do not say the action was done."
        )
    elif status == "not_done":
        lines.append(
            "The target is unavailable or unknown, or required setting values "
            "were not sent. Do not say the action was done."
        )
    elif status == "not_confirmed":
        lines.append(
            "The requested change was not confirmed by a state report. "
            "Do not say the action was done."
        )
    elif status == "sent":
        lines.append(
            "The command was delivered without an error, but the device has not "
            "reported the new state yet. Some devices take several seconds, for "
            "example a TV or a device controlled by infrared. Tell the user the "
            "command was sent and the device is switching or should respond "
            "shortly. Do not say it failed, and do not send the command again."
        )
    elif status == "already":
        lines.append("The target already matches the requested settings.")
    else:
        lines.append("The target state changed.")

    for row in rows:
        before = row.get("before") or {}
        after = row.get("after") or before
        row_status = str(row.get("status") or "")
        label = _label(before)
        if row_status == "not_done":
            dead = after if is_dead_snapshot(after) else before
            lines.append(f"{label} is {dead.get('state') or 'unknown'} and was not changed.")
        elif row_status == "no_change":
            lines.append(f"{label} was already {_summary(before)}.")
        elif row_status == "not_confirmed":
            lines.append(f"{label} is still {_summary(after)}.")
        elif row_status == "sent":
            waited = row.get("waited")
            if waited is not None:
                lines.append(
                    f"{label}: command sent; after {waited:.0f} s it still reports "
                    f"{_summary(after)}."
                )
            else:
                lines.append(f"{label}: command sent; it still reports {_summary(after)}.")
        elif row_status == "already":
            lines.append(f"{label} already matches ({_summary(after)}).")
        else:
            lines.append(f"{label}: {_summary(before)} -> {_summary(after)}.")
    return "\n".join(lines), status in _ERROR_STATUSES, status


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compact before/after record for the tool result."""
    published = {
        "entity_id": snapshot.get("entity_id"),
        "name": snapshot.get("name"),
        "state": snapshot.get("state"),
        "last_updated": snapshot.get("last_updated"),
        "last_reported": snapshot.get("last_reported"),
    }
    attributes = snapshot.get("attributes") or {}
    for key in _SUMMARY_ATTRS:
        if key == "brightness_pct" and snapshot.get("brightness_pct") is not None:
            published["brightness_pct"] = snapshot.get("brightness_pct")
            continue
        if key in attributes:
            published[key] = attributes[key]
    return published


def fact_has_numeric_setting(text: str) -> bool:
    """True when the fact contains a number. Units are not required."""
    return _NUMBER.search(text or "") is not None


def _folded_phrase(text: str) -> str:
    return " ".join(fold_token(token) for token in re.findall(r"\w+", text or ""))


def fact_names_entity(fact: str, record: dict[str, Any]) -> bool:
    """True when the fact shares a human name, alias, or area with the entity.

    Area is used only when the entity has no other human name. Matching is
    stem overlap, not a list of device words.
    """
    labels = _human_labels(record, include_area=False)
    if not labels:
        labels = _human_labels(record, include_area=True)
    fact_stems = {stem for stem in content_stems(fact) if len(stem) >= 4}
    folded_fact = _folded_phrase(fact)
    for label in labels:
        label_stems = {stem for stem in content_stems(label) if len(stem) >= 4}
        if label_stems and fact_stems and any(
            stems_match(left, right) for left in label_stems for right in fact_stems
        ):
            return True
        folded_label = _folded_phrase(label)
        if len(folded_label) >= 4 and folded_label in folded_fact:
            return True
    return False


def guard_empty_activation(
    action: str,
    data: dict[str, Any] | None,
    snapshots: list[dict[str, Any]],
    facts: list[str] | None,
) -> str | None:
    """NOT DONE text when an activation has no data but a remembered fact has settings.

    Returns None when the call should proceed. The fact text is quoted from
    this turn; no setting values are stored in code.
    """
    if str(action or "") not in _ACTIVATION_ACTIONS:
        return None
    if data:
        return None
    matched: list[tuple[dict[str, Any], str]] = []
    for snapshot in snapshots:
        if is_dead_snapshot(snapshot):
            continue
        for fact in facts or []:
            text = str(fact or "").strip()
            if not text:
                continue
            if fact_names_entity(text, snapshot) and fact_has_numeric_setting(text):
                matched.append((snapshot, text))
                break
    if not matched:
        return None
    lines = [
        "NOT DONE",
        "No setting values were sent in data, but a remembered fact for this "
        "target includes settings. Call perform_action again with those values "
        "in data. The service was not called. Do not say the action was done.",
    ]
    for snapshot, fact in matched:
        excerpt = fact if len(fact) <= 240 else fact[:240] + "…"
        lines.append(f"{_label(snapshot)}: {excerpt}")
    return "\n".join(lines)


def facts_from_context(context: Any) -> list[str]:
    """Read the facts injected into this turn, if the caller passed them."""
    if not isinstance(context, dict):
        return []
    raw = context.get("injected_facts") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    facts: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text:
            facts.append(text[:500])
        if len(facts) >= 8:
            break
    return facts
