"""Compact, model-friendly summary of an entity's current state.

Discovery results include this so the model can answer "is X on?" or
"what is the temperature?" without a separate get_entity_details round trip.
"""

from __future__ import annotations

from typing import Any, Mapping

MAX_SUMMARY_ATTRIBUTES = 6
MAX_VALUE_CHARS = 40

_UNAVAILABLE = {"unavailable", "unknown", "none", ""}


def _num(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 1] + "…"


def _percent_from_255(value: Any) -> str | None:
    try:
        return f"{round(float(value) * 100 / 255)}%"
    except (TypeError, ValueError):
        return None


def _fraction_to_percent(value: Any) -> str | None:
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return None


def summarize_attributes(attributes: Mapping[str, Any] | None) -> list[str]:
    """Return short 'name value' strings for the attributes worth showing."""
    if not attributes:
        return []
    attrs = attributes
    parts: list[str] = []

    def add(label: str, value: Any, unit: str = "") -> None:
        if value is None or len(parts) >= MAX_SUMMARY_ATTRIBUTES:
            return
        text = _clip(f"{_num(value)}{unit}")
        parts.append(f"{label} {text}" if label else text)

    if attrs.get("brightness") is not None:
        pct = _percent_from_255(attrs.get("brightness"))
        if pct:
            add("brightness", pct)
    if attrs.get("color_temp_kelvin") is not None:
        add("color temp", attrs.get("color_temp_kelvin"), "K")
    if attrs.get("current_temperature") is not None:
        add("current temp", attrs.get("current_temperature"))
    if attrs.get("temperature") is not None:
        add("target temp", attrs.get("temperature"))
    if attrs.get("target_temp_low") is not None and attrs.get("target_temp_high") is not None:
        add("target range", f"{_num(attrs['target_temp_low'])}-{_num(attrs['target_temp_high'])}")
    if attrs.get("current_humidity") is not None:
        add("humidity", attrs.get("current_humidity"), "%")
    if attrs.get("hvac_action") is not None:
        add("action", attrs.get("hvac_action"))
    if attrs.get("current_position") is not None:
        add("position", attrs.get("current_position"), "%")
    if attrs.get("percentage") is not None:
        add("speed", attrs.get("percentage"), "%")
    if attrs.get("volume_level") is not None:
        pct = _fraction_to_percent(attrs.get("volume_level"))
        if pct:
            add("volume", pct)
    if attrs.get("media_title") is not None:
        add("playing", attrs.get("media_title"))
    if attrs.get("battery_level") is not None:
        add("battery", attrs.get("battery_level"), "%")
    return parts


def summarize_state(state: Any, attributes: Mapping[str, Any] | None = None) -> str:
    """State with unit plus key attributes, e.g. 'on (brightness 60%, color temp 2700K)'."""
    attrs = attributes or {}
    state_text = "unknown" if state is None else str(state)
    unit = attrs.get("unit_of_measurement")
    if unit and state_text.lower() not in _UNAVAILABLE:
        state_text = f"{state_text} {unit}"
    if state_text.lower() in _UNAVAILABLE:
        return state_text
    extras = summarize_attributes(attrs)
    return f"{state_text} ({', '.join(extras)})" if extras else state_text


MAX_ALIASES = 3


def format_display_name(name: Any, aliases: Any = None) -> str:
    """Name with up to a few aliases, e.g. 'Ceiling (aka Chandelier, Main light)'."""
    base = str(name or "").strip()
    extra: list[str] = []
    for alias in aliases or ():
        text = str(alias or "").strip()
        if text and text.casefold() != base.casefold() and text not in extra:
            extra.append(text)
        if len(extra) >= MAX_ALIASES:
            break
    return f"{base} (aka {', '.join(extra)})" if extra else base
