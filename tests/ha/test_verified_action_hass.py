"""perform_action waiting against a real Home Assistant core.

Needs pytest-homeassistant-custom-component. Run with:
    PYTHONPATH=. pytest tests/ha -o asyncio_mode=auto
The plain unittest suite (python -m unittest discover -s tests) skips this folder.
"""

from __future__ import annotations

import asyncio

from homeassistant.core import HomeAssistant, ServiceCall
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mcp_assist_se.const import (
    CONF_ACTION_CONFIRM_SECONDS,
    DOMAIN,
    SYSTEM_ENTRY_UNIQUE_ID,
)
from custom_components.mcp_assist_se.mcp_server import MCPServer

TV = "media_player.test_tv"
LAMP = "light.test_lamp"


def _server(hass: HomeAssistant) -> MCPServer:
    server = object.__new__(MCPServer)
    server.hass = hass
    server.entry = None
    server.publish_progress = lambda *args, **kwargs: None
    return server


def _add_system_entry(hass: HomeAssistant, seconds: int) -> None:
    MockConfigEntry(
        domain=DOMAIN,
        unique_id=SYSTEM_ENTRY_UNIQUE_ID,
        data={},
        options={CONF_ACTION_CONFIRM_SECONDS: seconds},
    ).add_to_hass(hass)


def _register_slow_turn_on(hass: HomeAssistant, delays: dict[str, float]) -> None:
    """A turn_on that returns at once and reports the new state later."""

    async def _turn_on(call: ServiceCall) -> None:
        for entity_id in call.data["entity_id"]:
            delay = delays.get(entity_id)
            if delay is None:
                continue
            hass.loop.call_later(
                delay,
                hass.states.async_set,
                entity_id,
                "on",
                {"friendly_name": entity_id},
            )

    hass.services.async_register("media_player", "turn_on", _turn_on)


async def _run(hass: HomeAssistant, entity_ids: list[str]) -> dict:
    return await _server(hass)._execute_verified_action(
        domain="media_player",
        action="turn_on",
        service="turn_on",
        data={},
        resolved_target={"entity_id": entity_ids},
        context=None,
    )


async def test_slow_device_is_confirmed_within_the_wait(hass: HomeAssistant) -> None:
    _add_system_entry(hass, 3)
    hass.states.async_set(TV, "off", {"friendly_name": "Test TV"})
    _register_slow_turn_on(hass, {TV: 1.0})
    start = asyncio.get_running_loop().time()
    result = await _run(hass, [TV])
    elapsed = asyncio.get_running_loop().time() - start
    assert result["action_result"]["status"] == "changed"
    assert not result.get("isError")
    assert 0.9 <= elapsed < 2.5


async def test_no_report_within_the_wait_is_sent_not_failed(
    hass: HomeAssistant,
) -> None:
    _add_system_entry(hass, 2)
    hass.states.async_set(TV, "off", {"friendly_name": "Test TV"})
    _register_slow_turn_on(hass, {})
    result = await _run(hass, [TV])
    assert result["action_result"]["status"] == "sent"
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert text.startswith("SENT")
    assert "Do not say it failed" in text


async def test_fast_device_does_not_wait_for_the_slow_one(hass: HomeAssistant) -> None:
    _add_system_entry(hass, 3)
    hass.states.async_set(TV, "off", {"friendly_name": "Test TV"})
    hass.states.async_set(LAMP, "off", {"friendly_name": "Test lamp"})
    _register_slow_turn_on(hass, {LAMP: 0.05, TV: 0.8})
    result = await _run(hass, [TV, LAMP])
    statuses = {
        row["entity_id"]: row["status"] for row in result["action_result"]["entities"]
    }
    assert statuses == {TV: "changed", LAMP: "changed"}


async def test_unavailable_target_still_fails(hass: HomeAssistant) -> None:
    hass.states.async_set(TV, "unavailable", {"friendly_name": "Test TV"})
    _register_slow_turn_on(hass, {})
    result = await _run(hass, [TV])
    assert result["action_result"]["status"] == "not_done"
    assert result["isError"]
