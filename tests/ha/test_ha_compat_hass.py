"""Registry and chat-log adapters against a real Home Assistant core.

Runs on every supported core (2025.10 and newer). Run with:
    PYTHONPATH=. pytest tests/ha -o asyncio_mode=auto
"""

from __future__ import annotations

import logging

import pytest
from homeassistant.components.conversation import chat_log
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    label_registry as lr,
)
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mcp_assist_se.access_hass import snapshot_from_hass
from custom_components.mcp_assist_se.ha_compat import (
    build_tool_result_content,
    device_area_id,
    devices_by_label,
    devices_in_area,
    entity_aliases,
    entity_area_id,
)
from custom_components.mcp_assist_se.index_manager import IndexManager


def _no_deprecation_logged(caplog: pytest.LogCaptureFixture) -> None:
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    assert not [m for m in messages if "deprecat" in m.lower()], messages


@pytest.fixture
def home(hass: HomeAssistant):
    """One area with a labelled device and two entities."""
    entry = MockConfigEntry(domain="test")
    entry.add_to_hass(hass)
    area = ar.async_get(hass).async_create("Kitchen")
    label = lr.async_get(hass).async_create("Night")
    device_reg = dr.async_get(hass)
    device = device_reg.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={("test", "dev1")}, name="Lamp"
    )
    device = device_reg.async_update_device(
        device.id, area_id=area.id, labels={label.label_id}
    )
    entity_reg = er.async_get(hass)
    via_device = entity_reg.async_get_or_create(
        "light", "test", "l1", config_entry=entry, device_id=device.id
    )
    via_device = entity_reg.async_update_entity(
        via_device.entity_id, aliases={"Ceiling light"}
    )
    loose = entity_reg.async_get_or_create("switch", "test", "s1", config_entry=entry)
    return area, label, device, via_device, loose


async def test_entity_and_device_area(hass: HomeAssistant, home) -> None:
    area, _label, device, via_device, loose = home
    assert device_area_id(hass, device) == area.id
    assert entity_area_id(hass, via_device) == area.id
    assert entity_area_id(hass, loose) is None
    assert [d.id for d in devices_in_area(dr.async_get(hass), area.id)] == [device.id]


async def test_devices_by_label(hass: HomeAssistant, home, caplog) -> None:
    _area, label, device, _via_device, _loose = home
    mapping = devices_by_label(dr.async_get(hass), lr.async_get(hass))
    assert mapping == {label.label_id: {device.id}}
    _no_deprecation_logged(caplog)


async def test_entity_aliases_are_plain_strings(hass: HomeAssistant, home) -> None:
    _area, _label, _device, via_device, loose = home
    assert entity_aliases(hass, via_device) == ["Ceiling light"]
    assert entity_aliases(hass, loose) == []


async def test_snapshot_follows_device_area_and_labels(
    hass: HomeAssistant, home, caplog
) -> None:
    area, label, _device, via_device, _loose = home
    hass.states.async_set(via_device.entity_id, "on", {"friendly_name": "Lamp"})
    snapshot = snapshot_from_hass(hass)
    ref = snapshot.entities[via_device.entity_id]
    assert ref.area_id == area.id
    assert ref.area_name == "Kitchen"
    assert label.label_id in ref.label_ids
    assert "Night" in ref.label_names
    _no_deprecation_logged(caplog)


async def test_tool_result_content_builds_and_logs(
    hass: HomeAssistant, caplog
) -> None:
    content = build_tool_result_content(
        "conversation.test", "call-1", "get_state", {"result": "ok"}
    )
    assert isinstance(content, chat_log.ToolResultContent)
    assert content.tool_call_id == "call-1"
    failed = build_tool_result_content(
        "conversation.test", "call-2", "get_state", "ERROR: nope"
    )
    assert failed.tool_name == "get_state"
    _no_deprecation_logged(caplog)


async def test_script_fields_come_from_action_descriptions(
    hass: HomeAssistant,
) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "greet": {
                    "fields": {"who": {"description": "Person to greet"}},
                    "sequence": [],
                }
            }
        },
    )
    await hass.async_block_till_done()
    entity_reg = er.async_get(hass)
    entity_reg.async_update_entity("script.greet", new_entity_id="script.renamed")
    await hass.async_block_till_done()
    async_expose_entity(hass, "conversation", "script.renamed", True)

    scripts = await IndexManager(hass)._get_scripts()
    assert scripts == [
        {
            "id": "renamed",
            "name": scripts[0]["name"],
            "fields": {"who": {"description": "Person to greet"}},
        }
    ]
