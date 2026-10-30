"""Per-target waiting woken by state events, without Home Assistant."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_components" / "mcp_assist_se"
sys.path.insert(0, str(ROOT / "custom_components"))

if "mcp_assist_se" not in sys.modules:
    package = types.ModuleType("mcp_assist_se")
    package.__path__ = [str(PACKAGE)]
    package.__package__ = "mcp_assist_se"
    sys.modules["mcp_assist_se"] = package

from mcp_assist_se.action_verify import (
    WAIT_EXPECT,
    WAIT_OBSERVE,
    snapshot_from_mapping,
    wait_satisfied,
)
from mcp_assist_se.action_wait import wait_for_targets


def _snap(entity_id: str, state: str) -> dict:
    return snapshot_from_mapping(
        {"entity_id": entity_id, "state": state, "name": entity_id, "attributes": {}}
    )


class FakeStates:
    """Minimal state store with a state-changed subscription."""

    def __init__(self, states: dict[str, str]) -> None:
        self.states = dict(states)
        self.listeners: list = []
        self.unsubscribed = 0

    def read(self, entity_id: str) -> dict:
        return _snap(entity_id, self.states[entity_id])

    def subscribe(self, entity_ids, on_change):
        entry = (set(entity_ids), on_change)
        self.listeners.append(entry)

        def _unsub() -> None:
            self.unsubscribed += 1
            self.listeners.remove(entry)

        return _unsub

    def set(self, entity_id: str, state: str) -> None:
        self.states[entity_id] = state
        for ids, on_change in list(self.listeners):
            if entity_id in ids:
                on_change({"entity_id": entity_id})


def _satisfied(action: str):
    def _check(_entity_id, mode, before, after):
        return wait_satisfied(mode, action, {}, before, after)

    return _check


class WaitForTargetsTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_device_ends_the_wait_on_its_event(self) -> None:
        store = FakeStates({"light.a": "off"})
        before = {"light.a": store.read("light.a")}
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, store.set, "light.a", "on")
        start = loop.time()
        latest, waited, confirmed = await wait_for_targets(
            ["light.a"],
            before,
            {"light.a": (WAIT_EXPECT, 5.0)},
            store.read,
            _satisfied("turn_on"),
            store.subscribe,
        )
        self.assertLess(loop.time() - start, 1.0)
        self.assertEqual(latest["light.a"]["state"], "on")
        self.assertTrue(confirmed["light.a"])
        self.assertLess(waited["light.a"], 1.0)
        self.assertEqual(store.unsubscribed, 1)
        self.assertEqual(store.listeners, [])

    async def test_each_target_is_waited_on_separately(self) -> None:
        store = FakeStates({"media_player.tv": "off", "light.b": "off"})
        ids = ["media_player.tv", "light.b"]
        before = {entity_id: store.read(entity_id) for entity_id in ids}
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, store.set, "light.b", "on")
        loop.call_later(0.3, store.set, "media_player.tv", "on")
        latest, waited, confirmed = await wait_for_targets(
            ids,
            before,
            {entity_id: (WAIT_EXPECT, 3.0) for entity_id in ids},
            store.read,
            _satisfied("turn_on"),
            store.subscribe,
        )
        self.assertEqual(latest["media_player.tv"]["state"], "on")
        self.assertEqual(latest["light.b"]["state"], "on")
        self.assertTrue(all(confirmed.values()))
        self.assertLess(waited["light.b"], waited["media_player.tv"])

    async def test_slow_device_times_out_without_confirmation(self) -> None:
        store = FakeStates({"media_player.tv": "off"})
        before = {"media_player.tv": store.read("media_player.tv")}
        latest, waited, confirmed = await wait_for_targets(
            ["media_player.tv"],
            before,
            {"media_player.tv": (WAIT_EXPECT, 0.15)},
            store.read,
            _satisfied("turn_on"),
            store.subscribe,
        )
        self.assertFalse(confirmed["media_player.tv"])
        self.assertGreaterEqual(waited["media_player.tv"], 0.14)
        self.assertEqual(latest["media_player.tv"]["state"], "off")
        self.assertEqual(store.unsubscribed, 1)

    async def test_unrelated_event_does_not_end_an_expected_wait(self) -> None:
        store = FakeStates({"cover.c": "closed"})
        before = {"cover.c": store.read("cover.c")}
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, store.set, "cover.c", "opening")
        loop.call_later(0.2, store.set, "cover.c", "open")
        latest, _waited, confirmed = await wait_for_targets(
            ["cover.c"],
            before,
            {"cover.c": (WAIT_EXPECT, 3.0)},
            store.read,
            _satisfied("open"),
            store.subscribe,
        )
        self.assertEqual(latest["cover.c"]["state"], "open")
        self.assertTrue(confirmed["cover.c"])

    async def test_observe_mode_stops_on_any_value_change(self) -> None:
        store = FakeStates({"media_player.tv": "on"})
        before = {"media_player.tv": store.read("media_player.tv")}
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, store.set, "media_player.tv", "playing")
        _latest, waited, confirmed = await wait_for_targets(
            ["media_player.tv"],
            before,
            {"media_player.tv": (WAIT_OBSERVE, 2.0)},
            store.read,
            _satisfied("media_play"),
            store.subscribe,
        )
        self.assertTrue(confirmed["media_player.tv"])
        self.assertLess(waited["media_player.tv"], 1.0)

    async def test_no_targets_returns_at_once(self) -> None:
        store = FakeStates({})
        latest, waited, confirmed = await wait_for_targets(
            [], {}, {}, store.read, _satisfied("turn_on"), store.subscribe
        )
        self.assertEqual((latest, waited, confirmed), ({}, {}, {}))


if __name__ == "__main__":
    unittest.main()
