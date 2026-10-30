"""Post-state checks, remembered-setting guard, and unnamed-entity filtering."""

from __future__ import annotations

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
    CONFIRM_WAIT_SECONDS,
    SETTLED_WAIT_SECONDS,
    WAIT_EXPECT,
    WAIT_OBSERVE,
    WAIT_SETTLE,
    confirm_wait_seconds,
    delivered_status,
    expected_label,
    fact_names_entity,
    guard_empty_activation,
    judge_entity,
    listing_disposition,
    prepare_listed_entity,
    render_verification,
    snapshot_from_mapping,
    wait_plan,
    wait_satisfied,
)
from mcp_assist_se.const import (
    ACTION_RESULT_INSTRUCTIONS,
    DEFAULT_TECHNICAL_PROMPT,
    PERFORM_ACTION_DATA_DESCRIPTION,
)

# Fixture only. Production code must not contain this home's ids or phrases.
STRIP_FACT = "Лента в гостиной: яркость 100%, 2000 K"
IEEE_ID = "light.0xa4c1388a80ba7f4f"


def _lamp(**overrides):
    raw = {
        "entity_id": "light.kitchen_lamp",
        "state": "on",
        "name": "Kitchen lamp",
        "friendly_name": "Kitchen lamp",
        "area": "Kitchen",
        "aliases": [],
        "attributes": {
            "friendly_name": "Kitchen lamp",
            "brightness": 158,
            "color_temp_kelvin": 2000,
        },
        "last_updated": "2026-10-10T18:00:00+00:00",
        "last_reported": "2026-10-10T18:00:00+00:00",
    }
    raw.update(overrides)
    if "attributes" in overrides:
        raw["attributes"] = overrides["attributes"]
    return snapshot_from_mapping(raw)


def _judge(action, data, before, after):
    status = judge_entity(action, data, before, after)
    text, is_error, overall = render_verification(
        [{"status": status, "before": before, "after": after}]
    )
    return status, text, is_error, overall


class ActionVerifyTests(unittest.TestCase):
    def test_brightness_percent_comes_from_the_attribute(self) -> None:
        snapshot = _lamp()
        self.assertEqual(snapshot["brightness_pct"], 62)
        self.assertEqual(snapshot["attributes"]["color_temp_kelvin"], 2000)

    def test_bare_turn_on_of_an_already_on_device_is_no_change(self) -> None:
        before = _lamp()
        status, text, is_error, overall = _judge("turn_on", {}, before, before)
        self.assertEqual(status, "no_change")
        self.assertEqual(overall, "no_change")
        self.assertTrue(is_error)
        self.assertIn("NO CHANGE", text)
        self.assertIn("62%", text)
        self.assertIn("2000 K", text)
        self.assertIn("No setting values were sent", text)
        self.assertNotIn("Successfully executed", text)

    def test_turn_on_from_off_is_changed(self) -> None:
        before = _lamp(state="off", attributes={"brightness": 0, "friendly_name": "Kitchen lamp"})
        after = _lamp()
        status, text, is_error, _overall = _judge("turn_on", {}, before, after)
        self.assertEqual(status, "changed")
        self.assertFalse(is_error)
        self.assertTrue(text.startswith("CHANGED"))

    def test_requested_setting_that_never_arrives_is_not_confirmed(self) -> None:
        before = _lamp()
        status, text, is_error, _overall = _judge(
            "turn_on",
            {"brightness_pct": 80, "color_temp_kelvin": 4000},
            before,
            before,
        )
        self.assertEqual(status, "not_confirmed")
        self.assertTrue(is_error)
        self.assertIn("NOT CONFIRMED", text)

    def test_requested_setting_that_lands_is_changed(self) -> None:
        before = _lamp()
        after = _lamp(
            attributes={
                "friendly_name": "Kitchen lamp",
                "brightness": 255,
                "color_temp_kelvin": 4000,
            },
            last_updated="2026-10-10T18:00:02+00:00",
            last_reported="2026-10-10T18:00:02+00:00",
        )
        status, text, is_error, _overall = _judge(
            "turn_on",
            {"brightness_pct": 100, "color_temp_kelvin": 4000},
            before,
            after,
        )
        self.assertEqual(status, "changed")
        self.assertFalse(is_error)
        self.assertIn("CHANGED", text)

    def test_already_at_the_requested_setting_is_already(self) -> None:
        current = _lamp(
            attributes={
                "friendly_name": "Kitchen lamp",
                "brightness": 255,
                "color_temp_kelvin": 4000,
            }
        )
        status, text, is_error, _overall = _judge(
            "turn_on",
            {"brightness_pct": 100, "color_temp_kelvin": 4000},
            current,
            current,
        )
        self.assertEqual(status, "already")
        self.assertFalse(is_error)
        self.assertTrue(text.startswith("ALREADY"))

    def test_unavailable_target_is_not_done(self) -> None:
        before = _lamp(state="unavailable")
        status, text, is_error, _overall = _judge("turn_on", {}, before, before)
        self.assertEqual(status, "not_done")
        self.assertTrue(is_error)
        self.assertIn("NOT DONE", text)

    def test_unknown_target_is_not_done(self) -> None:
        before = snapshot_from_mapping(
            {
                "entity_id": "light.kitchen_lamp",
                "missing": True,
                "name": "Kitchen lamp",
                "friendly_name": "Kitchen lamp",
            }
        )
        self.assertEqual(judge_entity("turn_on", {}, before, before), "not_done")

    def test_climate_setting_is_confirmed_from_attributes(self) -> None:
        before = snapshot_from_mapping(
            {
                "entity_id": "climate.hall",
                "state": "heat",
                "name": "Hall thermostat",
                "friendly_name": "Hall thermostat",
                "attributes": {"temperature": 19, "friendly_name": "Hall thermostat"},
            }
        )
        after = snapshot_from_mapping(
            {
                "entity_id": "climate.hall",
                "state": "heat",
                "name": "Hall thermostat",
                "friendly_name": "Hall thermostat",
                "attributes": {"temperature": 21, "friendly_name": "Hall thermostat"},
                "last_updated": "2026-10-10T18:00:02+00:00",
            }
        )
        self.assertEqual(
            judge_entity("set_temperature", {"temperature": 21}, before, after),
            "changed",
        )
        self.assertEqual(
            judge_entity("set_temperature", {"temperature": 21}, after, after),
            "already",
        )

    def test_timestamp_alone_does_not_count_as_a_device_change(self) -> None:
        before = _lamp()
        after = _lamp(last_updated="2026-10-10T18:00:02+00:00")
        self.assertEqual(judge_entity("turn_on", {}, before, after), "no_change")


class RememberedSettingGuardTests(unittest.TestCase):
    def test_injected_fact_with_a_number_blocks_empty_activation(self) -> None:
        snapshot = snapshot_from_mapping(
            {
                "entity_id": "light.hall_strip",
                "state": "on",
                "name": "Лента",
                "friendly_name": "Лента",
                "area": "Гостиная",
                "attributes": {"friendly_name": "Лента", "brightness": 158},
            }
        )
        text = guard_empty_activation("turn_on", {}, [snapshot], [STRIP_FACT])
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("NOT DONE", text)
        self.assertIn("100%", text)
        self.assertIn("not called", text.casefold())

    def test_fact_that_does_not_name_the_entity_does_not_block(self) -> None:
        snapshot = _lamp(name="Ceiling fan", friendly_name="Ceiling fan", area="Hall")
        snapshot = snapshot_from_mapping(
            {
                "entity_id": "fan.hall",
                "state": "on",
                "name": "Ceiling fan",
                "friendly_name": "Ceiling fan",
                "area": "Hall",
                "attributes": {"friendly_name": "Ceiling fan"},
            }
        )
        self.assertFalse(fact_names_entity(STRIP_FACT, snapshot))
        self.assertIsNone(guard_empty_activation("turn_on", {}, [snapshot], [STRIP_FACT]))

    def test_fact_without_a_number_does_not_block(self) -> None:
        snapshot = _lamp()
        fact = "The kitchen lamp is next to the window"
        self.assertTrue(fact_names_entity(fact, snapshot))
        self.assertIsNone(guard_empty_activation("turn_on", {}, [snapshot], [fact]))

    def test_area_alone_does_not_block_a_named_entity(self) -> None:
        snapshot = snapshot_from_mapping(
            {
                "entity_id": "light.ceiling",
                "state": "on",
                "name": "Ceiling",
                "friendly_name": "Ceiling",
                "area": "Kitchen",
                "attributes": {"friendly_name": "Ceiling"},
            }
        )
        fact = "Kitchen brightness 80%"
        self.assertFalse(fact_names_entity(fact, snapshot))
        self.assertIsNone(guard_empty_activation("turn_on", {}, [snapshot], [fact]))

    def test_data_already_present_is_not_blocked(self) -> None:
        snapshot = _lamp()
        self.assertIsNone(
            guard_empty_activation(
                "turn_on",
                {"brightness_pct": 50},
                [snapshot],
                ["Kitchen lamp brightness 50%"],
            )
        )

    def test_turn_off_is_not_blocked(self) -> None:
        snapshot = _lamp()
        self.assertIsNone(
            guard_empty_activation(
                "turn_off",
                {},
                [snapshot],
                ["Kitchen lamp brightness 50%"],
            )
        )


class ListingFilterTests(unittest.TestCase):
    def test_unavailable_ieee_without_a_human_name_is_skipped(self) -> None:
        info = {
            "entity_id": IEEE_ID,
            "state": "unavailable",
            "name": "0xa4c1388a80ba7f4f",
            "aliases": [],
            "area": "",
        }
        self.assertEqual(listing_disposition(info), "skip")
        self.assertIsNone(prepare_listed_entity(info))

    def test_unavailable_entity_id_as_its_only_name_is_skipped(self) -> None:
        info = {
            "entity_id": IEEE_ID,
            "state": "unknown",
            "name": IEEE_ID,
            "friendly_name": IEEE_ID,
        }
        self.assertIsNone(prepare_listed_entity(info))

    def test_unavailable_named_entity_is_marked(self) -> None:
        info = {
            "entity_id": "light.kitchen_lamp",
            "state": "unavailable",
            "name": "Kitchen lamp",
            "friendly_name": "Kitchen lamp",
            "area": "Kitchen",
        }
        listed = prepare_listed_entity(info)
        self.assertIsNotNone(listed)
        assert listed is not None
        self.assertIn("unavailable", listed["state"].casefold())

    def test_unavailable_named_unknown_state_is_marked(self) -> None:
        info = {
            "entity_id": "switch.porch",
            "state": "unknown",
            "name": "Porch switch",
            "area": "Porch",
        }
        listed = prepare_listed_entity(info)
        assert listed is not None
        self.assertEqual(listed["state"], "unknown (unavailable)")

    def test_available_ieee_entity_is_kept(self) -> None:
        info = {
            "entity_id": IEEE_ID,
            "state": "on",
            "name": "0xa4c1388a80ba7f4f",
        }
        listed = prepare_listed_entity(info)
        self.assertEqual(listed, info)

    def test_unavailable_with_only_an_area_is_kept(self) -> None:
        info = {
            "entity_id": IEEE_ID,
            "state": "unavailable",
            "name": "0xa4c1388a80ba7f4f",
            "area": "Kitchen",
        }
        self.assertEqual(listing_disposition(info), "mark")
        self.assertIsNotNone(prepare_listed_entity(info))


class SentStatusTests(unittest.TestCase):
    def _tv(self, state: str, **extra):
        raw = {
            "entity_id": "media_player.living_room_tv",
            "state": state,
            "name": "Living room TV",
            "friendly_name": "Living room TV",
            "attributes": {"friendly_name": "Living room TV"},
        }
        raw.update(extra)
        return snapshot_from_mapping(raw)

    def test_unconfirmed_change_after_a_good_call_is_sent_not_an_error(self) -> None:
        before = self._tv("off")
        status = delivered_status(judge_entity("turn_on", {}, before, before))
        self.assertEqual(status, "sent")
        text, is_error, overall = render_verification(
            [{"status": status, "before": before, "after": before, "waited": 10.0}]
        )
        self.assertEqual(overall, "sent")
        self.assertFalse(is_error)
        self.assertTrue(text.startswith("SENT"))
        self.assertIn("Do not say it failed", text)
        self.assertIn("do not send the command again", text)
        self.assertIn("after 10 s", text)
        self.assertNotIn("Do not say the action was done", text)

    def test_delivered_status_keeps_other_outcomes(self) -> None:
        for status in ("changed", "already", "no_change", "not_done"):
            self.assertEqual(delivered_status(status), status)

    def test_unavailable_still_fails(self) -> None:
        before = self._tv("unavailable")
        status = delivered_status(judge_entity("turn_on", {}, before, before))
        self.assertEqual(status, "not_done")
        _text, is_error, _overall = render_verification(
            [{"status": status, "before": before, "after": before}]
        )
        self.assertTrue(is_error)

    def test_target_that_goes_unavailable_after_the_call_is_not_done(self) -> None:
        before = self._tv("off")
        after = self._tv("unavailable")
        self.assertEqual(judge_entity("turn_on", {}, before, after), "not_done")
        text, is_error, _overall = render_verification(
            [{"status": "not_done", "before": before, "after": after}]
        )
        self.assertTrue(is_error)
        self.assertIn("is unavailable", text)

    def test_mixed_rows_report_sent_overall(self) -> None:
        tv = self._tv("off")
        lamp_before = _lamp(state="off", attributes={"friendly_name": "Kitchen lamp"})
        lamp_after = _lamp()
        _text, is_error, overall = render_verification(
            [
                {"status": "sent", "before": tv, "after": tv},
                {"status": "changed", "before": lamp_before, "after": lamp_after},
            ]
        )
        self.assertEqual(overall, "sent")
        self.assertFalse(is_error)


class WaitPlanTests(unittest.TestCase):
    def test_expected_change_waits_up_to_the_configured_time(self) -> None:
        before = _lamp(state="off", attributes={"friendly_name": "Kitchen lamp"})
        self.assertEqual(wait_plan("turn_on", {}, before, 9), (WAIT_EXPECT, 9.0))
        self.assertEqual(wait_plan("toggle", {}, before, 9), (WAIT_EXPECT, 9.0))

    def test_expected_wait_is_never_shorter_than_the_quick_wait(self) -> None:
        before = _lamp(state="off", attributes={"friendly_name": "Kitchen lamp"})
        self.assertEqual(
            wait_plan("turn_on", {}, before, 0.5), (WAIT_EXPECT, CONFIRM_WAIT_SECONDS)
        )

    def test_nothing_to_change_only_settles(self) -> None:
        before = _lamp()
        self.assertEqual(
            wait_plan("turn_on", {}, before, 10), (WAIT_SETTLE, SETTLED_WAIT_SECONDS)
        )

    def test_unpredictable_action_gets_the_quick_wait(self) -> None:
        before = _lamp()
        self.assertEqual(
            wait_plan("volume_up", {}, before, 10), (WAIT_OBSERVE, CONFIRM_WAIT_SECONDS)
        )

    def test_satisfied_needs_the_expected_state_not_just_any_report(self) -> None:
        before = snapshot_from_mapping(
            {"entity_id": "cover.door", "state": "closed", "name": "Door"}
        )
        opening = snapshot_from_mapping(
            {"entity_id": "cover.door", "state": "opening", "name": "Door"}
        )
        opened = snapshot_from_mapping(
            {"entity_id": "cover.door", "state": "open", "name": "Door"}
        )
        self.assertFalse(wait_satisfied(WAIT_EXPECT, "open", {}, before, opening))
        self.assertTrue(wait_satisfied(WAIT_EXPECT, "open", {}, before, opened))
        self.assertTrue(wait_satisfied(WAIT_OBSERVE, "open", {}, before, opening))
        self.assertFalse(wait_satisfied(WAIT_SETTLE, "open", {}, before, opened))

    def test_confirm_wait_is_clamped(self) -> None:
        self.assertEqual(confirm_wait_seconds(10), 10.0)
        self.assertEqual(confirm_wait_seconds("8"), 8.0)
        self.assertEqual(confirm_wait_seconds(0), 2.0)
        self.assertEqual(confirm_wait_seconds(999), 30.0)
        self.assertEqual(confirm_wait_seconds(None), 10.0)
        self.assertEqual(confirm_wait_seconds("abc"), 10.0)

    def test_expected_label(self) -> None:
        self.assertEqual(expected_label("turn_on", {}), "on")
        self.assertEqual(expected_label("toggle", {}), "toggled")
        self.assertEqual(
            expected_label("turn_on", {"brightness_pct": 50}), "on + settings brightness_pct"
        )
        self.assertEqual(expected_label("volume_up", None), "any change")


class PromptRuleTests(unittest.TestCase):
    def test_rules_are_generic(self) -> None:
        blob = "\n".join(
            [
                ACTION_RESULT_INSTRUCTIONS,
                PERFORM_ACTION_DATA_DESCRIPTION,
                DEFAULT_TECHNICAL_PROMPT,
            ]
        )
        lowered = blob.casefold()
        self.assertNotIn("0xa4c", lowered)
        self.assertNotIn("лента", lowered)
        self.assertNotIn("яркий", lowered)
        self.assertNotIn("2000", blob)
        self.assertIn("CHANGED", ACTION_RESULT_INSTRUCTIONS)
        self.assertIn("NO CHANGE", ACTION_RESULT_INSTRUCTIONS)
        self.assertIn("NOT DONE", ACTION_RESULT_INSTRUCTIONS)
        self.assertIn("NOT CONFIRMED", ACTION_RESULT_INSTRUCTIONS)
        self.assertIn("SENT", ACTION_RESULT_INSTRUCTIONS)
        self.assertIn("only to that device", ACTION_RESULT_INSTRUCTIONS)
        self.assertIn("only to that device", DEFAULT_TECHNICAL_PROMPT)
        self.assertIn("brightness_pct", PERFORM_ACTION_DATA_DESCRIPTION)
        self.assertIn("50", PERFORM_ACTION_DATA_DESCRIPTION)
        self.assertIn("4000", PERFORM_ACTION_DATA_DESCRIPTION)


if __name__ == "__main__":
    unittest.main()
