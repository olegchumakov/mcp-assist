"""Discovery shows state with unit and key attributes compactly."""

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

from mcp_assist_se.state_summary import (
    MAX_SUMMARY_ATTRIBUTES,
    format_display_name,
    summarize_attributes,
    summarize_state,
)


class StateSummaryTests(unittest.TestCase):
    def test_light_on(self):
        self.assertEqual(
            summarize_state("on", {"brightness": 153, "color_temp_kelvin": 2700, "friendly_name": "L"}),
            "on (brightness 60%, color temp 2700K)",
        )

    def test_light_off_has_no_extras(self):
        self.assertEqual(summarize_state("off", {"brightness": None, "friendly_name": "L"}), "off")

    def test_sensor_unit(self):
        self.assertEqual(
            summarize_state("21.5", {"unit_of_measurement": "°C", "device_class": "temperature"}),
            "21.5 °C",
        )

    def test_unavailable_has_no_unit(self):
        self.assertEqual(summarize_state("unavailable", {"unit_of_measurement": "°C"}), "unavailable")
        self.assertEqual(summarize_state(None, None), "unknown")

    def test_climate(self):
        self.assertEqual(
            summarize_state("heat", {"current_temperature": 20.5, "temperature": 22, "hvac_action": "heating"}),
            "heat (current temp 20.5, target temp 22, action heating)",
        )

    def test_media_and_cover(self):
        self.assertEqual(
            summarize_attributes({"volume_level": 0.35, "media_title": "x" * 100}),
            ["volume 35%", "playing " + "x" * 39 + "…"],
        )
        self.assertEqual(summarize_attributes({"current_position": 40}), ["position 40%"])

    def test_cap(self):
        attrs = {
            "brightness": 255, "color_temp_kelvin": 3000, "current_temperature": 1,
            "temperature": 2, "current_humidity": 3, "hvac_action": "idle",
            "current_position": 5, "percentage": 6, "battery_level": 7,
        }
        self.assertEqual(len(summarize_attributes(attrs)), MAX_SUMMARY_ATTRIBUTES)

    def test_bad_values_are_skipped(self):
        self.assertEqual(summarize_attributes({"brightness": "x", "volume_level": "y"}), [])

    def test_display_name(self):
        self.assertEqual(format_display_name("Люстра"), "Люстра")
        self.assertEqual(
            format_display_name("Relay L1", ["Люстра", "relay l1", "Люстра", "", "a", "b"]),
            "Relay L1 (aka Люстра, a, b)",
        )
        self.assertEqual(format_display_name(None, None), "")


if __name__ == "__main__":
    unittest.main()
