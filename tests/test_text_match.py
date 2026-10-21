"""Name/area matching tolerates Latin look-alike letters inside Cyrillic words."""

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

from mcp_assist_se.text_match import (
    any_key_equal,
    key_contains,
    keys_equal,
    match_key,
)

LATIN_A = "a"  # U+0061


class MatchKeyTests(unittest.TestCase):
    def test_mixed_script_area_matches(self):
        typed = "Гостин" + LATIN_A + "я"
        self.assertNotEqual(typed.casefold(), "Гостиная".casefold())
        self.assertTrue(keys_equal(typed, "Гостиная"))
        self.assertTrue(keys_equal("Гостиная", typed))

    def test_every_lookalike_maps(self):
        latin = "aeopcyxkmthbAEOPCYXKMTHB"
        cyrillic = "аеорсухкмтнвАЕОРСУХКМТНВ"
        self.assertEqual(match_key(latin), match_key(cyrillic))

    def test_yo_is_ye(self):
        self.assertTrue(keys_equal("Ёлка", "елка"))

    def test_pure_latin_unchanged_semantics(self):
        self.assertTrue(keys_equal("Kitchen", "kitchen"))
        self.assertTrue(keys_equal("Living Room", "living  room"))
        self.assertFalse(keys_equal("Kitchen", "Bathroom"))
        self.assertTrue(key_contains("Kitchen Ceiling Light", "ceiling"))
        self.assertFalse(key_contains("Kitchen", "garage"))

    def test_substring_mixed(self):
        self.assertTrue(key_contains("Люстра в гостиной", "люстр"))
        self.assertTrue(key_contains("Люстра", "Люстp" + LATIN_A))  # Latin p and a

    def test_empty(self):
        self.assertFalse(keys_equal("", ""))
        self.assertFalse(keys_equal(None, None))
        self.assertFalse(key_contains("abc", ""))
        self.assertEqual(match_key(None), "")

    def test_aliases(self):
        self.assertTrue(any_key_equal("Зал", ["Гостиная", "зал"]))
        self.assertFalse(any_key_equal("Кухня", None))


if __name__ == "__main__":
    unittest.main()
