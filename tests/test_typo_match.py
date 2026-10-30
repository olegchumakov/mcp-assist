"""Typo-tolerant fallback for area, floor, label and name matching."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_components" / "mcp_assist_se"
sys.path.insert(0, str(ROOT / "custom_components"))

if "mcp_assist_se" not in sys.modules:
    package = types.ModuleType("mcp_assist_se")
    package.__path__ = [str(PACKAGE)]
    package.__package__ = "mcp_assist_se"
    sys.modules["mcp_assist_se"] = package


def _stub_homeassistant() -> None:
    """Minimal stand-ins so discovery.py imports without Home Assistant."""
    try:
        import homeassistant.helpers.area_registry  # noqa: F401
        return
    except ImportError:
        pass
    names = [
        "homeassistant",
        "homeassistant.core",
        "homeassistant.helpers",
        "homeassistant.helpers.area_registry",
        "homeassistant.helpers.entity_registry",
        "homeassistant.helpers.device_registry",
        "homeassistant.helpers.floor_registry",
        "homeassistant.helpers.label_registry",
        "homeassistant.helpers.llm",
        "homeassistant.components",
        "homeassistant.components.homeassistant",
        "homeassistant.components.conversation",
        "homeassistant.components.conversation.chat_log",
    ]
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        module.__getattr__ = lambda attr: MagicMock(name=attr)
        sys.modules[name] = module
    for name in names:
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, sys.modules[name])


_stub_homeassistant()

from mcp_assist_se.discovery import SmartDiscovery  # noqa: E402
from mcp_assist_se.text_match import (  # noqa: E402
    best_typo_match,
    fuzzy_key,
    typo_contains,
    typo_distance,
)


def _names(entry):
    return [entry.name, *sorted(entry.aliases)]


def _area(area_id, name, aliases=()):
    return SimpleNamespace(
        id=area_id, name=name, aliases=set(aliases), floor_id=None, labels=set()
    )


def _floor(floor_id, name, aliases=()):
    return SimpleNamespace(floor_id=floor_id, name=name, aliases=set(aliases))


def _normalized(name):
    """Home Assistant registries compare names casefolded without spaces."""
    return "".join(name.casefold().split())


class AreaRegistry:
    def __init__(self, *areas):
        self._areas = list(areas)

    def async_list_areas(self):
        return list(self._areas)

    def async_get_area_by_name(self, name):
        return next(
            (a for a in self._areas if _normalized(a.name) == _normalized(name)), None
        )

    def async_get_areas_by_alias(self, alias):
        return [
            a for a in self._areas
            if _normalized(alias) in {_normalized(x) for x in a.aliases}
        ]


class FloorRegistry:
    def __init__(self, *floors):
        self._floors = list(floors)

    def async_list_floors(self):
        return list(self._floors)

    def async_get_floor_by_name(self, name):
        return next(
            (f for f in self._floors if _normalized(f.name) == _normalized(name)), None
        )

    def async_get_floors_by_alias(self, alias):
        return [
            f for f in self._floors
            if _normalized(alias) in {_normalized(x) for x in f.aliases}
        ]


class LabelRegistry:
    def __init__(self, *labels):
        self._labels = list(labels)

    def async_list_labels(self):
        return list(self._labels)

    def async_get_label_by_name(self, name):
        return next(
            (lb for lb in self._labels if _normalized(lb.name) == _normalized(name)),
            None,
        )


class TypoDistanceTests(unittest.TestCase):
    def test_doubled_letter_is_free(self):
        self.assertEqual(fuzzy_key("Гостинная"), fuzzy_key("Гостиная"))
        self.assertEqual(typo_distance("Гостинная", "Гостиная"), 0)

    def test_single_typo_within_budget(self):
        self.assertEqual(typo_distance("Спалня", "Спальня"), 1)
        self.assertEqual(typo_distance("Kitchn", "Kitchen"), 1)

    def test_transposition_counts_as_one(self):
        self.assertEqual(typo_distance("Kicthen", "Kitchen"), 1)

    def test_long_names_allow_two(self):
        self.assertEqual(typo_distance("Гостинаая комната", "Гостиная комната"), 0)
        self.assertEqual(typo_distance("Детскаая спльня", "Детская спальня"), 1)
        self.assertEqual(typo_distance("Bathrom upstars", "Bathroom upstairs"), 1)
        self.assertEqual(typo_distance("Bathrum upstars", "Bathroom upstairs"), 2)

    def test_numbers_are_identifiers(self):
        self.assertIsNone(typo_distance("Зал", "Зал 2"))
        self.assertIsNone(typo_distance("Спальня 1", "Спальня 2"))
        self.assertIsNone(typo_distance("Bedroom 1", "Bedroom"))

    def test_short_names_never_fuzzy(self):
        self.assertIsNone(typo_distance("Зал", "Зол"))
        self.assertIsNone(typo_distance("Ваня", "Ванна"))
        self.assertIsNone(typo_distance("Hall", "Hell"))

    def test_different_words_do_not_match(self):
        self.assertIsNone(typo_distance("Bedroom", "Bathroom"))
        self.assertIsNone(typo_distance("Кухня", "Ванная"))
        self.assertIsNone(typo_distance("Office", "Garage"))

    def test_homoglyphs_still_fold(self):
        self.assertEqual(typo_distance("Гостинн" + "a" + "я", "Гостиная"), 0)

    def test_contains_word_span(self):
        self.assertEqual(typo_contains("Люстра в гостиной", "люстрра"), 0)
        self.assertEqual(typo_contains("Ceiling lamp", "celing"), 1)
        self.assertIsNone(typo_contains("Люстра", "лю"))
        self.assertIsNone(typo_contains("Люстра", "люстра в зале"))


class BestMatchTests(unittest.TestCase):
    def test_unique_best_wins(self):
        areas = [_area("a1", "Гостиная"), _area("a2", "Кухня")]
        result = best_typo_match("Гостинная", areas, _names)
        self.assertIs(result.match, areas[0])

    def test_closer_candidate_wins(self):
        areas = [_area("a1", "Kitchen"), _area("a2", "Kitchenette")]
        self.assertIs(best_typo_match("Kitchn", areas, _names).match, areas[0])

    def test_tie_is_ambiguous(self):
        areas = [_area("a1", "Спальня"), _area("a2", "Спальни")]
        result = best_typo_match("Спальн", areas, _names)
        self.assertIsNone(result.match)
        self.assertTrue(result.ambiguous)
        self.assertEqual(result.candidates, ["Спальни", "Спальня"])

    def test_alias_counts(self):
        areas = [_area("a1", "Living room", aliases={"Гостиная"})]
        self.assertIs(best_typo_match("Гостинная", areas, _names).match, areas[0])

    def test_nothing_close(self):
        result = best_typo_match("Garage", [_area("a1", "Kitchen")], _names)
        self.assertIsNone(result.match)
        self.assertFalse(result.ambiguous)


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.discovery = SmartDiscovery(MagicMock())
        self.discovery.begin_resolution_hints()

    def test_area_typo_resolves_with_note(self):
        registry = AreaRegistry(_area("living", "Гостиная"), _area("kitchen", "Кухня"))
        entry = self.discovery._resolve_area_entry("Гостинная", registry)
        self.assertEqual(entry.id, "living")
        self.assertIn("Гостиная", " ".join(self.discovery.resolution_hints()))

    def test_exact_hit_beats_fuzzy(self):
        registry = AreaRegistry(_area("a1", "Kitchen"), _area("a2", "Kitchn"))
        self.assertEqual(self.discovery._resolve_area_entry("Kitchn", registry).id, "a2")
        self.assertEqual(self.discovery.resolution_hints(), [])

    def test_ambiguous_area_not_guessed(self):
        registry = AreaRegistry(_area("s1", "Спальня"), _area("s2", "Спальни"))
        self.assertIsNone(self.discovery._resolve_area_entry("Спальн", registry))
        hint = " ".join(self.discovery.resolution_hints())
        self.assertIn("Спальня", hint)
        self.assertIn("Спальни", hint)

    def test_numbered_rooms_not_confused(self):
        registry = AreaRegistry(_area("h1", "Зал"))
        self.assertIsNone(self.discovery._resolve_area_entry("Зал 2", registry))

    def test_exact_floor_beats_fuzzy_area(self):
        areas = AreaRegistry(_area("a1", "Upstairs hall"))
        floors = FloorRegistry(_floor("f1", "Upstairs hal"))
        area, floor = self.discovery._resolve_area_or_floor("Upstairs hal", areas, floors)
        self.assertIsNone(area)
        self.assertEqual(floor.floor_id, "f1")

    def test_floor_and_label_typos(self):
        floors = FloorRegistry(_floor("f2", "Второй этаж"))
        self.assertEqual(
            self.discovery._resolve_floor_entry("Втарой этаж", floors).floor_id, "f2"
        )
        labels = LabelRegistry(SimpleNamespace(label_id="l1", name="Освещение"))
        self.assertEqual(
            self.discovery._resolve_label_entry("Освещенние", labels).label_id, "l1"
        )

    def test_pure_latin(self):
        registry = AreaRegistry(_area("lr", "Living Room"), _area("br", "Bedroom"))
        self.assertEqual(self.discovery._resolve_area_entry("living rom", registry).id, "lr")
        self.assertIsNone(self.discovery._resolve_area_entry("Bathroom", registry))


class _State(SimpleNamespace):
    @property
    def domain(self):
        return self.entity_id.split(".")[0]


class NameFallbackTests(unittest.IsolatedAsyncioTestCase):
    """discover_entities: exact name search first, typo-tolerant only if empty."""

    def setUp(self):
        import mcp_assist_se.discovery as discovery_module

        self.module = discovery_module
        living = _area("living", "Гостиная")
        self.states = [
            _State(entity_id="light.chandelier", name="Люстра", state="off", attributes={}),
            _State(entity_id="light.lamp", name="Лампа", state="on", attributes={}),
            _State(entity_id="sensor.temp", name="Температура", state="22", attributes={}),
        ]
        areas = {"living": living}
        entries = {
            s.entity_id: SimpleNamespace(
                device_id=None, area_id="living", labels=set(), aliases=set()
            )
            for s in self.states
        }
        area_registry = AreaRegistry(*areas.values())
        area_registry.async_get_area = areas.get
        self.saved = {
            name: getattr(discovery_module, name)
            for name in (
                "ar", "er", "dr", "fr", "lr", "async_should_expose",
                "entity_area_id", "entity_aliases",
            )
        }
        discovery_module.ar = SimpleNamespace(async_get=lambda hass: area_registry)
        discovery_module.er = SimpleNamespace(
            async_get=lambda hass: SimpleNamespace(async_get=entries.get)
        )
        discovery_module.dr = SimpleNamespace(
            async_get=lambda hass: SimpleNamespace(async_get=lambda _id: None)
        )
        # No floors or labels in this home.
        discovery_module.fr = SimpleNamespace(async_get=lambda hass: None)
        discovery_module.lr = SimpleNamespace(async_get=lambda hass: None)
        discovery_module.async_should_expose = lambda hass, assistant, entity_id: True
        discovery_module.entity_area_id = lambda hass, entry: entry.area_id
        discovery_module.entity_aliases = lambda hass, entry: sorted(entry.aliases)
        hass = MagicMock()
        hass.states.async_all = lambda: list(self.states)
        hass.states.get = lambda entity_id: None
        self.discovery = SmartDiscovery(hass)
        self.discovery.begin_resolution_hints()

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(self.module, name, value)

    async def _ids(self, **kwargs):
        found = await self.discovery._discover_general_entities(
            entity_type=None, area=None, floor=None, label=None, domain=None,
            state=None, limit=20, **kwargs
        )
        return [e["entity_id"] for e in found]

    async def test_exact_substring_unchanged(self):
        self.assertEqual(await self._ids(name_contains="люстр"), ["light.chandelier"])
        self.assertEqual(self.discovery.resolution_hints(), [])

    async def test_typo_in_name(self):
        self.assertEqual(await self._ids(name_contains="люсстра"), ["light.chandelier"])
        self.assertEqual(await self._ids(name_contains="Лустра"), ["light.chandelier"])
        self.assertTrue(self.discovery.resolution_hints())

    async def test_typo_in_area_with_domain(self):
        found = await self.discovery._discover_general_entities(
            entity_type=None, area="Гостинная", floor=None, label=None,
            domain="light", state=None, name_contains=None, limit=20,
        )
        self.assertEqual(
            sorted(e["entity_id"] for e in found), ["light.chandelier", "light.lamp"]
        )

    async def test_no_false_fuzzy_name(self):
        self.assertEqual(await self._ids(name_contains="Камера"), [])
        self.assertEqual(await self._ids(name_contains="Лампочка"), [])


if __name__ == "__main__":
    unittest.main()
