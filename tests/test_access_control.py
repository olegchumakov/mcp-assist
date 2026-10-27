"""Speaker tag, permission resolution, and enforcement. Time is frozen."""

from __future__ import annotations

import hashlib
import hmac
import sys
import types
import unittest
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_components" / "mcp_assist_se"
sys.path.insert(0, str(ROOT / "custom_components"))

if "mcp_assist_se" not in sys.modules:
    package = types.ModuleType("mcp_assist_se")
    package.__path__ = [str(PACKAGE)]
    package.__package__ = "mcp_assist_se"
    sys.modules["mcp_assist_se"] = package

from mcp_assist_se.access_control import (
    LEVEL_ALL,
    LEVEL_NONE,
    LEVEL_SELECTED,
    REFUSAL_TEXT,
    AccessConfig,
    DeviceClass,
    EntityRef,
    Grant,
    HomeSnapshot,
    PersonRule,
    authorize_action,
    decide_turn,
    entity_allowed,
    external_tool_allowed,
    filter_discovery_rows,
    filter_index,
    grant_for_person,
    load_access_config,
    parse_people_payload,
    people_catalog,
    people_endpoint,
    people_from_response,
    should_fetch_people,
    split_names,
)
from mcp_assist_se.speaker_tag import (
    HMAC_CYRILLIC_VECTOR,
    HMAC_TEST_VECTOR,
    HMAC_UNKNOWN_VECTOR,
    VECTOR_NOW,
    NonceCache,
    verify_speaker_tag,
)


def _signed(name: str, transcript: str = "включи свет") -> str:
    encoded = quote(name, safe="")
    confidence, created, nonce = "0.90", "1700000000", "abcdef0123456789"
    message = f"spk1|{encoded}|{confidence}|{created}|{nonce}|{transcript}"
    signature = hmac.new(b"test-secret", message.encode(), hashlib.sha256).hexdigest()
    return (
        f"⟦spk1;n={encoded};c={confidence};t={created};r={nonce};s={signature}⟧ {transcript}"
    )


def _tag(vector: dict, sig: str | None = None) -> str:
    signature = vector["sig"] if sig is None else sig
    return (
        f"⟦spk1;n={vector['n']};c={vector['c']};t={vector['t']};"
        f"r={vector['r']};s={signature}⟧ {vector['transcript']}"
    )


def _entity(entity_id: str, **kwargs) -> EntityRef:
    return EntityRef(entity_id=entity_id, domain=entity_id.split(".", 1)[0], **kwargs)


def _home() -> HomeSnapshot:
    return HomeSnapshot(
        entities={
            "light.kitchen": _entity(
                "light.kitchen",
                area_id="kitchen",
                area_name="Kitchen",
                name="Kitchen light",
                device_id="dev-kitchen",
                label_ids=frozenset({"lamps"}),
                label_names=frozenset({"Lamps"}),
            ),
            "light.bed": _entity(
                "light.bed",
                area_id="bed",
                area_name="Bedroom",
                name="Bed light",
            ),
            "sensor.kitchen": _entity(
                "sensor.kitchen",
                area_id="kitchen",
                area_name="Kitchen",
                name="Kitchen temperature",
            ),
            "lock.front": _entity(
                "lock.front",
                area_id="hall",
                area_name="Hall",
                name="Front lock",
                device_id="dev-lock",
            ),
            "light.group": _entity(
                "light.group",
                name="Light group",
                member_ids=("light.kitchen", "lock.front"),
            ),
            "script.cam": _entity("script.cam", name="Camera script"),
            "automation.night": _entity("automation.night", name="Night"),
            "scene.movie": _entity("scene.movie", name="Movie"),
        },
        area_floors={"kitchen": "down", "bed": "up", "hall": "down"},
    )


def _config(**kwargs) -> AccessConfig:
    classes = {
        "lights": DeviceClass(
            id="lights",
            name="Lights",
            entity_ids=frozenset({"light.kitchen", "light.bed"}),
        ),
        "kitchen": DeviceClass(id="kitchen", name="Kitchen", area_ids=frozenset({"kitchen"})),
        "lockdev": DeviceClass(id="lockdev", name="Lock device", device_ids=frozenset({"dev-lock"})),
        "lamps": DeviceClass(id="lamps", name="Lamps", label_ids=frozenset({"lamps"})),
        "downstairs": DeviceClass(
            id="downstairs", name="Downstairs", area_ids=frozenset({"kitchen", "hall"})
        ),
        "scripts": DeviceClass(id="scripts", name="Scripts", entity_ids=frozenset({"script.cam"})),
        "movie": DeviceClass(id="movie", name="Movie", entity_ids=frozenset({"scene.movie"})),
        "night": DeviceClass(
            id="night", name="Night", entity_ids=frozenset({"automation.night"})
        ),
        "group": DeviceClass(id="group", name="Group", entity_ids=frozenset({"light.group"})),
    }
    base = dict(
        enabled=True,
        classes=classes,
        rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"lights"}))},
        unknown_rule=PersonRule(LEVEL_NONE),
        manual_people=("Anna",),
    )
    base.update(kwargs)
    return AccessConfig(**base)


class SpeakerTagTests(unittest.TestCase):
    def test_canonical_vector_at_frozen_time(self) -> None:
        verdict = verify_speaker_tag(
            _tag(HMAC_TEST_VECTOR), "test-secret", VECTOR_NOW, NonceCache()
        )
        self.assertEqual(verdict.status, "valid")
        self.assertEqual(verdict.person, "Anna")
        self.assertEqual(verdict.clean_text, "turn on the light")
        self.assertNotIn("spk1", verdict.clean_text)
        self.assertNotIn("Anna", verdict.clean_text)
        self.assertAlmostEqual(verdict.confidence, 0.73)

    def test_cyrillic_name_uses_literal_encoding(self) -> None:
        self.assertEqual(quote("Анна", safe=""), HMAC_CYRILLIC_VECTOR["n"])
        verdict = verify_speaker_tag(
            _tag(HMAC_CYRILLIC_VECTOR), "test-secret", VECTOR_NOW, NonceCache()
        )
        self.assertEqual(verdict.status, "valid")
        self.assertEqual(verdict.person, "Анна")
        self.assertEqual(verdict.clean_text, "включи свет")
        self.assertNotIn("%D0%90", verdict.clean_text)

    def test_unknown_voice_is_empty_name(self) -> None:
        verdict = verify_speaker_tag(
            _tag(HMAC_UNKNOWN_VECTOR), "test-secret", VECTOR_NOW, NonceCache()
        )
        self.assertEqual(verdict.status, "valid")
        self.assertEqual(verdict.person, "")
        self.assertEqual(verdict.clean_text, "hello")

    def test_bad_signature(self) -> None:
        bad = HMAC_TEST_VECTOR["sig"][:-1] + ("0" if HMAC_TEST_VECTOR["sig"][-1] != "0" else "1")
        verdict = verify_speaker_tag(_tag(HMAC_TEST_VECTOR, bad), "test-secret", VECTOR_NOW, NonceCache())
        self.assertEqual(verdict.status, "invalid")
        self.assertEqual(verdict.reason, "bad_sig")
        self.assertEqual(verdict.clean_text, "turn on the light")

    def test_stale_and_fresh_boundary_use_frozen_clock(self) -> None:
        text = _tag(HMAC_TEST_VECTOR)
        self.assertEqual(
            verify_speaker_tag(text, "test-secret", VECTOR_NOW + 120, NonceCache()).status,
            "valid",
        )
        stale = verify_speaker_tag(text, "test-secret", VECTOR_NOW + 121, NonceCache())
        self.assertEqual(stale.reason, "stale")
        self.assertEqual(
            verify_speaker_tag(text, "test-secret", VECTOR_NOW - 121, NonceCache()).reason,
            "stale",
        )

    def test_stale_tag_does_not_consume_the_nonce(self) -> None:
        text = _tag(HMAC_TEST_VECTOR)
        cache = NonceCache()
        self.assertEqual(
            verify_speaker_tag(text, "test-secret", VECTOR_NOW + 121, cache).reason,
            "stale",
        )
        self.assertEqual(len(cache), 0)
        self.assertEqual(
            verify_speaker_tag(text, "test-secret", VECTOR_NOW, cache).status,
            "valid",
        )

    def test_replay_within_ten_minutes(self) -> None:
        text = _tag(HMAC_TEST_VECTOR)
        cache = NonceCache()
        self.assertEqual(verify_speaker_tag(text, "test-secret", VECTOR_NOW, cache).status, "valid")
        again = verify_speaker_tag(text, "test-secret", VECTOR_NOW + 10, cache)
        self.assertEqual(again.reason, "replay")
        self.assertEqual(again.clean_text, "turn on the light")

    def test_missing_tag(self) -> None:
        verdict = verify_speaker_tag("turn on the light", "test-secret", VECTOR_NOW, NonceCache())
        self.assertEqual(verdict.status, "absent")
        self.assertEqual(verdict.clean_text, "turn on the light")

    def test_other_markers_are_stripped(self) -> None:
        verdict = verify_speaker_tag(
            "hello ⟦note⟧ there", "test-secret", VECTOR_NOW, NonceCache()
        )
        self.assertEqual(verdict.status, "absent")
        self.assertEqual(verdict.clean_text, "hello there")
        self.assertNotIn("⟦", verdict.clean_text)


class PermissionTests(unittest.TestCase):
    def test_selected_matches_entity_area_device_and_label(self) -> None:
        config = _config()
        home = _home()
        anna = grant_for_person(config, "Anna")
        self.assertTrue(entity_allowed(home.entities["light.kitchen"], anna, config))
        self.assertFalse(entity_allowed(home.entities["lock.front"], anna, config))
        kitchen = grant_for_person(
            _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"kitchen"}))}),
            "Anna",
        )
        self.assertTrue(entity_allowed(home.entities["sensor.kitchen"], kitchen, config))
        self.assertFalse(entity_allowed(home.entities["light.bed"], kitchen, config))
        by_device = grant_for_person(
            _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"lockdev"}))}),
            "Anna",
        )
        self.assertTrue(entity_allowed(home.entities["lock.front"], by_device, config))
        by_label = grant_for_person(
            _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"lamps"}))}),
            "Anna",
        )
        self.assertTrue(entity_allowed(home.entities["light.kitchen"], by_label, config))

    def test_missing_person_is_all_and_empty_selection_is_none(self) -> None:
        config = _config()
        self.assertEqual(grant_for_person(config, "Nobody").level, LEVEL_ALL)
        self.assertTrue(_grant_blocks(grant_for_person(config, "")))
        empty = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset())})
        self.assertTrue(_grant_blocks(grant_for_person(empty, "Anna")))

    def test_follow_up_intersection_does_not_widen(self) -> None:
        selected = Grant(LEVEL_SELECTED, frozenset({"lights"}))
        wider = selected.intersect(Grant(LEVEL_ALL))
        self.assertEqual(wider.level, LEVEL_SELECTED)
        tighter = Grant(LEVEL_ALL).intersect(Grant(LEVEL_NONE))
        self.assertEqual(tighter.level, LEVEL_NONE)


def _grant_blocks(grant: Grant) -> bool:
    return grant.level == LEVEL_NONE or any(part.level == LEVEL_NONE for part in grant.all_of)


class EnforcementTests(unittest.TestCase):
    def _anna(self):
        config = _config()
        return config, grant_for_person(config, "Anna"), _home()

    def test_allowed_entity_is_rewritten_to_that_id(self) -> None:
        config, grant, home = self._anna()
        decision = authorize_action(
            "light", "turn_on", {"entity_id": "light.kitchen"}, {}, grant, config, home
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.passthrough)
        self.assertEqual(decision.entity_ids, ("light.kitchen",))

    def test_forbidden_entity_is_refused(self) -> None:
        config, grant, home = self._anna()
        decision = authorize_action(
            "lock", "lock", {"entity_id": "lock.front"}, {}, grant, config, home
        )
        self.assertFalse(decision.allowed)
        self.assertIn("NOT ALLOWED", decision.refusal)
        self.assertNotIn("lock.front", decision.refusal)

    def test_area_light_action_ignores_sensors_and_drops_the_area(self) -> None:
        config, grant, home = self._anna()
        decision = authorize_action(
            "light", "turn_on", {"area_id": "kitchen"}, {}, grant, config, home
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.entity_ids, ("light.kitchen",))

    def test_homeassistant_area_action_is_refused_when_any_entity_is_forbidden(self) -> None:
        config, grant, home = self._anna()
        decision = authorize_action(
            "homeassistant", "turn_off", {"area_id": "kitchen"}, {}, grant, config, home
        )
        self.assertFalse(decision.allowed)

    def test_device_label_and_floor_targets(self) -> None:
        home = _home()
        config = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"lockdev"}))})
        grant = grant_for_person(config, "Anna")
        device = authorize_action(
            "lock", "lock", {"device_id": "dev-lock"}, {}, grant, config, home
        )
        self.assertTrue(device.allowed)
        self.assertEqual(device.entity_ids, ("lock.front",))

        config = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"lamps"}))})
        grant = grant_for_person(config, "Anna")
        label = authorize_action(
            "light", "turn_on", {"label_id": "lamps"}, {}, grant, config, home
        )
        self.assertEqual(label.entity_ids, ("light.kitchen",))

        config = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"lights"}))})
        grant = grant_for_person(config, "Anna")
        floor = authorize_action(
            "light", "turn_on", {"floor_id": "down"}, {}, grant, config, home
        )
        self.assertEqual(floor.entity_ids, ("light.kitchen",))

    def test_template_all_and_empty_target_are_refused(self) -> None:
        config, grant, home = self._anna()
        template = authorize_action(
            "light",
            "turn_on",
            {"entity_id": "light.kitchen"},
            {"brightness": "{{ 100 }}"},
            grant,
            config,
            home,
        )
        self.assertFalse(template.allowed)
        everything = authorize_action(
            "light", "turn_off", {"entity_id": "all"}, {}, grant, config, home
        )
        self.assertFalse(everything.allowed)
        empty = authorize_action("light", "turn_on", {}, {}, grant, config, home)
        self.assertFalse(empty.allowed)

    def test_group_with_a_forbidden_member_is_refused(self) -> None:
        config = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"group", "lights"}))})
        grant = grant_for_person(config, "Anna")
        decision = authorize_action(
            "light", "turn_on", {"entity_id": "light.group"}, {}, grant, config, _home()
        )
        self.assertFalse(decision.allowed)

    def test_script_scene_and_automation(self) -> None:
        home = _home()
        allowed = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"scripts"}))})
        grant = grant_for_person(allowed, "Anna")
        script = authorize_action(
            "script", "cam", {}, {}, grant, allowed, home, broad_data=True
        )
        self.assertTrue(script.allowed)
        denied = authorize_action(
            "script", "cam", {}, {}, grant_for_person(_config(), "Anna"), _config(), home, broad_data=True
        )
        self.assertFalse(denied.allowed)

        broad = authorize_action(
            "script",
            "cam",
            {"entity_id": "script.cam"},
            {"area_id": "kitchen"},
            grant,
            allowed,
            home,
            broad_data=True,
        )
        self.assertFalse(broad.allowed)

        scene_config = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"movie"}))})
        scene = authorize_action(
            "scene",
            "turn_on",
            {"entity_id": "scene.movie"},
            {},
            grant_for_person(scene_config, "Anna"),
            scene_config,
            home,
        )
        self.assertTrue(scene.allowed)
        night = _config(rules={"Anna": PersonRule(LEVEL_SELECTED, frozenset({"night"}))})
        automation = authorize_action(
            "automation",
            "trigger",
            {"entity_id": "automation.night"},
            {},
            grant_for_person(night, "Anna"),
            night,
            home,
            broad_data=True,
        )
        self.assertTrue(automation.allowed)
        self.assertFalse(
            authorize_action(
                "automation",
                "trigger",
                {"entity_id": "automation.night"},
                {},
                grant_for_person(_config(), "Anna"),
                _config(),
                home,
                broad_data=True,
            ).allowed
        )

    def test_all_level_passes_the_original_target_through(self) -> None:
        config = _config(unknown_rule=PersonRule(LEVEL_ALL))
        decision = authorize_action(
            "lock",
            "lock",
            {"area_id": "hall"},
            {"message": "{{ now() }}"},
            grant_for_person(config, ""),
            config,
            _home(),
        )
        self.assertTrue(decision.passthrough)

    def test_reads_hide_forbidden_entities(self) -> None:
        config, grant, home = self._anna()
        rows = filter_discovery_rows(
            [
                {"entity_id": "_summary", "total_found": 2},
                {"entity_id": "light.kitchen", "name": "Kitchen light"},
                {"entity_id": "lock.front", "name": "Front lock"},
            ],
            grant,
            config,
            home,
        )
        self.assertEqual([row["entity_id"] for row in rows], ["_summary", "light.kitchen"])
        self.assertEqual(rows[0]["total_found"], 1)
        index = filter_index(
            {
                "areas": [{"name": "Kitchen"}, {"name": "Hall"}],
                "scripts": ["Camera script", "Other"],
                "automations": ["Night"],
            },
            grant,
            config,
            home,
        )
        self.assertEqual([area["name"] for area in index["areas"]], ["Kitchen"])
        self.assertEqual(index["scripts"], [])
        self.assertEqual(index["automations"], [])

    def test_external_custom_tools_require_all(self) -> None:
        self.assertFalse(external_tool_allowed(Grant(LEVEL_SELECTED, frozenset({"lights"})), True))
        self.assertTrue(external_tool_allowed(Grant(LEVEL_ALL), True))
        self.assertTrue(external_tool_allowed(None, True))
        self.assertEqual(REFUSAL_TEXT.splitlines()[0], "NOT ALLOWED")


class ModeTests(unittest.TestCase):
    def test_switch_off_does_not_parse_or_strip(self) -> None:
        config = AccessConfig(enabled=False, secret="test-secret", addon_url="http://local-wyoming-speaker-id:8099")
        cache = NonceCache()
        text = _tag(HMAC_TEST_VECTOR)
        decision, stored = decide_turn(
            config,
            text,
            user_id=None,
            previous=Grant(LEVEL_NONE),
            now=VECTOR_NOW,
            nonces=cache,
        )
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.text, text)
        self.assertIsNone(decision.grant)
        self.assertIsNone(stored)
        self.assertFalse(decision.blocked)
        self.assertEqual(len(cache), 0)
        self.assertFalse(should_fetch_people(config.enabled, config.addon_url))

    def test_switch_on_without_addon_uses_typed_names_and_user_mapping(self) -> None:
        config = _config(
            secret="",
            addon_url="",
            manual_people=("Anna",),
            cached_people=(),
            user_map={"user-1": "Anna"},
            unknown_rule=PersonRule(LEVEL_NONE),
        )
        self.assertFalse(should_fetch_people(config.enabled, config.addon_url))
        self.assertEqual(
            people_endpoint("http://local-wyoming-speaker-id:8099"),
            "http://local-wyoming-speaker-id:8099/api/people",
        )
        self.assertIsNone(people_from_response(401, {"people": [{"name": "Anna"}]}))
        self.assertEqual(people_from_response(200, {"people": [{"name": "Anna"}]}), ["Anna"])

        typed, _stored = decide_turn(
            config,
            "turn on the light",
            user_id="user-1",
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertEqual(typed.person, "Anna")
        self.assertEqual(typed.text, "turn on the light")
        self.assertFalse(typed.blocked)
        self.assertEqual(typed.grant.level, LEVEL_SELECTED)

        unmapped, _stored = decide_turn(
            config,
            "turn on the light",
            user_id="user-2",
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertTrue(unmapped.blocked)

        voice, _stored = decide_turn(
            config,
            "turn on the light",
            user_id=None,
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertTrue(voice.blocked)

        stripped, _stored = decide_turn(
            config,
            _tag(HMAC_TEST_VECTOR),
            user_id=None,
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertEqual(stripped.text, "turn on the light")
        self.assertNotIn("Anna", stripped.text)
        self.assertTrue(stripped.blocked)

    def test_follow_up_without_a_tag_cannot_escalate(self) -> None:
        config = _config(secret="test-secret", unknown_rule=PersonRule(LEVEL_ALL))
        cache = NonceCache()
        first, stored = decide_turn(
            config,
            _tag(HMAC_TEST_VECTOR),
            user_id=None,
            previous=None,
            now=VECTOR_NOW,
            nonces=cache,
        )
        self.assertEqual(first.person, "Anna")
        self.assertEqual(first.grant.level, LEVEL_SELECTED)
        second, stored = decide_turn(
            config,
            "turn it off",
            user_id=None,
            previous=stored,
            now=VECTOR_NOW,
            nonces=cache,
        )
        self.assertEqual(second.text, "turn it off")
        self.assertEqual(second.grant.level, LEVEL_SELECTED)
        self.assertEqual(second.grant.class_ids, frozenset({"lights"}))

    def test_low_confidence_becomes_unknown(self) -> None:
        config = _config(secret="test-secret", min_confidence=0.9, unknown_rule=PersonRule(LEVEL_NONE))
        decision, _stored = decide_turn(
            config,
            _tag(HMAC_TEST_VECTOR),
            user_id=None,
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertEqual(decision.person, "")
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.text, "turn on the light")


class PersonNameTests(unittest.TestCase):
    def _oleg(self, **kwargs) -> AccessConfig:
        values = dict(
            enabled=True,
            secret="test-secret",
            rules={"Олег": PersonRule(LEVEL_SELECTED, frozenset({"lights"}))},
            unknown_rule=PersonRule(LEVEL_NONE),
            classes=_config().classes,
        )
        values.update(kwargs)
        return AccessConfig(**values)

    def test_oleg_matches_regardless_of_capitals_and_spaces(self) -> None:
        config = self._oleg()
        self.assertEqual(grant_for_person(config, "олег").level, LEVEL_SELECTED)
        self.assertEqual(grant_for_person(config, "ОЛЕГ").level, LEVEL_SELECTED)
        self.assertEqual(grant_for_person(config, "  Олег  ").level, LEVEL_SELECTED)
        self.assertEqual(grant_for_person(config, "олег   петров").level, LEVEL_ALL)
        spaced = self._oleg(rules={"Олег Петров": PersonRule(LEVEL_NONE)})
        self.assertTrue(_grant_blocks(grant_for_person(spaced, "олег   петров")))
        self.assertEqual(grant_for_person(config, "Анна").level, LEVEL_ALL)

    def test_signed_lowercase_name_uses_the_saved_rule(self) -> None:
        decision, _stored = decide_turn(
            self._oleg(),
            _signed("олег"),
            user_id=None,
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertEqual(decision.person, "олег")
        self.assertEqual(decision.grant.level, LEVEL_SELECTED)
        self.assertFalse(decision.blocked)

    def test_typed_user_mapping_ignores_capitals(self) -> None:
        config = AccessConfig(
            enabled=True,
            rules={"Олег": PersonRule(LEVEL_NONE)},
            unknown_rule=PersonRule(LEVEL_ALL),
            user_map={"user-1": "олег"},
        )
        decision, _stored = decide_turn(
            config,
            "включи свет",
            user_id="user-1",
            previous=None,
            now=VECTOR_NOW,
            nonces=NonceCache(),
        )
        self.assertEqual(decision.person, "олег")
        self.assertTrue(decision.blocked)

    def test_people_lists_collapse_the_same_name(self) -> None:
        self.assertEqual(
            parse_people_payload(
                {"people": [{"name": "Олег"}, {"name": "олег"}, {"name": "  ОЛЕГ "}]}
            ),
            ["Олег"],
        )
        self.assertEqual(split_names("Олег, олег\nАнна"), ["Олег", "Анна"])
        self.assertEqual(
            people_catalog(
                AccessConfig(
                    cached_people=("Олег",),
                    manual_people=("олег",),
                    rules={"ОЛЕГ": PersonRule(LEVEL_NONE)},
                )
            ),
            ["Олег"],
        )

    def test_saved_rules_keep_the_first_spelling(self) -> None:
        loaded = load_access_config(
            {
                "access_control_enabled": True,
                "access_rules": {
                    "Олег": {"level": "none", "classes": []},
                    "олег": {"level": "all", "classes": []},
                },
            }
        )
        self.assertEqual(list(loaded.rules), ["Олег"])
        self.assertEqual(grant_for_person(loaded, "олег").level, LEVEL_NONE)


if __name__ == "__main__":
    unittest.main()
