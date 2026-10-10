"""Server-side remember decisions. mode=add must not create a duplicate."""

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

from mcp_assist_se.memory_rank import (
    format_similarity_decision,
    format_store_decision_prompt,
    memory_compare_key,
    parse_llm_store_decision,
    plan_store,
)
from mcp_assist_se.store_decision import openai_chat_url

NAMES = [
    {"id": "abc123def456", "text": "Меня зовут Олег", "category": None},
    {"id": "def456abc123", "text": "Пользователя зовут Олег.", "category": None},
]
NAME_SCORES = {"abc123def456": 0.92, "def456abc123": 0.92}


def _plan(**overrides):
    values = {
        "mode": None,
        "new_text": "Меня зовут Олег",
        "items": NAMES,
        "similarities": NAME_SCORES,
        "requested_id": None,
        "candidate_threshold": 0.65,
        "near_copy_threshold": 0.91,
        "candidates_shown": False,
        "failed_update": False,
        "llm": None,
    }
    values.update(overrides)
    return plan_store(**values)


class StoreDecisionTests(unittest.TestCase):
    def test_normalized_text_matches_case_and_punctuation(self) -> None:
        self.assertEqual(memory_compare_key("Меня зовут Олег."), "меня зовут олег")
        self.assertEqual(memory_compare_key("  МЕНЯ   зовут олег! "), "меня зовут олег")
        self.assertEqual(memory_compare_key("ёлка"), memory_compare_key("елка"))

    def test_exact_text_is_noop_for_every_mode(self) -> None:
        for mode in (None, "add", "update", "noop"):
            plan = _plan(mode=mode, requested_id="user_name")
            self.assertEqual(plan["action"], "noop", mode)
            self.assertEqual(plan["reason"], "exact")
            self.assertEqual(plan["target_id"], "abc123def456")

    def test_live_add_after_bad_update_does_not_store_a_third_name(self) -> None:
        failed = _plan(mode="update", requested_id="user_name")
        self.assertEqual(failed["action"], "noop")
        added = _plan(
            mode="add",
            new_text="Меня зовут Олег!",
            candidates_shown=True,
            failed_update=True,
        )
        self.assertEqual(added["action"], "noop")
        self.assertEqual(added["reason"], "exact")

    def test_near_copy_is_noop_without_a_model(self) -> None:
        plan = _plan(
            mode="add",
            new_text="Имя пользователя — Олег",
            similarities={"abc123def456": 0.92, "def456abc123": 0.88},
        )
        self.assertEqual(plan["action"], "noop")
        self.assertEqual(plan["reason"], "near_copy")
        self.assertEqual(plan["target_id"], "abc123def456")

    def test_different_fact_is_not_auto_updated(self) -> None:
        plan = _plan(
            mode=None,
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90, "def456abc123": 0.89},
        )
        self.assertEqual(plan["action"], "candidates")
        self.assertEqual(
            [item["id"] for item in plan["candidates"]],
            ["abc123def456", "def456abc123"],
        )
        listed = format_similarity_decision("memory", plan["target"] or "", plan["candidates"])
        self.assertIn("abc123def456", listed)
        self.assertIn("not an id", listed)

    def test_mode_add_without_a_candidate_round_does_not_store(self) -> None:
        plan = _plan(
            mode="add",
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90},
            llm={"mode": "add", "id": None},
        )
        self.assertEqual(plan["action"], "candidates")
        self.assertNotEqual(plan["reason"], "llm")

    def test_mode_add_after_candidates_stores_a_different_fact(self) -> None:
        plan = _plan(
            mode="add",
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90},
            candidates_shown=True,
        )
        self.assertEqual(plan["action"], "add")
        self.assertEqual(plan["reason"], "confirmed_add")

    def test_mode_add_after_a_failed_update_returns_real_ids(self) -> None:
        plan = _plan(
            mode="add",
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90},
            candidates_shown=True,
            failed_update=True,
            llm={"mode": "add", "id": None},
        )
        self.assertEqual(plan["action"], "candidates")
        self.assertEqual(plan["candidates"][0]["id"], "abc123def456")

    def test_unknown_id_returns_candidates(self) -> None:
        plan = _plan(
            mode="update",
            new_text="Олега зовут хозяином",
            requested_id="user_name",
            similarities={"abc123def456": 0.90},
        )
        self.assertEqual(plan["action"], "candidates")
        self.assertEqual(plan["reason"], "unknown_id")
        self.assertIn("abc123def456", {item["id"] for item in plan["candidates"]})

    def test_update_by_category_when_exactly_one_candidate(self) -> None:
        items = [
            {"id": "abc123def456", "text": "Меня зовут Олег", "category": "user_name"},
            {"id": "light1", "text": "Яркий белый свет", "category": "lighting"},
        ]
        plan = _plan(
            mode="update",
            new_text="Меня зовут Олег Петрович",
            items=items,
            similarities={"abc123def456": 0.80, "light1": 0.20},
            requested_id="user_name",
        )
        self.assertEqual(plan["action"], "update")
        self.assertEqual(plan["reason"], "category")
        self.assertEqual(plan["target_id"], "abc123def456")

    def test_category_with_two_candidates_is_not_an_update(self) -> None:
        items = [
            {"id": "one", "text": "Меня зовут Олег", "category": "user_name"},
            {"id": "two", "text": "Пользователя зовут Олег", "category": "user_name"},
        ]
        plan = _plan(
            mode="update",
            new_text="Моё имя Олег",
            items=items,
            similarities={"one": 0.80, "two": 0.79},
            requested_id="user_name",
        )
        self.assertEqual(plan["action"], "candidates")
        self.assertNotEqual(plan["reason"], "category")

    def test_real_id_updates_that_row(self) -> None:
        plan = _plan(
            mode="update",
            new_text="Меня зовут Олег Петрович",
            requested_id="abc123def456",
            similarities={"abc123def456": 0.80},
        )
        self.assertEqual(plan["action"], "update")
        self.assertEqual(plan["reason"], "id")
        self.assertEqual(plan["target_id"], "abc123def456")

    def test_internal_decision_is_one_call(self) -> None:
        ambiguous = _plan(
            mode=None,
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90},
            llm={"mode": "add", "id": None},
        )
        self.assertEqual(ambiguous["action"], "add")
        self.assertEqual(ambiguous["reason"], "llm")
        merged = _plan(
            mode=None,
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90},
            llm={"mode": "update", "id": "abc123def456"},
        )
        self.assertEqual(merged["action"], "update")
        self.assertEqual(merged["target_id"], "abc123def456")
        known = _plan(
            mode=None,
            new_text="Олега зовут хозяином",
            similarities={"abc123def456": 0.90},
            llm={"mode": "noop", "id": None},
        )
        self.assertEqual(known["action"], "noop")
        self.assertEqual(known["reason"], "llm")

    def test_parser_rejects_invented_ids(self) -> None:
        allowed = {"abc123def456"}
        self.assertEqual(
            parse_llm_store_decision('{"mode":"update","id":"user_name"}', allowed),
            None,
        )
        self.assertEqual(
            parse_llm_store_decision(
                '```json\n{"mode":"update","id":"abc123def456"}\n```', allowed
            ),
            {"mode": "update", "id": "abc123def456"},
        )
        self.assertEqual(
            parse_llm_store_decision('{"mode":"noop"}', allowed),
            {"mode": "noop", "id": None},
        )
        self.assertIsNone(parse_llm_store_decision("store it", allowed))
        prompt = format_store_decision_prompt(
            "memory", "Олега зовут хозяином", NAMES
        )
        self.assertIn("abc123def456", prompt)
        self.assertIn("хозяином", prompt)
        self.assertLess(len(prompt), 800)

    def test_new_fact_with_no_candidates_is_added(self) -> None:
        plan = _plan(
            mode="add",
            new_text="Дверь спальни закрыта.",
            similarities={"abc123def456": 0.40, "def456abc123": 0.30},
        )
        self.assertEqual(plan["action"], "add")
        self.assertEqual(plan["reason"], "new")

    def test_chat_url_does_not_double_v1(self) -> None:
        self.assertEqual(
            openai_chat_url("https://openai.bothub.chat"),
            "https://openai.bothub.chat/v1/chat/completions",
        )
        self.assertEqual(
            openai_chat_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
