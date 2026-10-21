"""A turn with final text and only bookkeeping tools needs no extra LLM round."""

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

from mcp_assist_se.round_trips import (
    NON_INFORMATIONAL_TOOLS,
    can_finish_without_followup,
    tool_call_names,
)


def call(name):
    return {"id": f"id_{name}", "type": "function", "function": {"name": name, "arguments": "{}"}}


class RoundTripTests(unittest.TestCase):
    def test_text_with_only_bookkeeping_tool_finishes(self):
        self.assertTrue(
            can_finish_without_followup("Готово. Что ещё?", [call("set_conversation_state")])
        )

    def test_empty_text_needs_followup(self):
        self.assertFalse(can_finish_without_followup("", [call("set_conversation_state")]))
        self.assertFalse(can_finish_without_followup("   ", [call("set_conversation_state")]))
        self.assertFalse(can_finish_without_followup(None, [call("set_conversation_state")]))

    def test_informational_tool_needs_followup(self):
        self.assertFalse(
            can_finish_without_followup(
                "Сейчас проверю", [call("set_conversation_state"), call("discover_entities")]
            )
        )
        self.assertFalse(can_finish_without_followup("Включаю", [call("perform_action")]))

    def test_no_tool_calls_is_not_this_path(self):
        self.assertFalse(can_finish_without_followup("Ответ", []))
        self.assertFalse(can_finish_without_followup("Ответ", None))

    def test_custom_set(self):
        self.assertTrue(
            can_finish_without_followup("ok", [call("x")], frozenset({"x"}))
        )
        self.assertIn("set_conversation_state", NON_INFORMATIONAL_TOOLS)

    def test_names(self):
        self.assertEqual(tool_call_names([call("a"), {"function": {}}]), ["a", ""])


if __name__ == "__main__":
    unittest.main()
