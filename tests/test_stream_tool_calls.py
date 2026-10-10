"""Streamed tool calls must all execute, one result per tool_call_id.

Gemini via an OpenAI-compatible stream (openai.bothub.chat) can emit every
parallel call with index 0 and no id, or with one repeated id. Keying only on
that index kept forget_memory and dropped remove_personality_instruction.
"""

from __future__ import annotations

import json
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

from mcp_assist_se.stream_tool_calls import (
    finalize_streamed_tool_calls,
    merge_streamed_tool_fragment,
)


def _accumulate(fragments: list[dict]) -> list[dict]:
    tool_calls: list[dict] = []
    for fragment in fragments:
        merge_streamed_tool_fragment(tool_calls, fragment)
    return finalize_streamed_tool_calls(tool_calls)


def _result_messages(tool_calls: list[dict]) -> list[dict]:
    """The OpenAI tool messages the agent returns after each call."""
    return [
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": tool_call["function"]["name"],
        }
        for tool_call in tool_calls
    ]


class StreamedToolCallTests(unittest.TestCase):
    def test_same_index_and_missing_ids_keep_both_calls(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "remove_personality_instruction",
                        "arguments": '{"instruction_id":"p1"}',
                    },
                },
                {
                    "index": 0,
                    "function": {
                        "name": "forget_memory",
                        "arguments": '{"query":"пиратов"}',
                    },
                },
            ]
        )
        self._assert_both_executed(finished)

    def test_both_calls_in_one_delta_are_kept(self) -> None:
        delta = [
            {
                "index": 0,
                "function": {
                    "name": "remove_personality_instruction",
                    "arguments": {"instruction_id": "p1"},
                },
            },
            {
                "index": 0,
                "function": {
                    "name": "forget_memory",
                    "arguments": {"query": "пиратов"},
                },
            },
        ]
        finished = _accumulate(delta)
        self._assert_both_executed(finished)

    def test_argument_deltas_after_a_reused_index_stay_on_their_call(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "remove_personality_instruction",
                        "arguments": "",
                    },
                },
                {"index": 0, "function": {"arguments": '{"instruction_id":'}},
                {"index": 0, "function": {"arguments": '"p1"}'}},
                {
                    "index": 0,
                    "function": {"name": "forget_memory", "arguments": ""},
                },
                {"index": 0, "function": {"arguments": '{"query":'}},
                {"index": 0, "function": {"arguments": '"пиратов"}'}},
            ]
        )
        self._assert_both_executed(finished)

    def test_repeated_id_does_not_collapse_two_names(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "id": "call_same",
                    "function": {
                        "name": "remove_personality_instruction",
                        "arguments": '{"instruction_id":',
                    },
                },
                {"index": 0, "id": "call_same", "function": {"arguments": '"p1"}'}},
                {
                    "index": 0,
                    "id": "call_same",
                    "function": {"name": "forget_memory", "arguments": '{"query":'},
                },
                {
                    "index": 0,
                    "id": "call_same",
                    "function": {"arguments": '"пиратов"}'},
                },
            ]
        )
        self._assert_both_executed(finished)
        self.assertNotEqual(finished[0]["id"], finished[1]["id"])

    def test_distinct_ids_at_the_same_index_stay_distinct(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "id": "call_remove",
                    "function": {
                        "name": "remove_personality_instruction",
                        "arguments": '{"instruction_id":"p1"}',
                    },
                },
                {
                    "index": 0,
                    "id": "call_forget",
                    "function": {
                        "name": "forget_memory",
                        "arguments": '{"query":"пиратов"}',
                    },
                },
            ]
        )
        self._assert_both_executed(finished)
        self.assertEqual(
            [call["id"] for call in finished], ["call_remove", "call_forget"]
        )

    def test_openai_indexes_keep_interleaved_argument_fragments(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "id": "call_remove",
                    "function": {
                        "name": "remove_personality_instruction",
                        "arguments": "",
                    },
                },
                {"index": 0, "function": {"arguments": '{"instruction_id":'}},
                {
                    "index": 1,
                    "id": "call_forget",
                    "function": {"name": "forget_memory", "arguments": ""},
                },
                {"index": 0, "function": {"arguments": '"p1"}'}},
                {"index": 1, "function": {"arguments": '{"query":"пиратов"}'}},
            ]
        )
        self._assert_both_executed(finished)
        self.assertEqual(
            [call["id"] for call in finished], ["call_remove", "call_forget"]
        )

    def test_indexless_named_calls_stay_separate(self) -> None:
        finished = _accumulate(
            [
                {
                    "function": {
                        "name": "remove_personality_instruction",
                        "arguments": '{"instruction_id":"p1"}',
                    }
                },
                {
                    "function": {
                        "name": "forget_memory",
                        "arguments": '{"query":"пиратов"}',
                    }
                },
            ]
        )
        self._assert_both_executed(finished)

    def test_one_call_argument_continuation_is_not_split(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "id": "call_forget",
                    "function": {"name": "forget_memory", "arguments": ""},
                },
                {"index": 0, "function": {"arguments": '{"query":'}},
                {"index": 0, "function": {"arguments": '"пиратов"}'}},
            ]
        )
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["id"], "call_forget")
        self.assertEqual(finished[0]["function"]["name"], "forget_memory")
        self.assertEqual(
            json.loads(finished[0]["function"]["arguments"]),
            {"query": "пиратов"},
        )

    def test_same_name_twice_at_index_zero_stays_two_calls(self) -> None:
        finished = _accumulate(
            [
                {
                    "index": 0,
                    "function": {
                        "name": "forget_memory",
                        "arguments": '{"query":"пиратов"}',
                    },
                },
                {
                    "index": 0,
                    "function": {
                        "name": "forget_memory",
                        "arguments": '{"query":"дверь"}',
                    },
                },
            ]
        )
        self.assertEqual(len(finished), 2)
        self.assertEqual(
            [json.loads(call["function"]["arguments"])["query"] for call in finished],
            ["пиратов", "дверь"],
        )
        results = _result_messages(finished)
        self.assertEqual(len({item["tool_call_id"] for item in results}), 2)

    def _assert_both_executed(self, finished: list[dict]) -> None:
        self.assertEqual(
            [call["function"]["name"] for call in finished],
            ["remove_personality_instruction", "forget_memory"],
        )
        self.assertEqual(
            [json.loads(call["function"]["arguments"]) for call in finished],
            [{"instruction_id": "p1"}, {"query": "пиратов"}],
        )
        results = _result_messages(finished)
        self.assertEqual(
            [item["tool_call_id"] for item in results],
            [call["id"] for call in finished],
        )
        self.assertEqual(len({item["tool_call_id"] for item in results}), 2)
        for call in finished:
            self.assertNotIn("_partial", call["function"])
            self.assertNotIn("_stream_index", call)


if __name__ == "__main__":
    unittest.main()
