"""Upstream behaviors ported without a full merge.

Covers ordered tool calls, Assist exposure checks, a plain entity_id,
unclosed thinking tags, page text, and discovery filters.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_components" / "mcp_assist_se"
sys.path.insert(0, str(ROOT / "custom_components"))

if "mcp_assist_se" not in sys.modules:
    package = types.ModuleType("mcp_assist_se")
    package.__path__ = [str(PACKAGE)]
    package.__package__ = "mcp_assist_se"
    sys.modules["mcp_assist_se"] = package

from mcp_assist_se.action_args import (  # noqa: E402
    explicit_entity_ids,
    exposure_refusal_text,
    normalize_action_arguments,
    unexposed_entity_ids,
)
from mcp_assist_se.discovery_filters import (  # noqa: E402
    area_listing_applies,
    domain_means_any,
    empty_discovery_message,
    name_is_aggregate_query,
)
from mcp_assist_se.model_output import prompt_clock_value, strip_thinking_tags  # noqa: E402
from mcp_assist_se.tool_batch import (  # noqa: E402
    READ_ONLY_TOOLS,
    batch_runs_in_parallel,
    execute_tool_batch,
)


def _load_read_url():
    spec = importlib.util.spec_from_file_location(
        "mcp_assist_se_read_url_under_test",
        PACKAGE / "custom_tools" / "read_url.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(name: str) -> dict:
    return {"function": {"name": name}}


class ToolBatchTests(unittest.TestCase):
    def test_read_only_batch_may_run_together(self):
        names = ["discover_entities", "read_url", "recall_memories", "convert_unit"]
        self.assertTrue(batch_runs_in_parallel(names))
        for name in names:
            self.assertIn(name, READ_ONLY_TOOLS)

    def test_a_device_command_forces_order(self):
        self.assertFalse(
            batch_runs_in_parallel(["discover_entities", "perform_action"])
        )
        self.assertFalse(
            batch_runs_in_parallel(["perform_action", "perform_action"])
        )
        self.assertFalse(batch_runs_in_parallel(["remember_memory", "get_index"]))
        self.assertFalse(batch_runs_in_parallel(["run_script"]))
        self.assertFalse(batch_runs_in_parallel(["discover_entities"]))

    def test_unknown_tools_are_not_read_only(self):
        self.assertFalse(batch_runs_in_parallel(["discover_entities", "custom_switch"]))

    def test_state_changes_do_not_overlap(self):
        order: list[str] = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def run(call):
            name = call["function"]["name"]
            order.append(f"start {name}")
            if name == "perform_action":
                started.set()
                await release.wait()
            order.append(f"end {name}")
            return name

        async def scenario():
            task = asyncio.create_task(
                execute_tool_batch(
                    [_call("perform_action"), _call("discover_entities")],
                    run,
                )
            )
            await started.wait()
            self.assertEqual(order, ["start perform_action"])
            release.set()
            result = await task
            self.assertEqual(result, ["perform_action", "discover_entities"])

        asyncio.run(scenario())
        self.assertLess(
            order.index("end perform_action"),
            order.index("start discover_entities"),
        )

    def test_read_only_calls_overlap(self):
        active = 0
        peak = 0

        async def run(call):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return call["function"]["name"]

        async def scenario():
            return await execute_tool_batch(
                [_call("discover_entities"), _call("read_url"), _call("get_index")],
                run,
            )

        result = asyncio.run(scenario())
        self.assertEqual(peak, 3)
        self.assertEqual(result, ["discover_entities", "read_url", "get_index"])

    def test_two_actions_stay_in_order(self):
        order: list[str] = []

        async def run(call):
            name = call["id"]
            order.append(f"start {name}")
            await asyncio.sleep(0)
            order.append(f"end {name}")
            return name

        async def scenario():
            return await execute_tool_batch(
                [
                    {"id": "first", "function": {"name": "perform_action"}},
                    {"id": "second", "function": {"name": "perform_action"}},
                ],
                run,
            )

        result = asyncio.run(scenario())
        self.assertEqual(result, ["first", "second"])
        self.assertEqual(
            order,
            ["start first", "end first", "start second", "end second"],
        )


class ThinkingTagTests(unittest.TestCase):
    def test_closed_tag_is_removed(self):
        spoken, notes = strip_thinking_tags("<think>plan</think>The light is on.")
        self.assertEqual(spoken, "The light is on.")
        self.assertEqual(notes, "plan")

    def test_unclosed_tag_is_not_spoken(self):
        spoken, notes = strip_thinking_tags("<think>still planning the hallway")
        self.assertEqual(spoken, "")
        self.assertIn("hallway", notes)

    def test_closing_tag_without_an_opening_one(self):
        spoken, notes = strip_thinking_tags("private note</think>The lamp is off.")
        self.assertEqual(spoken, "The lamp is off.")
        self.assertIn("private note", notes)

    def test_plain_text_is_unchanged(self):
        text = "  Already fine.\n"
        self.assertEqual(strip_thinking_tags(text), (text, ""))

    def test_clock_stays_digits_with_seconds(self):
        clock = prompt_clock_value(datetime(2026, 11, 9, 15, 4, 5))
        self.assertEqual(clock, "15:04:05")
        self.assertNotIn("fifteen", clock.lower())


class PerformActionArgumentTests(unittest.TestCase):
    def test_plain_entity_id_is_accepted(self):
        target, data = normalize_action_arguments(
            {
                "domain": "light",
                "action": "turn_on",
                "entity_id": "light.kitchen",
                "data": {"brightness_pct": 40},
            }
        )
        self.assertEqual(target["entity_id"], ["light.kitchen"])
        self.assertEqual(data, {"brightness_pct": 40})
        self.assertEqual(explicit_entity_ids(target), ["light.kitchen"])

    def test_entity_id_hidden_in_data_moves_to_the_target(self):
        target, data = normalize_action_arguments(
            {
                "domain": "light",
                "action": "turn_on",
                "target": {"entity_id": "light.kitchen"},
                "data": {
                    "entity_id": "light.secret",
                    "device_id": "device-1",
                    "brightness_pct": 10,
                },
            }
        )
        self.assertEqual(target["entity_id"], ["light.kitchen", "light.secret"])
        self.assertEqual(target["device_id"], ["device-1"])
        self.assertEqual(data, {"brightness_pct": 10})

    def test_hidden_entity_is_refused_and_named(self):
        exposed = {"light.kitchen"}
        refused = unexposed_entity_ids(
            ["light.kitchen", "light.secret"],
            lambda entity_id: entity_id in exposed,
        )
        self.assertEqual(refused, ["light.secret"])
        text = exposure_refusal_text(refused, ["light.kitchen", "light.secret"])
        self.assertIn("light.secret", text)
        self.assertIn("not exposed", text)
        self.assertNotIn("light.kitchen", text)

    def test_device_expansion_does_not_name_hidden_ids(self):
        text = exposure_refusal_text(
            ["sensor.secret_diag"],
            requested=[],
        )
        self.assertNotIn("sensor.secret_diag", text)
        self.assertIn("not exposed", text)
        self.assertIn("not sent", text)

    def test_all_is_refused(self):
        text = exposure_refusal_text(["all"], ["all"])
        self.assertIn("all", text.lower())
        self.assertIn("not allowed", text)


class DiscoveryFilterTests(unittest.TestCase):
    def test_hallway_is_not_a_request_for_everything(self):
        self.assertFalse(name_is_aggregate_query("hallway"))
        self.assertFalse(name_is_aggregate_query("ball"))
        self.assertTrue(name_is_aggregate_query("all lights"))
        self.assertTrue(name_is_aggregate_query("who is home"))

    def test_room_plus_name_is_not_a_room_dump(self):
        self.assertFalse(
            area_listing_applies(
                area="Kitchen",
                name_contains="lamp",
                floor=None,
                label=None,
                device_class=None,
                name_pattern=None,
                inferred_type=None,
            )
        )
        self.assertTrue(
            area_listing_applies(
                area="Kitchen",
                name_contains=None,
                floor=None,
                label=None,
                device_class=None,
                name_pattern=None,
                inferred_type=None,
            )
        )

    def test_domain_all_means_every_type(self):
        self.assertTrue(domain_means_any("all"))
        self.assertTrue(domain_means_any(" * "))
        self.assertTrue(domain_means_any("Any"))
        self.assertFalse(domain_means_any("light"))
        self.assertFalse(domain_means_any(None))

    def test_empty_result_explains_the_next_try(self):
        text = empty_discovery_message(
            {"area": "Attic", "name_contains": "lamp"},
            area_is_known=True,
        )
        self.assertIn("fewer filters", text)
        unknown = empty_discovery_message(
            {"area": "Attic"},
            area_is_known=False,
            known_area_names=["Kitchen", "Hall"],
        )
        self.assertIn("Kitchen", unknown)
        hidden = empty_discovery_message(
            {"area": "Attic"},
            area_is_known=False,
            known_area_names=None,
        )
        self.assertNotIn("Kitchen", hidden)
        self.assertIn("not a known area", hidden)


class ReadUrlExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extract = _load_read_url().ReadUrlTool._extract_text

    def test_main_text_is_kept_and_navigation_is_dropped(self):
        html = """
        <html><head><title>News</title>
        <script>secret()</script><style>.nav{}</style></head>
        <body>
          <header>Site header</header>
          <nav>Home About</nav>
          <main><h1>The story</h1><p>Kitchen lamp is on &amp; bright.</p></main>
          <footer>Copyright</footer>
        </body></html>
        """
        text = ReadUrlExtractTests.extract(html, "text/html")
        self.assertIn("The story", text)
        self.assertIn("Kitchen lamp is on & bright.", text)
        self.assertNotIn("Site header", text)
        self.assertNotIn("Home About", text)
        self.assertNotIn("Copyright", text)
        self.assertNotIn("secret()", text)

    def test_plain_text_is_unchanged(self):
        self.assertEqual(
            ReadUrlExtractTests.extract("just text", "text/plain"), "just text"
        )


if __name__ == "__main__":
    unittest.main()
