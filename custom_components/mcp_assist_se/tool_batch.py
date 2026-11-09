"""Decide which tool calls may run together.

A batch that only looks things up can run at the same time. Any call that
changes something — a device, a memory, a script — makes the whole batch run
one after another, in the order the model asked for.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

_LOGGER = logging.getLogger(__name__)

# No side effects. Anything not listed here is treated as a change, including
# tools added later, so a new tool cannot run beside a device command.
READ_ONLY_TOOLS = frozenset(
    {
        "discover_entities",
        "get_entity_details",
        "get_entity_history",
        "list_areas",
        "list_domains",
        "get_index",
        "search",
        "read_url",
        "recall_memories",
        "view_personality",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sqrt",
        "power",
        "round_number",
        "average",
        "min_value",
        "max_value",
        "evaluate_expression",
        "convert_unit",
    }
)


def tool_call_name(tool_call: dict[str, Any]) -> str | None:
    """Return the function name of one OpenAI-style tool call."""
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def batch_runs_in_parallel(tool_names: list[str | None]) -> bool:
    """True only when every call in a multi-call batch is read-only."""
    return len(tool_names) > 1 and all(name in READ_ONLY_TOOLS for name in tool_names)


async def execute_tool_batch(
    tool_calls: list[dict[str, Any]],
    run_one: Callable[[dict[str, Any]], Awaitable[Any]],
) -> list[Any]:
    """Run tool calls, keeping side effects in the requested order."""
    if not tool_calls:
        return []

    names = [tool_call_name(tool_call) for tool_call in tool_calls]
    if batch_runs_in_parallel(names):
        _LOGGER.debug(
            "Running %d read-only tool calls at the same time", len(tool_calls)
        )
        return list(
            await asyncio.gather(*(run_one(tool_call) for tool_call in tool_calls))
        )

    if len(tool_calls) > 1:
        _LOGGER.debug(
            "Running %d tool calls one after another", len(tool_calls)
        )
    results = []
    for tool_call in tool_calls:
        results.append(await run_one(tool_call))
    return results
