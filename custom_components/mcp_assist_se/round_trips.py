"""Decide when a model turn can end without another LLM round trip.

Some tools only record bookkeeping for the integration itself (for example
whether the assistant expects a reply). Their result carries nothing the model
needs to finish its answer, so when a turn already contains the final text and
calls only such tools, the agent can run them and return the text right away
instead of asking the model again.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# Tools whose result is never needed by the model to complete its turn.
NON_INFORMATIONAL_TOOLS: frozenset[str] = frozenset({"set_conversation_state"})


def tool_call_names(tool_calls: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """Names of the functions in OpenAI-style tool calls."""
    names: list[str] = []
    for call in tool_calls or ():
        function = call.get("function") or {}
        names.append(str(function.get("name") or ""))
    return names


def can_finish_without_followup(
    text: str | None,
    tool_calls: Iterable[Mapping[str, Any]] | None,
    non_informational: frozenset[str] = NON_INFORMATIONAL_TOOLS,
) -> bool:
    """True when the turn has final text and only non-informational tool calls."""
    if not text or not text.strip():
        return False
    names = tool_call_names(tool_calls)
    if not names:
        return False
    return all(name in non_informational for name in names)
