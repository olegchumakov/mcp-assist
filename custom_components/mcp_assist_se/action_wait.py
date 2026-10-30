"""Wait for each action target to report its outcome.

Kept free of Home Assistant imports so it can be tested on its own. The
caller passes a snapshot reader and a subscribe function; in Home
Assistant that is async_track_state_change_event, so a fast device ends
the wait as soon as its state event arrives.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

Snapshot = dict[str, Any]


async def wait_for_targets(
    entity_ids: list[str],
    before: dict[str, Snapshot],
    plans: dict[str, tuple[str, float]],
    read: Callable[[str], Snapshot],
    satisfied: Callable[[str, str, Snapshot, Snapshot], bool],
    subscribe: Callable[[list[str], Callable[[Any], None]], Callable[[], None]],
) -> tuple[dict[str, Snapshot], dict[str, float], dict[str, bool]]:
    """Wait for every target separately.

    plans maps entity_id to (mode, longest wait). satisfied(entity_id, mode,
    before, latest) says when one target is done. A target that is done
    stops being waited on; the others keep their own deadline.

    Returns the latest snapshots, the seconds each target was waited on,
    and whether each target ended early because it was satisfied.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    latest = {entity_id: before[entity_id] for entity_id in entity_ids}
    waited: dict[str, float] = {}
    confirmed: dict[str, bool] = {}
    if not entity_ids:
        return latest, waited, confirmed

    wake = asyncio.Event()

    def _on_change(_event: Any) -> None:
        wake.set()

    unsubscribe = subscribe(list(entity_ids), _on_change)
    try:
        while True:
            # Clear before reading so an event between the read and the
            # wait below still wakes us.
            wake.clear()
            elapsed = loop.time() - start
            pending: list[str] = []
            for entity_id in entity_ids:
                if entity_id in waited:
                    continue
                mode, limit = plans[entity_id]
                latest[entity_id] = read(entity_id)
                if satisfied(entity_id, mode, before[entity_id], latest[entity_id]):
                    waited[entity_id] = elapsed
                    confirmed[entity_id] = True
                elif elapsed >= limit:
                    waited[entity_id] = elapsed
                    confirmed[entity_id] = False
                else:
                    pending.append(entity_id)
            if not pending:
                return latest, waited, confirmed
            remaining = min(plans[entity_id][1] for entity_id in pending) - elapsed
            try:
                await asyncio.wait_for(wake.wait(), timeout=max(remaining, 0.0))
            except TimeoutError:
                pass
    finally:
        unsubscribe()
