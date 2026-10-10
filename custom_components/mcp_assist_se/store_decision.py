"""One short chat completion that chooses add, update, or noop.

The call is capped so a slow gateway cannot turn remember into a long turn.
A timeout or a bad payload means the caller stores nothing.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

STORE_DECISION_TIMEOUT = 3.0


def openai_chat_url(base_url: str) -> str:
    """Build an OpenAI-compatible chat URL without doubling /v1."""
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


async def complete_store_decision(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float = STORE_DECISION_TIMEOUT,
) -> str | None:
    """Return the completion text, or None when the call does not finish."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 60,
        "messages": [
            {
                "role": "system",
                "content": "Reply with one JSON object and no other text.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    import aiohttp

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    _LOGGER.warning(
                        "Store decision failed: HTTP %s %s",
                        response.status,
                        body[:200],
                    )
                    return None
                data = await response.json(content_type=None)
    except Exception as err:
        _LOGGER.warning("Store decision skipped: %s", err)
        return None
    return _message_text(data)


def _message_text(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None
