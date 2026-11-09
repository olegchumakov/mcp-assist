"""Clean model text before it is shown or spoken.

The clock value passed into the prompt stays digits. The model is the one
that turns that time into words.
"""

from __future__ import annotations

import re
from datetime import datetime

_CLOSED_THOUGHTS = (
    r"<think>(.*?)</think>",
    r"<\|thought\|>(.*?)<\|/thought\|>",
)
_UNCLOSED_END = (r"</think>", r"<\|/thought\|>")
_UNCLOSED_START = (r"<think>", r"<\|thought\|>")


def prompt_clock_value(now: datetime) -> str:
    """Current time for the prompt, as digits with seconds (HH:MM:SS).

    Do not spell this out, and do not drop the seconds. The language model
    turns the digits into words.
    """
    return now.strftime("%H:%M:%S")


def strip_thinking_tags(text: str) -> tuple[str, str]:
    """Remove reasoning notes from model output.

    Closed blocks are removed first. A closing tag with no opening tag means
    the note started in the prompt, so everything before it is reasoning. An
    opening tag that is never closed means the rest of the output is reasoning
    and must not be spoken.

    Returns (spoken text, reasoning text). Text with no tags is returned
    unchanged.
    """
    if not text:
        return text, ""

    patterns = list(_CLOSED_THOUGHTS)
    all_matches: list[str] = []
    cleaned_text = text

    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        all_matches.extend(matches)
        cleaned_text = re.sub(
            pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
        )

    for close_tag in _UNCLOSED_END:
        match = re.search(
            rf"^(.*){close_tag}", cleaned_text, re.DOTALL | re.IGNORECASE
        )
        if match:
            all_matches.append(match.group(1))
            cleaned_text = cleaned_text[match.end() :]
    for open_tag in _UNCLOSED_START:
        match = re.search(
            rf"{open_tag}(.*)$", cleaned_text, re.DOTALL | re.IGNORECASE
        )
        if match:
            all_matches.append(match.group(1))
            cleaned_text = cleaned_text[: match.start()]

    if not all_matches:
        return text, ""

    thinking_content = "\n".join(all_matches)
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text, thinking_content
