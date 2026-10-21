"""Script-tolerant comparison keys for names, areas, floors, labels and aliases.

Language models sometimes write a Cyrillic word with a Latin look-alike letter
inside (for example a Latin "a" in an otherwise Cyrillic room name). Exact or
case-insensitive comparison then misses. ``match_key`` folds case, treats
"ё" as "е" and maps Latin letters that look like Cyrillic ones to their
Cyrillic twins. Both sides of every comparison go through the same key, so the
mapping is symmetric: pure-Latin names still match each other exactly as
before, and a name typed with mixed scripts matches the registry name.
"""

from __future__ import annotations

from typing import Any, Iterable

# Latin letter (after casefold) -> visually identical Cyrillic letter.
_LATIN_TO_CYRILLIC = str.maketrans(
    {
        "a": "а",
        "e": "е",
        "o": "о",
        "p": "р",
        "c": "с",
        "y": "у",
        "x": "х",
        "k": "к",
        "m": "м",
        "t": "т",
        "h": "н",
        "b": "в",
        "ё": "е",
    }
)


def match_key(value: Any) -> str:
    """Comparison key: casefold, ё→е, Latin look-alikes→Cyrillic, trimmed."""
    if value is None:
        return ""
    return " ".join(str(value).casefold().translate(_LATIN_TO_CYRILLIC).split())


def keys_equal(left: Any, right: Any) -> bool:
    """True when both values have the same non-empty comparison key."""
    left_key = match_key(left)
    return bool(left_key) and left_key == match_key(right)


def key_contains(haystack: Any, needle: Any) -> bool:
    """True when needle's key is a substring of haystack's key."""
    needle_key = match_key(needle)
    return bool(needle_key) and needle_key in match_key(haystack)


def any_key_equal(value: Any, candidates: Iterable[Any] | None) -> bool:
    """True when value's key equals the key of any candidate."""
    return any(keys_equal(value, candidate) for candidate in candidates or ())
