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

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

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


# ---------------------------------------------------------------------------
# Typo tolerance
#
# Used only as a fallback when the exact/normalized comparison above found
# nothing. A fuzzy hit is accepted only when it is unambiguous: if two
# different candidates are equally close, nothing is guessed and the caller
# gets the candidate names back so it can tell the model what exists.
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")
_REPEATS = re.compile(r"(.)\1+")

# Keys shorter than this only match after doubled-letter collapse, never by
# edit distance ("Ванна" vs "Ваня", "Зал" vs "Зол").
MIN_FUZZY_LENGTH = 5


def fuzzy_key(value: Any) -> str:
    """match_key with doubled letters collapsed ("гостинная" -> "гостиная")."""
    return _REPEATS.sub(r"\1", match_key(value))


def max_typo_distance(length: int) -> int:
    """Allowed edit distance for a key of this length."""
    if length < MIN_FUZZY_LENGTH:
        return 0
    if length < 8:
        return 1
    return 2


def edit_distance(left: str, right: str, limit: int | None = None) -> int:
    """Damerau-Levenshtein (optimal string alignment) distance.

    When ``limit`` is given the search stops early and returns ``limit + 1``
    once the distance is known to exceed it.
    """
    if left == right:
        return 0
    if limit is not None and abs(len(left) - len(right)) > limit:
        return limit + 1
    previous_previous: list[int] = []
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left_char == right[j - 2]
                and left[i - 2] == right_char
            ):
                current[j] = min(current[j], previous_previous[j - 2] + 1)
        if limit is not None and min(current) > limit:
            return limit + 1
        previous_previous, previous = previous, current
    return previous[-1]


def typo_distance(query: Any, candidate: Any) -> int | None:
    """Distance between two values if it is within the typo budget, else None.

    Values whose numbers differ never match ("Спальня 1" vs "Спальня 2",
    "Зал" vs "Зал 2"): a number is an identifier, not a typo.
    """
    query_key = fuzzy_key(query)
    candidate_key = fuzzy_key(candidate)
    if not query_key or not candidate_key:
        return None
    if _DIGITS.findall(query_key) != _DIGITS.findall(candidate_key):
        return None
    if query_key == candidate_key:
        return 0
    budget = max_typo_distance(min(len(query_key), len(candidate_key)))
    if budget == 0:
        return None
    distance = edit_distance(query_key, candidate_key, budget)
    return distance if distance <= budget else None


def typo_contains(haystack: Any, needle: Any) -> int | None:
    """Fuzzy substring test: best distance of ``needle`` to a word span of ``haystack``.

    The needle is compared with every run of consecutive words in the
    haystack that has the same number of words, so "люстрра" finds
    "Люстра в гостиной" but a single word never fuzzily matches half a word.
    """
    needle_words = match_key(needle).split()
    hay_words = match_key(haystack).split()
    if not needle_words or len(hay_words) < len(needle_words):
        return None
    width = len(needle_words)
    best: int | None = None
    for start in range(len(hay_words) - width + 1):
        distance = typo_distance(
            " ".join(needle_words), " ".join(hay_words[start : start + width])
        )
        if distance is not None and (best is None or distance < best):
            best = distance
    return best


@dataclass
class FuzzyResult:
    """Outcome of a fuzzy lookup: one match, or the tied candidates."""

    match: Any = None
    candidates: list[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.match is None and len(self.candidates) > 1


def best_typo_match(
    query: Any, items: Iterable[Any], names: Callable[[Any], Iterable[Any]]
) -> FuzzyResult:
    """Pick the item whose name (or alias) is uniquely closest to ``query``.

    ``names(item)`` yields the item's name and aliases; the first one is used
    as the display name. Returns an empty result when nothing is within the
    typo budget, and the tied display names when the best distance is shared
    by more than one item.
    """
    best_distance: int | None = None
    best_items: list[tuple[Any, str]] = []
    for item in items:
        item_names = [name for name in names(item) if name]
        if not item_names:
            continue
        distances = [
            d for d in (typo_distance(query, name) for name in item_names) if d is not None
        ]
        if not distances:
            continue
        distance = min(distances)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_items = [(item, str(item_names[0]))]
        elif distance == best_distance:
            best_items.append((item, str(item_names[0])))
    if len(best_items) == 1:
        return FuzzyResult(match=best_items[0][0], candidates=[best_items[0][1]])
    return FuzzyResult(candidates=sorted({name for _, name in best_items}, key=str.casefold))
