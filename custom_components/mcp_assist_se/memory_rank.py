"""Pure helpers for memory ranking, retrieval, and personality text.

No Home Assistant imports. Safe to unit-test on its own.
"""

from __future__ import annotations

import math
import re
from typing import Any

AUTO_KEYWORD_MIN_SCORE = 20
MAX_PERSONALITY_INSTRUCTIONS = 8
MAX_PERSONALITY_INSTRUCTION_CHARS = 240
MAX_PERSONALITY_TOTAL_CHARS = 1200
_AUTO_STOPWORDS = {
    "the",
    "and",
    "for",
    "you",
    "your",
    "that",
    "this",
    "with",
    "from",
    "have",
    "was",
    "are",
    "not",
    "but",
    "please",
    "just",
    "что",
    "это",
    "как",
    "для",
    "или",
    "мне",
    "меня",
    "мой",
    "моя",
    "мои",
    "его",
    "её",
    "ее",
    "они",
    "она",
    "оно",
    "так",
    "уже",
    "ещё",
    "еще",
    "при",
    "про",
    "над",
    "под",
    "включи",
    "включите",
    "поставь",
    "сделай",
}

_PERSONALITY_REQUEST = re.compile(
    r"(?:"
    r"\b(?:talk|speak|answer|respond|reply)\b.{0,48}\b(?:like|as)\b"
    r"|\b(?:call|address)\s+me\b"
    r"|\blike a pirate\b"
    r"|говори(?:ть)?(?:\s+со\s+мной)?\s+как"
    r"|общайся(?:\s+со\s+мной)?"
    r"|обращайся"
    r"|называй меня"
    r"|как пират"
    r"|пиратск"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def normalize_text(value: Any) -> str | None:
    """Collapse whitespace and drop empty values."""
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


def public_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """Return a memory without its cached embedding."""
    return {
        "id": memory.get("id"),
        "text": memory.get("text"),
        "category": memory.get("category"),
        "created_at": memory.get("created_at"),
    }


def keyword_score(memory: dict[str, Any], query: str) -> int:
    """Score one memory against a query. Zero means no keyword overlap.

    This matches the original recall/forget ranking: every word character
    token counts, and a full-query substring is a strong match.
    """
    return _keyword_score(memory, query, ignore_stopwords=False)


def _keyword_score(
    memory: dict[str, Any], query: str, *, ignore_stopwords: bool
) -> int:
    query_lower = query.casefold()
    terms = []
    for term in re.findall(r"\w+", query_lower):
        if ignore_stopwords and (len(term) < 3 or term in _AUTO_STOPWORDS):
            continue
        terms.append(term)
    haystack = " ".join(
        (
            str(memory.get("text") or ""),
            str(memory.get("category") or ""),
        )
    ).casefold()
    score = 0
    if query_lower and query_lower in haystack:
        score += 100
    for term in terms:
        if term in haystack:
            score += 10
    if terms and all(term in haystack for term in terms):
        score += 25
    return score


def rank_memories(
    memories: list[dict[str, Any]],
    *,
    query: str | None,
    category: str | None,
) -> list[dict[str, Any]]:
    """Rank memories by keyword overlap and recency.

    This is the explicit recall/forget ranking. A query with no overlap
    returns nothing. No query returns the category (or everything), newest first.
    """
    if not memories:
        return []

    filtered = memories
    if category:
        filtered = [
            item
            for item in filtered
            if normalize_text(item.get("category")) == category
        ]

    if not query:
        return sorted(
            (dict(item) for item in filtered),
            key=lambda item: item.get("created_at") or "",
            reverse=True,
        )

    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for memory in filtered:
        score = keyword_score(memory, query)
        if not score:
            continue
        ranked.append((score, str(memory.get("created_at") or ""), dict(memory)))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def keyword_auto_matches(
    memories: list[dict[str, Any]],
    query: str,
    *,
    limit: int,
    min_score: int = AUTO_KEYWORD_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Keyword fallback for automatic injection. Weak single-token hits are dropped."""
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for memory in memories:
        score = _keyword_score(memory, query, ignore_stopwords=True)
        if score < min_score:
            continue
        ranked.append((score, str(memory.get("created_at") or ""), memory))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [public_memory(item[2]) for item in ranked[: max(1, limit)]]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity. Returns 0 when either vector is empty or a different length."""
    if not left or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def memories_missing_embedding(
    memories: list[dict[str, Any]],
    model_key: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Newest memories whose cached vector was not built by ``model_key``."""
    missing = [
        memory
        for memory in memories
        if memory.get("embedding_model") != model_key
        or not isinstance(memory.get("embedding"), list)
        or not memory.get("embedding")
    ]
    missing.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return missing[: max(0, limit)]


def rank_by_cosine(
    memories: list[dict[str, Any]],
    query_vector: list[float],
    *,
    model_key: str,
    threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Return public memories at or above ``threshold``, best match first."""
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for memory in memories:
        if memory.get("embedding_model") != model_key:
            continue
        vector = memory.get("embedding")
        if not isinstance(vector, list) or len(vector) != len(query_vector):
            continue
        score = cosine_similarity(query_vector, vector)
        if score < threshold:
            continue
        scored.append((score, str(memory.get("created_at") or ""), memory))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [public_memory(item[2]) for item in scored[: max(1, limit)]]


def round_vector(vector: list[float], places: int = 5) -> list[float]:
    """Shrink a vector for JSON storage without changing cosine in a meaningful way."""
    return [round(float(value), places) for value in vector]


def build_search_text(
    current: str,
    previous: str | None,
    *,
    include_previous: bool,
    previous_limit: int = 240,
) -> str:
    """Utterance used for retrieval, optionally with the previous user turn."""
    current_text = normalize_text(current) or ""
    if not include_previous or not previous:
        return current_text
    previous_text = normalize_text(previous) or ""
    if len(previous_text) > previous_limit:
        previous_text = previous_text[:previous_limit].rstrip()
    if not previous_text or previous_text == current_text:
        return current_text
    return f"{current_text}\n{previous_text}"


def format_remembered_facts(items: list[dict[str, Any]]) -> str:
    """Prompt section for automatically retrieved memories. Empty when there are none."""
    if not items:
        return ""
    lines = [
        "## Remembered facts",
        "These facts were retrieved automatically from long-term memory because they may be relevant to the current request.",
        "Use them when they apply. Do not mention this section unless the user asks what you remember.",
    ]
    for item in items:
        category = normalize_text(item.get("category"))
        prefix = f"[{category}] " if category else ""
        text = normalize_text(item.get("text")) or ""
        if text:
            lines.append(f"- {prefix}{text}")
    if len(lines) == 3:
        return ""
    return "\n".join(lines)


def format_personality_block(instructions: list[str]) -> str:
    """Personality section injected on every turn while the feature is enabled."""
    lines = [
        "## Personality and behaviour",
        "These instructions set tone, persona, and how you address the user. Follow them on every reply.",
        "If they conflict with the general system prompt, follow them for tone and address.",
        "They do not override device-control rules: still discover entities and call perform_action to change devices.",
        "Update them with set_personality_instruction, remove_personality_instruction, or reset_personality.",
        "Do not store speaking style with remember_memory.",
    ]
    cleaned: list[str] = []
    for item in instructions:
        if isinstance(item, dict):
            text = normalize_text(item.get("text")) or ""
            instruction_id = normalize_text(item.get("id"))
            if text:
                cleaned.append(f"[{instruction_id}] {text}" if instruction_id else text)
            continue
        text = normalize_text(item) or ""
        if text:
            cleaned.append(text)
    if not cleaned:
        lines.append("No personality instructions are stored yet.")
        return "\n".join(lines)
    lines.append("If the user replaces a style, replace the existing instruction or reset the block. Do not stack contradictory styles.")
    for text in cleaned:
        lines.append(f"- {text}")
    return "\n".join(lines)


def looks_like_personality_request(text: str) -> bool:
    """True when text is about speaking style rather than a stored fact."""
    normalized = normalize_text(text) or ""
    if not normalized:
        return False
    return _PERSONALITY_REQUEST.search(normalized) is not None


def personality_char_count(instructions: list[dict[str, Any]]) -> int:
    """Total stored personality characters."""
    return sum(len(str(item.get("text") or "")) for item in instructions)


def validate_personality_text(text: str) -> str | None:
    """Return an error message, or None when the instruction can be stored."""
    normalized = normalize_text(text)
    if not normalized:
        return "Personality instruction text is required."
    if len(normalized) > MAX_PERSONALITY_INSTRUCTION_CHARS:
        return (
            f"That instruction is too long ({len(normalized)} characters). "
            f"Keep it under {MAX_PERSONALITY_INSTRUCTION_CHARS} characters."
        )
    return None
