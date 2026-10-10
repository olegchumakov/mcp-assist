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


# Command words stripped before forget/dedupe. Subjects such as "пират" stay.
_SUBJECT_STOPWORDS = _AUTO_STOPWORDS | {
    "забудь",
    "забудьте",
    "забыть",
    "запомни",
    "запомните",
    "вспомни",
    "вспомнить",
    "удали",
    "удалить",
    "удалите",
    "сотри",
    "стереть",
    "перестань",
    "перестаньте",
    "хватит",
    "больше",
    "пожалуйста",
    "все",
    "всего",
    "всех",
    "это",
    "эту",
    "этот",
    "эта",
    "эти",
    "того",
    "том",
    "мне",
    "меня",
    "мной",
    "мое",
    "моё",
    "твой",
    "твоя",
    "ты",
    "вы",
    "он",
    "она",
    "они",
    "не",
    "чтобы",
    "надо",
    "хочу",
    "говори",
    "говорить",
    "говорит",
    "говорите",
    "разговаривать",
    "разговаривай",
    "общайся",
    "обращайся",
    "forget",
    "forgot",
    "forgotten",
    "remember",
    "about",
    "everything",
    "all",
    "stop",
    "talk",
    "talking",
    "speak",
    "speaking",
    "like",
    "should",
    "must",
}

# Longest first so "пиратов" loses "ов", not a shorter tail.
_RU_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ах",
    "ях",
    "ов",
    "ев",
    "ом",
    "ем",
    "ой",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ую",
    "юю",
    "ам",
    "ям",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "и",
    "ы",
    "о",
    "ь",
)
_EN_SUFFIXES = ("ing", "ers", "ies", "es", "ed", "s")
_FORGET_ALL_RE = re.compile(r"\b(?:everything|all|все|всего|всех)\b", re.IGNORECASE)


def fold_token(token: str) -> str:
    """Casefold and treat ё as е so inflection checks share one form."""
    return token.casefold().replace("ё", "е")


def stem_token(token: str) -> str:
    """Strip one Russian or English inflectional ending.

    "пиратов" and "пиратах" become "пират". Short stems (at least 3
    characters) are kept so "кота" still meets "кот". A longer unrelated
    word is not collapsed onto a 4-letter prefix; that check is separate.
    """
    folded = fold_token(token)
    if len(folded) <= 3:
        return folded
    suffixes = _RU_SUFFIXES if _has_cyrillic(folded) else _EN_SUFFIXES
    minimum_stem = 3 if _has_cyrillic(folded) else 4
    for suffix in suffixes:
        if len(suffix) >= len(folded):
            continue
        if not folded.endswith(suffix):
            continue
        stem = folded[: -len(suffix)]
        if len(stem) >= minimum_stem:
            return stem
    return folded


def _has_cyrillic(text: str) -> bool:
    return any("а" <= char <= "я" or char == "ё" for char in text)


def stems_match(left: str, right: str) -> bool:
    """True when two stems are the same word, including a short leftover suffix."""
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) < 5 or not long.startswith(short):
        return False
    return len(long) - len(short) <= 2


def content_stems(text: str) -> list[str]:
    """Subject stems, without forget/remember command words."""
    stems: list[str] = []
    for token in re.findall(r"\w+", text or ""):
        folded = fold_token(token)
        if len(folded) < 3 or folded in _SUBJECT_STOPWORDS:
            continue
        stem = stem_token(folded)
        if len(stem) < 3 or stem in _SUBJECT_STOPWORDS:
            continue
        stems.append(stem)
    return stems


def inflection_match(query: str, text: str) -> bool:
    """True when a query stem meets a stem in the stored text.

    Substring matching fails for Russian: "пиратов" is not contained in
    "пират", while the stem of both is "пират".
    """
    query_stems = content_stems(query)
    text_stems = content_stems(text)
    if not query_stems or not text_stems:
        return False
    return any(
        any(stems_match(query_stem, text_stem) for text_stem in text_stems)
        for query_stem in query_stems
    )


def texts_look_like_duplicates(left: str, right: str) -> bool:
    """Stem fallback used when embeddings are not loaded yet."""
    left_stems = set(content_stems(left))
    right_stems = set(content_stems(right))
    if not left_stems or not right_stems:
        return False
    shared = {
        stem
        for stem in left_stems
        if any(stems_match(stem, other) for other in right_stems)
    }
    if not shared:
        return False
    smaller = min(len(left_stems), len(right_stems))
    if len(shared) >= 2 and len(shared) / smaller >= 0.6:
        return True
    if len(shared) == 1 and smaller == 1 and len(next(iter(shared))) >= 5:
        return True
    return False


def query_requests_forget_all(query: str | None) -> bool:
    """True for "forget everything" / "забудь всё про ..."."""
    if not query:
        return False
    return _FORGET_ALL_RE.search(fold_token(query)) is not None


def _recency(item: dict[str, Any]) -> str:
    return str(item.get("updated_at") or item.get("created_at") or "")


def select_forget_matches(
    items: list[dict[str, Any]],
    query: str | None,
    *,
    category: str | None = None,
    delete_all: bool = False,
    similarities: dict[str, float] | None = None,
    threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """Choose memories or personality lines to delete.

    A hit is an inflection/stem match or a cosine at or above ``threshold``.
    "Forget everything about X" deletes every hit. Otherwise only the best
    hit is deleted. No query with a category deletes that category.
    """
    normalized_category = normalize_text(category)
    filtered = items
    if normalized_category:
        filtered = [
            item
            for item in items
            if normalize_text(item.get("category")) == normalized_category
        ]
    normalized_query = normalize_text(query)
    if not normalized_query:
        if not normalized_category:
            return []
        ordered = sorted(filtered, key=_recency, reverse=True)
        chosen = ordered if delete_all else ordered[:1]
        return [dict(item) for item in chosen]

    forget_all = delete_all or query_requests_forget_all(normalized_query)
    hits: list[tuple[float, int, str, dict[str, Any]]] = []
    for item in filtered:
        similarity = 0.0
        if similarities:
            raw = similarities.get(str(item.get("id")))
            if isinstance(raw, (int, float)):
                similarity = float(raw)
        stem_hit = inflection_match(normalized_query, str(item.get("text") or ""))
        if not stem_hit and similarity < threshold:
            continue
        hits.append((similarity, 1 if stem_hit else 0, _recency(item), item))
    if not hits:
        return []
    hits.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    chosen_rows = hits if forget_all else hits[:1]
    return [dict(row[3]) for row in chosen_rows]


def select_duplicate(
    items: list[dict[str, Any]],
    new_text: str,
    new_category: str | None,
    similarities: dict[str, float] | None,
    threshold: float,
) -> dict[str, Any] | None:
    """Pick the existing row a new fact should replace, if it is the same subject.

    When ``new_category`` is set, only a row with that same category qualifies.
    A scored row below ``threshold`` is not a duplicate. Rows without a score
    can still match by stem overlap, which covers the period before the model loads.
    """
    category = normalize_text(new_category)
    best: tuple[float, str, dict[str, Any]] | None = None
    for item in items:
        if category and normalize_text(item.get("category")) != category:
            continue
        item_id = str(item.get("id"))
        score: float | None = None
        if similarities is not None and item_id in similarities:
            if float(similarities[item_id]) < threshold:
                continue
            score = float(similarities[item_id])
        elif texts_look_like_duplicates(new_text, str(item.get("text") or "")):
            score = threshold
        if score is None:
            continue
        candidate = (score, _recency(item), item)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return dict(best[2]) if best else None


def plan_duplicate_merges(
    items: list[dict[str, Any]],
    matrix: dict[str, dict[str, float]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Group near-duplicates. Each group keeps the newest text and lists the rest.

    Two rows with different non-empty categories are not merged.
    """
    if len(items) < 2:
        return []
    by_id = {str(item["id"]): item for item in items}
    parent = {item_id: item_id for item_id in by_id}

    def find(item_id: str) -> str:
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def members(root: str) -> list[str]:
        return [item_id for item_id in by_id if find(item_id) == root]

    def categories_compatible(left_ids: list[str], right_ids: list[str]) -> bool:
        categories = {
            category
            for item_id in left_ids + right_ids
            if (category := normalize_text(by_id[item_id].get("category")))
        }
        return len(categories) <= 1

    ids = list(by_id)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            score = (matrix.get(left) or {}).get(right)
            if score is None:
                score = (matrix.get(right) or {}).get(left, 0.0)
            if float(score) < threshold:
                continue
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                continue
            if not categories_compatible(members(left_root), members(right_root)):
                continue
            parent[right_root] = left_root

    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item_id in ids:
        root = find(item_id)
        if root in seen:
            continue
        seen.add(root)
        grouped = [by_id[member_id] for member_id in ids if find(member_id) == root]
        if len(grouped) < 2:
            continue
        grouped.sort(key=_recency, reverse=True)
        groups.append(
            {
                "keep": dict(grouped[0]),
                "remove": [dict(item) for item in grouped[1:]],
            }
        )
    return groups


def format_forget_result(
    deleted_memories: list[dict[str, Any]],
    deleted_personality: list[dict[str, Any]],
    *,
    searched_personality: bool,
    searched_memories: bool,
) -> str:
    """Tool text that states exactly what was deleted.

    An empty result tells the model not to claim success.
    """
    if not deleted_memories and not deleted_personality:
        return (
            "DELETED NOTHING. No stored memory and no personality instruction matched. "
            "Tell the user that nothing was deleted. Do not say that you forgot it."
        )
    lines = ["DELETED the following items. Tell the user only about these."]
    if deleted_memories:
        lines.append(f"Deleted {len(deleted_memories)} memory item(s):")
        for memory in deleted_memories[:10]:
            category = normalize_text(memory.get("category"))
            category_text = f" [{category}]" if category else ""
            lines.append(f"- {memory.get('id')}{category_text}: {memory.get('text')}")
        if len(deleted_memories) > 10:
            lines.append(f"{len(deleted_memories) - 10} additional deleted memories were omitted.")
    elif searched_memories:
        lines.append("No stored memory matched.")
    if deleted_personality:
        lines.append(f"Removed {len(deleted_personality)} personality instruction(s):")
        for item in deleted_personality[:10]:
            lines.append(f"- {item.get('id')}: {item.get('text')}")
    elif searched_personality:
        lines.append("No personality instruction matched.")
    return "\n".join(lines)


def format_merge_result(
    memory_groups: list[dict[str, Any]],
    personality_groups: list[dict[str, Any]],
) -> str:
    """Tool text for a duplicate merge. Names every removed line and the text that was kept."""
    if not memory_groups and not personality_groups:
        return (
            "No near-duplicate memories or personality instructions were found. "
            "Nothing was deleted."
        )
    lines = [
        "Merged near-duplicates. The newest text in each group was kept and the older wording was removed."
    ]
    for label, groups in (
        ("memory", memory_groups),
        ("personality instruction", personality_groups),
    ):
        for group in groups:
            keep = group["keep"]
            lines.append(
                f"Kept {label} [{keep.get('id')}]: {keep.get('text')}"
            )
            for item in group["remove"]:
                lines.append(f"Removed older {label} [{item.get('id')}]: {item.get('text')}")
    lines.append("Tell the user about these kept and removed lines.")
    return "\n".join(lines)


def format_remember_result(
    stored: dict[str, Any],
    *,
    replaced_text: str | None,
) -> str:
    """Tool text for a new memory or an in-place replacement."""
    category = normalize_text(stored.get("category"))
    category_text = f" Category: {category}." if category else ""
    if replaced_text:
        return (
            f"UPDATED memory [{stored.get('id')}] instead of adding a duplicate. "
            f"The previous text was: {replaced_text}. "
            f"The stored text is now: {stored.get('text')}.{category_text} "
            "Tell the user you updated the existing memory."
        )
    return (
        f"Stored memory [{stored.get('id')}].{category_text} "
        "It is kept until the user asks to forget it."
    )


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
        "If the user says to forget or stop a style, remove it here. Also check stored memories, because a style may have been saved as a fact.",
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
