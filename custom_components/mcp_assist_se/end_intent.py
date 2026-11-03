"""Local check for phrases that should end a conversation.

Speech-to-text that recognizes the speaker prepends a label to the
transcript. Two shapes show up here:

- a plain label in square brackets, such as ``[speaker: name] stop``
- a signed tag in ⟦ ⟧ when the Speaker ID add-on is signing

The label is not something the person said. Stop words are decided on the
words after it. The transcript that continues on to the language model is
left untouched, label included, so per-person access control and speaker
context keep working.
"""

from __future__ import annotations

import re

from .speaker_tag import TAG_CLOSE, TAG_OPEN

# One label at the very start. Brackets later in the sentence stay.
_LEADING_LABEL = re.compile(
    r"^\s*(?:\[[^\]\r\n]*\]|"
    + re.escape(TAG_OPEN)
    + r"[^"
    + re.escape(TAG_CLOSE)
    + r"\r\n]*"
    + re.escape(TAG_CLOSE)
    + r")\s*"
)

# Sentence marks speech-to-text sticks to a word. Apostrophes stay, so
# "that's" and "war's" still match the configured phrase.
_EDGE_PUNCT = ".,!?;:…。！？，、¡¿\"“”«»„‹›()[]{}—–-" + TAG_OPEN + TAG_CLOSE


def strip_leading_speaker_label(text: str) -> str:
    """Remove one leading ``[...]`` or ``⟦...⟧`` label and the space after it.

    A label that is not at the start is kept. The returned string is a new
    value; callers that still need the label pass the original text on.
    """
    if not text:
        return ""
    return _LEADING_LABEL.sub("", text, count=1)


def _spoken_tokens(text: str) -> list[str]:
    """Lowercased words with sentence punctuation peeled off both ends."""
    tokens: list[str] = []
    for raw in text.lower().split():
        token = raw.strip(_EDGE_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def _normalize_spoken(text: str) -> str:
    return " ".join(_spoken_tokens(text))


def detect_user_ending_intent(
    text: str | None,
    end_words: str | None,
    agent_name: str | None = "",
) -> bool:
    """Return True when the utterance should end the conversation locally.

    Stopwords+1: the text contains a configured stop word or phrase, and at
    most one other word remains after the agent name and matched phrases are
    removed. A leading speaker label is ignored. An empty utterance, or a
    transcript that is only a label, is not an ending.

    Examples:
    - "stop" → True (0 other words)
    - "no thanks" → True (both are stop words)
    - "no thank you" → True ("thank you" is a stop phrase)
    - "bye Jarvis" → True (Jarvis removed, 0 other words)
    - "ok please" → True (1 other word)
    - "no turn on light" → False (3 other words)
    - "[speaker: name] stop" → True (same as "stop")
    - "[speaker: name] stop!" → True
    - "[speaker: name] turn on the light" → False
    """
    if not text or not end_words:
        return False

    end_words_raw: list[str] = []
    for word in end_words.split(","):
        normalized = _normalize_spoken(word)
        if normalized:
            end_words_raw.append(normalized)
    if not end_words_raw:
        return False

    multi_word_phrases = [phrase for phrase in end_words_raw if " " in phrase]
    single_words = {word for word in end_words_raw if " " not in word}

    text_lower = _normalize_spoken(strip_leading_speaker_label(text))
    if not text_lower:
        return False

    has_stop_word = False
    remaining_text = text_lower
    for phrase in multi_word_phrases:
        if phrase in remaining_text:
            has_stop_word = True
            # Spaces keep the words on either side from gluing together.
            remaining_text = remaining_text.replace(phrase, " ")

    words = remaining_text.split()
    profile_name_lower = _normalize_spoken(agent_name or "")
    if profile_name_lower:
        words = [word for word in words if word != profile_name_lower]

    if any(word in single_words for word in words):
        has_stop_word = True
    if not has_stop_word:
        return False

    non_stop_words = [word for word in words if word not in single_words]
    return len(non_stop_words) <= 1
