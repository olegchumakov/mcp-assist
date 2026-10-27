"""Signed speaker tag from the Wyoming Speaker ID add-on.

The add-on prepends one tag to the transcript. This module verifies it and
strips it. It does not talk to the add-on.

Tag:
  ⟦spk1;n=<name>;c=<conf>;t=<unix_ts>;r=<nonce>;s=<sig>⟧ <transcript>

HMAC-SHA256 message (UTF-8), key = shared secret:
  spk1|<n exactly as written>|<c>|<t>|<r>|<transcript after the tag and one space>

The canonical vector (also in SPEC.md and the README) is the add-on's:

  secret:     test-secret
  name:       Anna
  n:          Anna
  c:          0.73
  t:          1700000000
  r:          0123456789abcdef
  transcript: turn on the light
  message:    spk1|Anna|0.73|1700000000|0123456789abcdef|turn on the light
  sig:        4e3854eaed0f699014eead3a73f1a358618a699bb372d76d6715658adad72386

``n``, ``c``, ``t`` and ``r`` are copied into the HMAC message as literal tag
strings. ``n`` is not decoded or re-encoded first. Cyrillic ``Анна`` is
``%D0%90%D0%BD%D0%BD%D0%B0``. An unknown voice is an empty ``n``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from urllib.parse import unquote

TAG_OPEN = "\u27e6"
TAG_CLOSE = "\u27e7"
TAG_PREFIX = "spk1;"
MAX_SKEW_SECONDS = 120
NONCE_TTL_SECONDS = 600

_MARKER_RE = re.compile(TAG_OPEN + "[^" + TAG_CLOSE + "]*" + TAG_CLOSE + " ?")
_NONCE_RE = re.compile(r"^[0-9a-fA-F]{16}$")
_SIG_RE = re.compile(r"^[0-9a-f]{64}$")

# Frozen contract vectors from the Speaker ID add-on (SPEC.md).
# Tests compare against these literals and pass now=1700000000 explicitly.
HMAC_TEST_VECTOR = {
    "secret": "test-secret",
    "name": "Anna",
    "n": "Anna",
    "c": "0.73",
    "t": "1700000000",
    "r": "0123456789abcdef",
    "transcript": "turn on the light",
    "message": "spk1|Anna|0.73|1700000000|0123456789abcdef|turn on the light",
    "sig": "4e3854eaed0f699014eead3a73f1a358618a699bb372d76d6715658adad72386",
}
HMAC_CYRILLIC_VECTOR = {
    "secret": "test-secret",
    "name": "Анна",
    "n": "%D0%90%D0%BD%D0%BD%D0%B0",
    "c": "0.81",
    "t": "1700000000",
    "r": "0123456789abcdef",
    "transcript": "включи свет",
    "message": "spk1|%D0%90%D0%BD%D0%BD%D0%B0|0.81|1700000000|0123456789abcdef|включи свет",
    "sig": "a3a00a8802e8b162bf01114baa3debc0d1b8da4df5c49382d0f0e70a90ca5318",
}
HMAC_UNKNOWN_VECTOR = {
    "secret": "test-secret",
    "name": "",
    "n": "",
    "c": "0.20",
    "t": "1700000000",
    "r": "0123456789abcdef",
    "transcript": "hello",
    "message": "spk1||0.20|1700000000|0123456789abcdef|hello",
    "sig": "cfbddbb902625d1c94fa99e6bfcc2e298f624f0ab858b76fe207d0ed85628d45",
}
VECTOR_NOW = 1700000000


@dataclass(frozen=True)
class TagVerdict:
    """Result of looking at one utterance. ``clean_text`` never contains a tag."""

    status: str  # absent, valid, invalid
    reason: str
    person: str
    confidence: float
    clean_text: str


class NonceCache:
    """Nonces accepted with a valid signature, remembered for ten minutes."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._seen)

    def seen_or_store(self, nonce: str, now: float) -> bool:
        """Return True if this nonce was already accepted within the TTL."""
        self._purge(now)
        previous = self._seen.get(nonce)
        if previous is not None and now - previous <= NONCE_TTL_SECONDS:
            return True
        self._seen[nonce] = now
        return False

    def _purge(self, now: float) -> None:
        expired = [key for key, seen_at in self._seen.items() if now - seen_at > NONCE_TTL_SECONDS]
        for key in expired:
            del self._seen[key]


def strip_speaker_markers(text: str) -> str:
    """Remove every ⟦...⟧ span and the single space that follows one."""
    if not text or TAG_OPEN not in text:
        return text
    return _MARKER_RE.sub("", text)


def _split_leading(text: str) -> tuple[dict[str, str] | None, str]:
    """Split a leading tag. Transcript is the text after the tag and one space."""
    if not text.startswith(TAG_OPEN):
        return None, text
    end = text.find(TAG_CLOSE)
    if end < 0:
        return None, text
    body = text[1:end]
    rest = text[end + 1 :]
    if rest.startswith(" "):
        rest = rest[1:]
    if not body.startswith(TAG_PREFIX):
        return None, rest
    fields: dict[str, str] = {}
    for part in body[len(TAG_PREFIX) :].split(";"):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields, rest


def _hmac(secret: str, fields: dict[str, str], transcript: str) -> str:
    message = "spk1|{n}|{c}|{t}|{r}|{transcript}".format(
        n=fields.get("n", ""),
        c=fields.get("c", ""),
        t=fields.get("t", ""),
        r=fields.get("r", ""),
        transcript=transcript,
    )
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _signatures_match(expected: str, presented: str) -> bool:
    if not _SIG_RE.fullmatch(presented):
        # Still compare something so a short signature is not an early return
        # with a different shape than a wrong full signature.
        hmac.compare_digest(expected, expected)
        return False
    return hmac.compare_digest(expected, presented)


def verify_speaker_tag(
    text: str,
    secret: str,
    now: float,
    nonces: NonceCache | None = None,
) -> TagVerdict:
    """Verify a leading tag. The returned text has every ⟦...⟧ span removed.

    An empty secret cannot verify a tag. Callers that have access control
    switched off must not call this: leaving the text untouched is their job.
    """
    raw = text or ""
    fields, transcript = _split_leading(raw)
    if fields is None:
        return TagVerdict("absent", "", "", 0.0, strip_speaker_markers(raw))

    clean = strip_speaker_markers(transcript)
    if not secret:
        return TagVerdict("invalid", "no_secret", "", 0.0, clean)

    required = ("n", "c", "t", "r", "s")
    if any(key not in fields for key in required):
        return TagVerdict("invalid", "malformed", "", 0.0, clean)
    if not _NONCE_RE.fullmatch(fields["r"]):
        return TagVerdict("invalid", "malformed", "", 0.0, clean)

    expected = _hmac(secret, fields, transcript)
    if not _signatures_match(expected, fields["s"]):
        return TagVerdict("invalid", "bad_sig", "", 0.0, clean)

    try:
        created = int(fields["t"])
    except (TypeError, ValueError):
        return TagVerdict("invalid", "malformed", "", 0.0, clean)

    # Freshness uses the caller's clock. Tests freeze ``now``; a stale tag
    # does not consume the nonce.
    if abs(float(now) - created) > MAX_SKEW_SECONDS:
        return TagVerdict("invalid", "stale", "", 0.0, clean)

    cache = nonces if nonces is not None else NonceCache()
    if cache.seen_or_store(fields["r"], now):
        return TagVerdict("invalid", "replay", "", 0.0, clean)

    try:
        confidence = float(fields["c"])
    except (TypeError, ValueError):
        confidence = 0.0
    person = unquote(fields["n"], encoding="utf-8", errors="replace").strip()
    return TagVerdict("valid", "", person, confidence, clean)
