"""OpenAI-compatible embeddings helpers with no Home Assistant imports."""

from __future__ import annotations

from typing import Any


def embeddings_url(base_url: str) -> str:
    """OpenAI-compatible embeddings endpoint for a chat base URL."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


def parse_openai_embeddings(payload: Any) -> list[list[float]] | None:
    """Parse a ``/v1/embeddings`` JSON body. None means the response is unusable."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    rows = data
    if all(isinstance(item, dict) and "index" in item for item in data):
        rows = sorted(data, key=lambda item: item["index"])
    vectors: list[list[float]] = []
    for item in rows:
        if not isinstance(item, dict):
            return None
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            return None
        try:
            vectors.append([float(value) for value in embedding])
        except (TypeError, ValueError):
            return None
    return vectors
