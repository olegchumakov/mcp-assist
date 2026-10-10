"""Local BERT embeddings for memory retrieval.

Home Assistant OS is Alpine (musl). ONNX Runtime publishes manylinux wheels
only, so it does not install there. This module runs a small BERT with
NumPy plus the Hugging Face ``tokenizers`` and ``safetensors`` packages,
which ship musllinux and manylinux wheels for x86_64 and aarch64.

The default model is ``sergeyzh/rubert-tiny-turbo`` (about 29M parameters,
312 dimensions, Russian sentence embeddings). ``intfloat/multilingual-e5-small``
is optional and much larger.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

EmbeddingRole = Literal["query", "passage"]

# File sizes are the Hugging Face resolve sizes. A mismatch means the
# download was truncated or upstream replaced the file, so we fetch it again.
_RUBERT_FILES = {
    "config.json": None,
    "tokenizer.json": 2_413_691,
    "model.safetensors": 116_781_184,
}
_E5_FILES = {
    "config.json": None,
    "tokenizer.json": 17_082_730,
    "model.safetensors": 470_641_600,
}


@dataclass(frozen=True)
class LocalModelSpec:
    """One downloadable local embedding model."""

    model_id: str
    choice_key: str
    repo: str
    pooling: Literal["cls", "mean"]
    query_prefix: str
    passage_prefix: str
    dim: int
    max_length: int
    download_mb: int
    ram_mb: int
    files: dict[str, int | None]
    summary: str


LOCAL_MODELS: dict[str, LocalModelSpec] = {
    "sergeyzh/rubert-tiny-turbo": LocalModelSpec(
        model_id="sergeyzh/rubert-tiny-turbo",
        choice_key="rubert-tiny-turbo",
        repo="sergeyzh/rubert-tiny-turbo",
        pooling="cls",
        query_prefix="",
        passage_prefix="",
        dim=312,
        max_length=64,
        download_mb=118,
        ram_mb=180,
        files=_RUBERT_FILES,
        summary=(
            "Russian sentence model, 3 layers, 312 dimensions, "
            "~118 MB download, ~180 MB RAM"
        ),
    ),
    "intfloat/multilingual-e5-small": LocalModelSpec(
        model_id="intfloat/multilingual-e5-small",
        choice_key="multilingual-e5-small",
        repo="intfloat/multilingual-e5-small",
        pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
        dim=384,
        max_length=64,
        download_mb=490,
        ram_mb=700,
        files=_E5_FILES,
        summary=(
            "Multilingual model, 12 layers, 384 dimensions, "
            "~490 MB download, ~700 MB RAM, uses query:/passage: prefixes"
        ),
    ),
}

_CHOICE_TO_ID = {spec.choice_key: spec.model_id for spec in LOCAL_MODELS.values()}


def canonical_local_model(value: str | None) -> str:
    """Map a config value or choice key to a known Hugging Face model id."""
    if not value:
        return "sergeyzh/rubert-tiny-turbo"
    text = str(value).strip()
    if text in LOCAL_MODELS:
        return text
    mapped = _CHOICE_TO_ID.get(text)
    if mapped:
        return mapped
    return "sergeyzh/rubert-tiny-turbo"


def local_model_spec(model_id: str | None) -> LocalModelSpec:
    """Return the spec for a model id, falling back to rubert-tiny-turbo."""
    return LOCAL_MODELS[canonical_local_model(model_id)]


def apply_prefix(text: str, spec: LocalModelSpec, role: EmbeddingRole) -> str:
    """Add the model-specific query or passage prefix once."""
    prefix = spec.query_prefix if role == "query" else spec.passage_prefix
    cleaned = " ".join(text.split()).strip()
    if not prefix or cleaned.startswith(prefix):
        return cleaned
    return f"{prefix}{cleaned}"


def huggingface_file_url(repo: str, filename: str) -> str:
    """Direct download URL for one model file."""
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def model_is_downloaded(directory: Path, spec: LocalModelSpec) -> bool:
    """True when every required file is present and, when known, the right size."""
    for name, expected_size in spec.files.items():
        path = directory / name
        if not path.is_file():
            return False
        actual = path.stat().st_size
        if actual <= 0:
            return False
        if expected_size is not None and actual != expected_size:
            return False
    return True


def _limit_blas_threads() -> None:
    """Keep a tiny BERT from grabbing every core on a small CPU."""
    try:
        import ctypes
    except Exception:
        return
    for library_name in ("libopenblas.so.0", "libopenblas.so"):
        try:
            library = ctypes.CDLL(library_name)
        except OSError:
            continue
        setter = getattr(library, "openblas_set_num_threads", None)
        if setter is None:
            continue
        try:
            setter(2)
        except Exception:
            return
        return


class BertTextEmbedder:
    """Loaded BERT encoder. ``embed`` is synchronous and must run off the event loop."""

    def __init__(
        self,
        weights: dict[str, Any],
        tokenizer: Any,
        *,
        spec: LocalModelSpec,
        config: dict[str, Any],
    ) -> None:
        self._weights = weights
        self._tokenizer = tokenizer
        self.spec = spec
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.intermediate_size = int(config["intermediate_size"])
        self.layer_norm_eps = float(config.get("layer_norm_eps", 1e-12))
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.pooling = spec.pooling
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"{spec.model_id} hidden size {self.hidden_size} is not divisible "
                f"by {self.num_heads} heads"
            )
        self._head_dim = self.hidden_size // self.num_heads

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed already-prefixed texts. Returns L2-normalized vectors."""
        import numpy as np

        cleaned = [text for text in texts if text and text.strip()]
        if len(cleaned) != len(texts):
            raise ValueError("embedding texts must be non-empty")
        if not cleaned:
            return []

        encoded = [self._encode(text) for text in cleaned]
        max_len = max(len(ids) for ids in encoded)
        batch = len(encoded)
        input_ids = np.full((batch, max_len), self.pad_token_id, dtype=np.int64)
        mask = np.zeros((batch, max_len), dtype=np.float32)
        for row, ids in enumerate(encoded):
            length = len(ids)
            input_ids[row, :length] = ids
            mask[row, :length] = 1.0

        hidden = self._forward(input_ids, mask)
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            masked = hidden * mask[:, :, None]
            denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
            pooled = masked.sum(axis=1) / denom
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)
        return pooled.astype(np.float32).tolist()

    def _encode(self, text: str) -> list[int]:
        ids = list(self._tokenizer.encode(text).ids)
        if len(ids) > self.spec.max_length:
            ids = ids[: self.spec.max_length]
        if not ids:
            raise ValueError("tokenizer produced no tokens")
        return ids

    def _forward(self, input_ids: Any, mask: Any) -> Any:
        import numpy as np

        word = self._weights["embeddings.word_embeddings.weight"][input_ids]
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64)
        position = self._weights["embeddings.position_embeddings.weight"][position_ids]
        token_type = self._weights["embeddings.token_type_embeddings.weight"][0]
        hidden = self._layer_norm(
            word + position + token_type,
            self._weights["embeddings.LayerNorm.weight"],
            self._weights["embeddings.LayerNorm.bias"],
        )

        for layer in range(self.num_layers):
            hidden = self._encoder_layer(hidden, mask, layer)
        return hidden

    def _encoder_layer(self, hidden: Any, mask: Any, layer: int) -> Any:
        import numpy as np

        prefix = f"encoder.layer.{layer}."
        query = self._linear(
            hidden,
            self._weights[prefix + "attention.self.query.weight"],
            self._weights[prefix + "attention.self.query.bias"],
        )
        key = self._linear(
            hidden,
            self._weights[prefix + "attention.self.key.weight"],
            self._weights[prefix + "attention.self.key.bias"],
        )
        value = self._linear(
            hidden,
            self._weights[prefix + "attention.self.value.weight"],
            self._weights[prefix + "attention.self.value.bias"],
        )
        batch, seq_len, _ = query.shape
        heads = self.num_heads
        head_dim = self._head_dim

        def split_heads(tensor: Any) -> Any:
            return tensor.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(head_dim))
        additive = (1.0 - mask)[:, None, None, :] * np.float32(-1.0e4)
        scores = scores + additive
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        context = np.matmul(probs, value)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_size)
        attention_output = self._linear(
            context,
            self._weights[prefix + "attention.output.dense.weight"],
            self._weights[prefix + "attention.output.dense.bias"],
        )
        hidden = self._layer_norm(
            attention_output + hidden,
            self._weights[prefix + "attention.output.LayerNorm.weight"],
            self._weights[prefix + "attention.output.LayerNorm.bias"],
        )
        intermediate = self._gelu(
            self._linear(
                hidden,
                self._weights[prefix + "intermediate.dense.weight"],
                self._weights[prefix + "intermediate.dense.bias"],
            )
        )
        output = self._linear(
            intermediate,
            self._weights[prefix + "output.dense.weight"],
            self._weights[prefix + "output.dense.bias"],
        )
        return self._layer_norm(
            output + hidden,
            self._weights[prefix + "output.LayerNorm.weight"],
            self._weights[prefix + "output.LayerNorm.bias"],
        )

    def _layer_norm(self, tensor: Any, weight: Any, bias: Any) -> Any:
        import numpy as np

        mean = tensor.mean(axis=-1, keepdims=True)
        variance = ((tensor - mean) ** 2).mean(axis=-1, keepdims=True)
        return (tensor - mean) / np.sqrt(variance + np.float32(self.layer_norm_eps)) * weight + bias

    @staticmethod
    def _linear(tensor: Any, weight: Any, bias: Any) -> Any:
        return tensor @ weight.T + bias

    @staticmethod
    def _gelu(tensor: Any) -> Any:
        """BERT erf GELU. NumPy 2 removed ``np.erf``; ``math.erf`` matches PyTorch."""
        import math

        import numpy as np

        flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
        erf_values = np.empty(flat.shape, dtype=np.float32)
        scale = np.float32(1.0 / math.sqrt(2.0))
        for index, value in enumerate(flat):
            erf_values[index] = math.erf(float(value) * float(scale))
        return 0.5 * tensor * (1.0 + erf_values.reshape(tensor.shape))


def load_embedder(directory: Path, spec: LocalModelSpec) -> BertTextEmbedder:
    """Load tokenizer and weights from a downloaded model directory."""
    import numpy as np
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    _limit_blas_threads()
    config_path = directory / "config.json"
    tokenizer_path = directory / "tokenizer.json"
    weights_path = directory / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=spec.max_length)
    raw_weights = load_file(str(weights_path))
    weights = {
        key: np.ascontiguousarray(value, dtype=np.float32)
        for key, value in raw_weights.items()
    }
    required = (
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise RuntimeError(
            f"{spec.model_id} is missing weights: {', '.join(missing)}"
        )
    return BertTextEmbedder(weights, tokenizer, spec=spec, config=config)
