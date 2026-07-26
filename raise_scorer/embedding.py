"""Backward-compatible embedding imports.

New code may import from :mod:`raise_scorer.backends`.
"""

from .backends.embedding import (
    EmbeddingBackend,
    TextEmbedder,
    default_dtype,
    get_reward_device,
    load_embedding_backend,
)

__all__ = [
    "EmbeddingBackend",
    "TextEmbedder",
    "default_dtype",
    "get_reward_device",
    "load_embedding_backend",
]
