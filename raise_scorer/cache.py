"""Backward-compatible cache imports.

New code may import from :mod:`raise_scorer.backends`.
"""

from .backends.cache import LRUCache

__all__ = ["LRUCache"]
