"""Dependency-free caches for long-running reward workers."""

from collections import OrderedDict
from typing import Callable, Generic, Optional, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Small dependency-free LRU cache for long-running reward workers."""

    def __init__(self, capacity: int = 4096):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> Optional[V]:
        if key not in self._data:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def get_or_put(self, key: K, factory: Callable[[], V]) -> V:
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.put(key, value)
        return value

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }
