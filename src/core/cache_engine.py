"""Core cache engine with memory-aware eviction."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..monitoring.memory_monitor import MemoryPressureMonitor
from .errors import CacheError


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    vector: List[float]
    hit_count: int
    last_access: datetime
    ttl: int
    size_bytes: int


class VectorCacheEngine:
    """Memory-aware vector cache engine."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._memory_monitor = MemoryPressureMonitor()
        self.logger = logging.getLogger(__name__)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_evictions": 0
        }
        
    async def start(self) -> None:
        """Start cache engine and memory monitoring."""
        asyncio.create_task(self._memory_monitor.start_monitoring())
        self.logger.info("Vector cache engine started")
        
    async def stop(self) -> None:
        """Stop cache engine."""
        self._memory_monitor.stop_monitoring()
        self.logger.info("Vector cache engine stopped")
        
    async def get(self, key: str) -> Optional[List[float]]:
        """Get vector from cache."""
        if key in self._cache:
            entry = self._cache[key]
            entry.hit_count += 1
            entry.last_access = datetime.utcnow()
            self._stats["hits"] += 1
            return entry.vector
            
        self._stats["misses"] += 1
        return None
        
    async def put(self, key: str, vector: List[float], ttl: int = 3600) -> None:
        """Put vector in cache with memory pressure awareness."""
        size_bytes = len(vector) * 4  # Rough float size estimation
        
        # Check memory pressure before adding
        if self._memory_monitor.should_aggressive_evict():
            await self._memory_pressure_eviction()
            
        # Regular size-based eviction
        if len(self._cache) >= self.max_size:
            await self._evict_lru()
            
        entry = CacheEntry(
            key=key,
            vector=vector,
            hit_count=0,
            last_access=datetime.utcnow(),
            ttl=ttl,
            size_bytes=size_bytes
        )
        
        self._cache[key] = entry
        
    async def _memory_pressure_eviction(self) -> None:
        """Aggressive eviction during memory pressure."""
        # Remove 25% of cache during high memory pressure
        target_removals = max(1, len(self._cache) // 4)
        
        # Sort by hit count (ascending) and last access (ascending)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].hit_count, x[1].last_access)
        )
        
        for key, _ in sorted_entries[:target_removals]:
            del self._cache[key]
            self._stats["memory_evictions"] += 1
            
        self.logger.info(f"Memory pressure eviction: removed {target_removals} entries")
        
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
            
        lru_key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].last_access)
        del self._cache[lru_key]
        self._stats["evictions"] += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        total_requests = self._stats["hits"] + self._stats["misses"]
        if total_requests > 0:
            hit_rate = self._stats["hits"] / total_requests
            
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": hit_rate,
            "memory_pressure": self._memory_monitor.get_current_pressure()
        }