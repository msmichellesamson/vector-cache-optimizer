"""Core cache engine with improved async context support."""

import asyncio
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

from .connection_pool import ConnectionPool
from .errors import CacheError, ConnectionError
from ..monitoring.logger import get_logger
from ..monitoring.hit_rate_tracker import HitRateTracker

logger = get_logger(__name__)


class CacheEngine:
    """Enhanced cache engine with context manager support."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.connection_pool = connection_pool
        self.hit_rate_tracker = HitRateTracker()
        self._stats = {"operations": 0, "errors": 0}
    
    @asynccontextmanager
    async def transaction(self):
        """Create a transaction context for batch operations."""
        from .context_manager import CacheContext
        context = CacheContext(self)
        async with context:
            yield context
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with improved error handling."""
        try:
            async with self.connection_pool.get_connection() as conn:
                self._stats["operations"] += 1
                result = await conn.get(key)
                
                if result is not None:
                    self.hit_rate_tracker.record_hit(key)
                    logger.debug(f"Cache HIT: {key}")
                    return result
                else:
                    self.hit_rate_tracker.record_miss(key)
                    logger.debug(f"Cache MISS: {key}")
                    return default
                    
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache GET error for {key}: {e}")
            if isinstance(e, (ConnectionError, asyncio.TimeoutError)):
                raise CacheError(f"Failed to get {key}: connection issue") from e
            raise CacheError(f"Failed to get {key}: {e}") from e
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with validation."""
        if not key or key.isspace():
            raise ValueError("Cache key cannot be empty or whitespace")
        
        try:
            async with self.connection_pool.get_connection() as conn:
                self._stats["operations"] += 1
                success = await conn.set(key, value, ex=ttl)
                
                if success:
                    logger.debug(f"Cache SET: {key} (ttl={ttl})")
                else:
                    logger.warning(f"Cache SET failed: {key}")
                    self._stats["errors"] += 1
                
                return success
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache SET error for {key}: {e}")
            raise CacheError(f"Failed to set {key}: {e}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "hit_rate": self.hit_rate_tracker.get_hit_rate(),
            "pool_stats": self.connection_pool.get_stats()
        }