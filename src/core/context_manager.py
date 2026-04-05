"""Async context manager for cache operations with automatic cleanup."""

import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager

from .cache_engine import CacheEngine
from .errors import CacheError
from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class CacheContext:
    """Context manager for safe cache operations."""
    
    def __init__(self, cache_engine: CacheEngine):
        self.cache_engine = cache_engine
        self._transaction_id: Optional[str] = None
        self._operations: list[str] = []
    
    async def __aenter__(self) -> 'CacheContext':
        """Enter context with transaction tracking."""
        self._transaction_id = f"txn_{asyncio.current_task().get_name()}"
        logger.debug(f"Starting cache context: {self._transaction_id}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context with cleanup and error handling."""
        if exc_type:
            logger.error(
                f"Cache context {self._transaction_id} failed: {exc_val}",
                extra={"operations": self._operations, "error_type": exc_type.__name__}
            )
            await self._rollback()
        else:
            logger.debug(
                f"Cache context {self._transaction_id} completed",
                extra={"operations_count": len(self._operations)}
            )
        
        await self._cleanup()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value with operation tracking."""
        self._operations.append(f"GET:{key}")
        try:
            return await self.cache_engine.get(key, default)
        except Exception as e:
            logger.warning(f"Context GET failed for {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with operation tracking."""
        self._operations.append(f"SET:{key}")
        return await self.cache_engine.set(key, value, ttl)
    
    async def _rollback(self):
        """Attempt to rollback operations on failure."""
        logger.info(f"Rolling back {len(self._operations)} operations")
        # In a real implementation, we'd store reversible operations
        # For now, just log the attempted rollback
    
    async def _cleanup(self):
        """Clean up context resources."""
        self._operations.clear()
        self._transaction_id = None


@asynccontextmanager
async def cache_context(cache_engine: CacheEngine):
    """Async context manager factory for cache operations."""
    context = CacheContext(cache_engine)
    async with context:
        yield context