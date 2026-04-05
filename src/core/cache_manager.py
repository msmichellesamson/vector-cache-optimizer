from typing import Optional, Dict, Any, AsyncContextManager
import asyncio
import logging
from contextlib import asynccontextmanager

from .cache_engine import CacheEngine
from .connection_pool import ConnectionPool
from .errors import CacheError
from ..monitoring.logger import get_logger

logger = get_logger(__name__)

class CacheManager:
    """Main cache manager with async context management for proper cleanup."""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self._connection_pool: Optional[ConnectionPool] = None
        self._cache_engine: Optional[CacheEngine] = None
        self._shutdown_event = asyncio.Event()
    
    async def __aenter__(self) -> 'CacheManager':
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.shutdown()
    
    async def initialize(self) -> None:
        """Initialize connection pool and cache engine."""
        try:
            logger.info("Initializing cache manager")
            self._connection_pool = ConnectionPool(
                redis_url=self.redis_url,
                max_connections=self.max_connections
            )
            await self._connection_pool.initialize()
            
            self._cache_engine = CacheEngine(self._connection_pool)
            await self._cache_engine.initialize()
            
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            await self.shutdown()  # Cleanup on failure
            raise CacheError(f"Cache manager initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        if self._shutdown_event.is_set():
            return  # Already shutting down
        
        logger.info("Starting cache manager shutdown")
        self._shutdown_event.set()
        
        # Shutdown in reverse order
        if self._cache_engine:
            try:
                await asyncio.wait_for(self._cache_engine.shutdown(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Cache engine shutdown timed out")
            except Exception as e:
                logger.error(f"Error during cache engine shutdown: {e}")
        
        if self._connection_pool:
            try:
                await asyncio.wait_for(self._connection_pool.close(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Connection pool shutdown timed out")
            except Exception as e:
                logger.error(f"Error during connection pool shutdown: {e}")
        
        logger.info("Cache manager shutdown complete")
    
    @property
    def cache_engine(self) -> CacheEngine:
        """Get cache engine instance."""
        if not self._cache_engine:
            raise CacheError("Cache manager not initialized")
        return self._cache_engine
    
    def is_healthy(self) -> bool:
        """Check if cache manager is healthy."""
        return (
            self._connection_pool is not None and
            self._cache_engine is not None and
            not self._shutdown_event.is_set()
        )

@asynccontextmanager
async def create_cache_manager(redis_url: str, max_connections: int = 20) -> AsyncContextManager[CacheManager]:
    """Factory function to create cache manager with context management."""
    manager = CacheManager(redis_url, max_connections)
    async with manager:
        yield manager
