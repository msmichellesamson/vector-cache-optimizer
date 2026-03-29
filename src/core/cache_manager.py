from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import asyncio
import logging

from .cache_engine import CacheEngine
from .connection_pool import ConnectionPool
from .errors import CacheManagerError
from ..monitoring.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cache engine lifecycle with proper resource cleanup."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.connection_pool = connection_pool
        self.cache_engine: Optional[CacheEngine] = None
        self.memory_monitor: Optional[MemoryMonitor] = None
        self._shutdown_event = asyncio.Event()
        self._running = False
    
    @asynccontextmanager
    async def managed_cache(self) -> AsyncGenerator[CacheEngine, None]:
        """Async context manager for cache engine with proper cleanup."""
        try:
            await self._startup()
            yield self.cache_engine
        except Exception as e:
            logger.error(f"Cache manager error: {e}")
            raise CacheManagerError(f"Failed to manage cache: {e}") from e
        finally:
            await self._shutdown()
    
    async def _startup(self) -> None:
        """Initialize cache engine and monitoring."""
        try:
            self.cache_engine = CacheEngine(self.connection_pool)
            self.memory_monitor = MemoryMonitor()
            
            await self.cache_engine.initialize()
            await self.memory_monitor.start_monitoring()
            
            self._running = True
            logger.info("Cache manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start cache manager: {e}")
            await self._cleanup_resources()
            raise CacheManagerError(f"Startup failed: {e}") from e
    
    async def _shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        if not self._running:
            return
        
        try:
            logger.info("Shutting down cache manager...")
            self._shutdown_event.set()
            
            # Wait for ongoing operations to complete
            await asyncio.sleep(0.1)
            
            await self._cleanup_resources()
            self._running = False
            logger.info("Cache manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise CacheManagerError(f"Shutdown failed: {e}") from e
    
    async def _cleanup_resources(self) -> None:
        """Clean up all resources."""
        cleanup_tasks = []
        
        if self.memory_monitor:
            cleanup_tasks.append(self.memory_monitor.stop_monitoring())
        
        if self.cache_engine:
            cleanup_tasks.append(self.cache_engine.close())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    @property
    def is_running(self) -> bool:
        """Check if cache manager is running."""
        return self._running and not self._shutdown_event.is_set()
