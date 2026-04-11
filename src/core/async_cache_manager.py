"""Async context manager for cache operations with proper resource cleanup."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass

from .cache_engine import CacheEngine
from .connection_pool import ConnectionPool
from .errors import CacheError, ConnectionError

logger = logging.getLogger(__name__)

@dataclass
class CacheSession:
    """Represents an active cache session with metrics tracking."""
    session_id: str
    start_time: float
    operations: int = 0
    hits: int = 0
    misses: int = 0

class AsyncCacheManager:
    """Async context manager for cache operations with automatic resource cleanup."""
    
    def __init__(self, cache_engine: CacheEngine, connection_pool: ConnectionPool):
        self.cache_engine = cache_engine
        self.connection_pool = connection_pool
        self._active_sessions: Dict[str, CacheSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncGenerator[CacheSession, None]:
        """Create a cache session with automatic cleanup."""
        session = CacheSession(
            session_id=session_id,
            start_time=asyncio.get_event_loop().time()
        )
        
        try:
            self._active_sessions[session_id] = session
            logger.info(f"Started cache session: {session_id}")
            yield session
            
        except Exception as e:
            logger.error(f"Error in cache session {session_id}: {e}")
            raise CacheError(f"Cache session failed: {e}") from e
            
        finally:
            await self._cleanup_session(session_id)
            
    async def _cleanup_session(self, session_id: str) -> None:
        """Clean up session resources and log metrics."""
        if session_id not in self._active_sessions:
            return
            
        session = self._active_sessions.pop(session_id)
        duration = asyncio.get_event_loop().time() - session.start_time
        hit_rate = session.hits / max(session.operations, 1) * 100
        
        logger.info(
            f"Session {session_id} completed: "
            f"duration={duration:.2f}s, ops={session.operations}, "
            f"hit_rate={hit_rate:.1f}%"
        )
        
    async def __aenter__(self):
        """Initialize async context manager."""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all resources on exit."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Cleanup remaining sessions
        for session_id in list(self._active_sessions.keys()):
            await self._cleanup_session(session_id)
            
    async def _periodic_cleanup(self) -> None:
        """Periodically cleanup stale sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                current_time = asyncio.get_event_loop().time()
                
                stale_sessions = [
                    session_id for session_id, session in self._active_sessions.items()
                    if current_time - session.start_time > 3600  # 1 hour timeout
                ]
                
                for session_id in stale_sessions:
                    logger.warning(f"Cleaning up stale session: {session_id}")
                    await self._cleanup_session(session_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")