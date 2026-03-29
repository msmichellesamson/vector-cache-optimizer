import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .errors import ConnectionPoolError, ConnectionTimeoutError

@dataclass
class PoolConfig:
    min_connections: int = 5
    max_connections: int = 50
    connection_timeout: float = 5.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    health_check_interval: float = 30.0

class ConnectionPool:
    def __init__(self, config: PoolConfig):
        self.config = config
        self._pool: Dict[str, Any] = {}
        self._active_count = 0
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def get_connection(self, redis_url: str) -> Any:
        """Get connection with exponential backoff on failures"""
        for attempt in range(self.config.max_retries):
            try:
                async with asyncio.wait_for(
                    self._acquire_connection(redis_url),
                    timeout=self.config.connection_timeout
                ):
                    return await self._acquire_connection(redis_url)
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise ConnectionTimeoutError(f"Connection timeout after {self.config.max_retries} attempts")
                
                backoff_time = self.config.backoff_factor * (2 ** attempt)
                self.logger.warning(f"Connection attempt {attempt + 1} failed, backing off for {backoff_time}s")
                await asyncio.sleep(backoff_time)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise ConnectionPoolError(f"Failed to acquire connection: {e}")
                await asyncio.sleep(self.config.backoff_factor * (2 ** attempt))
                
    async def _acquire_connection(self, redis_url: str) -> Any:
        async with self._lock:
            if self._active_count >= self.config.max_connections:
                raise ConnectionPoolError("Connection pool exhausted")
            
            # Simulate connection creation
            self._active_count += 1
            return f"connection_{self._active_count}"
            
    async def release_connection(self, connection: Any) -> None:
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            
    async def health_check(self) -> bool:
        """Basic health check for pool status"""
        return self._active_count < self.config.max_connections