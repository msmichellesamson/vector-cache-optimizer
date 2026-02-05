"""Redis connection pool with health monitoring and graceful degradation."""
import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass
import redis.asyncio as redis
from .errors import CacheConnectionError, CacheHealthError


@dataclass
class PoolHealth:
    """Connection pool health metrics."""
    active_connections: int
    failed_connections: int
    last_health_check: float
    is_healthy: bool
    avg_response_time: float


class HealthyConnectionPool:
    """Redis connection pool with health monitoring."""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self._health = PoolHealth(0, 0, 0.0, False, 0.0)
        self._health_check_interval = 30.0  # seconds
        self._last_health_check = 0.0
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                decode_responses=True
            )
            self.client = redis.Redis(connection_pool=self.pool)
            await self._health_check()
            self.logger.info(f"Connection pool initialized with {self.max_connections} max connections")
        except Exception as e:
            raise CacheConnectionError(f"Failed to initialize connection pool: {e}")
    
    async def _health_check(self) -> None:
        """Perform health check on connection pool."""
        start_time = time.time()
        
        try:
            # Test connection with ping
            await self.client.ping()
            
            # Get pool stats
            active = len(self.pool._available_connections)
            
            response_time = time.time() - start_time
            
            self._health = PoolHealth(
                active_connections=active,
                failed_connections=0,
                last_health_check=time.time(),
                is_healthy=True,
                avg_response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health = PoolHealth(
                active_connections=0,
                failed_connections=1,
                last_health_check=time.time(),
                is_healthy=False,
                avg_response_time=0.0
            )
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client with health validation."""
        # Check if health check is needed
        if time.time() - self._last_health_check > self._health_check_interval:
            await self._health_check()
            self._last_health_check = time.time()
        
        if not self._health.is_healthy:
            raise CacheHealthError("Connection pool is unhealthy")
        
        if not self.client:
            raise CacheConnectionError("Connection pool not initialized")
        
        return self.client
    
    def get_health(self) -> PoolHealth:
        """Get current pool health status."""
        return self._health
    
    async def close(self) -> None:
        """Close connection pool gracefully."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self.logger.info("Connection pool closed")
