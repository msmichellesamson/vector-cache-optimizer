"""Redis configuration with circuit breaker integration."""

import os
import redis
import logging
from typing import Optional
from dataclasses import dataclass

from ..core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    max_connections: int = 50

class RedisManager:
    """Redis client with circuit breaker protection."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2
            )
        )
        self._client = self._create_client()
        
    def _create_client(self) -> redis.Redis:
        """Create Redis client with connection pooling."""
        return redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            retry_on_timeout=self.config.retry_on_timeout,
            max_connections=self.config.max_connections,
            decode_responses=True
        )
        
    def get(self, key: str) -> Optional[str]:
        """Get value with circuit breaker protection."""
        try:
            return self.circuit_breaker.call(self._client.get, key)
        except ConnectionError:
            logger.error(f"Redis unavailable - circuit breaker open for key: {key}")
            return None
            
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value with circuit breaker protection."""
        try:
            return self.circuit_breaker.call(self._client.set, key, value, ex=ex)
        except ConnectionError:
            logger.error(f"Redis unavailable - circuit breaker open for key: {key}")
            return False
            
    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        return self.circuit_breaker.is_available