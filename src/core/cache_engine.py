"""Core cache engine with ML-driven eviction and intelligent prefetching."""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

from .errors import CacheError, ValidationError
from .circuit_breaker import CircuitBreaker
from .retry import with_retry, RetryConfig
from ..ml.predictor import CachePredictor
from ..config.redis_config import RedisConfig

logger = logging.getLogger(__name__)

# Metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')
operation_duration = Histogram('cache_operation_duration_seconds', 'Cache operation duration')
cache_size = Gauge('cache_size_bytes', 'Current cache size in bytes')

@dataclass
class CacheEntry:
    """Cache entry with metadata for ML optimization."""
    key: str
    value: Any
    embedding: Optional[np.ndarray]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int] = None
    predicted_next_access: Optional[datetime] = None

class VectorCacheEngine:
    """Intelligent cache engine with ML-driven optimization."""
    
    def __init__(self, config: RedisConfig, predictor: CachePredictor):
        self.config = config
        self.predictor = predictor
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Circuit breaker for Redis operations
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=redis.RedisError
        )
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0
        )
        
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'ml_predictions': 0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self.redis_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.database,
                password=self.config.password,
                max_connections=self.config.max_connections,
                retry_on_timeout=True,
                socket_connect_timeout=self.config.connect_timeout,
                socket_timeout=self.config.socket_timeout
            )
            
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self._ping_redis()
            logger.info("Cache engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache engine: {e}")
            raise CacheError(f"Cache initialization failed: {e}") from e
    
    @with_retry(RetryConfig(max_attempts=3), None)  # Will be updated to use circuit_breaker
    async def _ping_redis(self) -> None:
        """Test Redis connection."""
        if not self.redis_client:
            raise CacheError("Redis client not initialized")
        
        await self.redis_client.ping()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            'hit_rate': self._stats['hits'] / max(self._stats['hits'] + self._stats['misses'], 1),
            'circuit_breaker_state': self.circuit_breaker.state.name,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }