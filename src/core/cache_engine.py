"""High-performance vector cache with ML-driven eviction policies."""
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .errors import CacheError, ConnectionError
from .connection_pool import RedisConnectionPool
from .circuit_breaker import CircuitBreaker
from .eviction_selector import EvictionSelector
from .ttl_optimizer import TTLOptimizer
from ..monitoring.logger import get_logger, set_correlation_id
from ..ml.hit_predictor import HitPredictor
from ..metrics.collector import MetricsCollector

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: float = 0.0
    avg_response_time: float = 0.0

class VectorCacheEngine:
    """Intelligent vector cache with ML-driven optimization."""
    
    def __init__(self, 
                 connection_pool: RedisConnectionPool,
                 hit_predictor: HitPredictor,
                 metrics_collector: MetricsCollector):
        self.pool = connection_pool
        self.hit_predictor = hit_predictor
        self.metrics = metrics_collector
        self.logger = get_logger('cache_engine')
        
        self.circuit_breaker = CircuitBreaker()
        self.eviction_selector = EvictionSelector()
        self.ttl_optimizer = TTLOptimizer()
        
        self.stats = CacheStats()
        self._local_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._max_local_size = 1000
    
    async def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Retrieve vector from cache with correlation tracking."""
        correlation_id = set_correlation_id()
        start_time = time.time()
        
        try:
            # Check local cache first
            if key in self._local_cache:
                vector, timestamp = self._local_cache[key]
                self.stats.hits += 1
                self.logger.info("Cache hit (local)", key=key, source="local")
                return vector
            
            # Check Redis with circuit breaker
            vector = await self.circuit_breaker.call(
                self._get_from_redis, key
            )
            
            if vector is not None:
                self.stats.hits += 1
                self._update_local_cache(key, vector)
                self.logger.info("Cache hit (redis)", key=key, source="redis")
            else:
                self.stats.misses += 1
                self.logger.info("Cache miss", key=key)
            
            return vector
            
        except Exception as e:
            self.logger.error("Cache get failed", key=key, error=str(e))
            raise CacheError(f"Failed to get vector: {e}") from e
        finally:
            duration = time.time() - start_time
            self.metrics.record_latency('cache_get', duration)
    
    async def _get_from_redis(self, key: str) -> Optional[np.ndarray]:
        """Get vector from Redis."""
        async with self.pool.get_connection() as conn:
            data = await conn.get(key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
            return None
    
    def _update_local_cache(self, key: str, vector: np.ndarray):
        """Update local cache with LRU eviction."""
        if len(self._local_cache) >= self._max_local_size:
            # Remove oldest entry
            oldest_key = next(iter(self._local_cache))
            del self._local_cache[oldest_key]
        
        self._local_cache[key] = (vector, time.time())
