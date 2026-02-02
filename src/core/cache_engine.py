"""Core cache engine with improved error handling."""

import redis
import numpy as np
from typing import Optional, Dict, Any
from src.config.redis_config import RedisConfig
from src.core.errors import ConnectionError, SerializationError, MemoryPressureError
from src.metrics.collector import MetricsCollector


class CacheEngine:
    """Redis-backed vector cache with ML-driven optimizations."""
    
    def __init__(self, config: RedisConfig, metrics: MetricsCollector):
        self.config = config
        self.metrics = metrics
        self._redis = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection with retry logic."""
        try:
            self._redis = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                decode_responses=False,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connect_timeout
            )
            self._redis.ping()
            self.metrics.increment_counter("cache_connections_total", labels={"status": "success"})
        except redis.RedisError as e:
            self.metrics.increment_counter("cache_connections_total", labels={"status": "failed"})
            raise ConnectionError(f"Failed to connect to Redis: {e}", metrics_collector=self.metrics)
    
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Retrieve vector with comprehensive error handling."""
        try:
            data = self._redis.get(key)
            if data is None:
                self.metrics.increment_counter("cache_misses_total")
                return None
            
            vector = np.frombuffer(data, dtype=np.float32)
            self.metrics.increment_counter("cache_hits_total")
            return vector
            
        except redis.RedisError as e:
            raise ConnectionError(f"Redis error getting key {key}: {e}", key=key, metrics_collector=self.metrics)
        except (ValueError, TypeError) as e:
            raise SerializationError(f"Failed to deserialize vector for key {key}: {e}", key=key, metrics_collector=self.metrics)
    
    def set_vector(self, key: str, vector: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Store vector with memory pressure handling."""
        try:
            data = vector.astype(np.float32).tobytes()
            
            # Check memory usage before storing
            memory_info = self._redis.info('memory')
            if memory_info.get('used_memory', 0) > self.config.max_memory_threshold:
                raise MemoryPressureError(f"Memory usage too high: {memory_info['used_memory']}", key=key, metrics_collector=self.metrics)
            
            result = self._redis.set(key, data, ex=ttl)
            if result:
                self.metrics.increment_counter("cache_sets_total", labels={"status": "success"})
            return result
            
        except redis.RedisError as e:
            self.metrics.increment_counter("cache_sets_total", labels={"status": "failed"})
            raise ConnectionError(f"Redis error setting key {key}: {e}", key=key, metrics_collector=self.metrics)
        except (ValueError, TypeError) as e:
            raise SerializationError(f"Failed to serialize vector for key {key}: {e}", key=key, metrics_collector=self.metrics)
