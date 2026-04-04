"""Core cache engine with ML-driven eviction and health monitoring."""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.exceptions import RedisError

from .connection_pool import ConnectionPool
from .connection_validator import ConnectionValidator, ConnectionHealth
from .eviction_selector import EvictionSelector
from .circuit_breaker import CircuitBreaker
from .errors import CacheError, ConnectionError as CacheConnectionError
from ..ml.hit_predictor import HitPredictor
from ..monitoring.metrics_collector import MetricsCollector


class CacheEngine:
    """High-performance cache engine with ML-driven optimizations."""
    
    def __init__(
        self,
        connection_pool: ConnectionPool,
        eviction_selector: EvictionSelector,
        hit_predictor: HitPredictor,
        metrics_collector: MetricsCollector
    ):
        self.pool = connection_pool
        self.eviction_selector = eviction_selector
        self.hit_predictor = hit_predictor
        self.metrics = metrics_collector
        self.validator = ConnectionValidator(timeout=5.0)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=RedisError
        )
        self.logger = logging.getLogger(__name__)
        self._health_check_interval = 60  # seconds
        self._last_health_check = datetime.now() - timedelta(minutes=5)
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value with health validation and circuit breaker protection."""
        await self._ensure_healthy_connection()
        
        async with self.circuit_breaker:
            try:
                client = await self.pool.get_connection()
                value = await client.get(key)
                
                # Record metrics
                if value:
                    self.metrics.record_cache_hit(key)
                else:
                    self.metrics.record_cache_miss(key)
                
                return value
                
            except RedisError as e:
                self.logger.error(f"Cache get failed for key {key}: {e}")
                raise CacheError(f"Failed to get key {key}") from e
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value with connection health validation."""
        await self._ensure_healthy_connection()
        
        async with self.circuit_breaker:
            try:
                client = await self.pool.get_connection()
                
                if ttl:
                    success = await client.setex(key, ttl, value)
                else:
                    success = await client.set(key, value)
                
                if success:
                    self.metrics.record_cache_write(key, len(value))
                
                return bool(success)
                
            except RedisError as e:
                self.logger.error(f"Cache set failed for key {key}: {e}")
                raise CacheError(f"Failed to set key {key}") from e
    
    async def _ensure_healthy_connection(self) -> None:
        """Ensure connection is healthy before operations."""
        now = datetime.now()
        if (now - self._last_health_check).seconds < self._health_check_interval:
            return
        
        try:
            client = await self.pool.get_connection()
            health = await self.validator.validate_connection(client)
            
            if health.status == ConnectionHealth.UNHEALTHY:
                self.logger.error(f"Unhealthy connection detected: {health.error}")
                raise CacheConnectionError(f"Connection unhealthy: {health.error}")
            elif health.status == ConnectionHealth.DEGRADED:
                self.logger.warning(f"Degraded connection: latency={health.latency_ms:.2f}ms")
            
            self.metrics.record_connection_health(health.status.value, health.latency_ms)
            self._last_health_check = now
            
        except Exception as e:
            self.logger.error(f"Connection health check failed: {e}")
            raise CacheConnectionError("Connection health validation failed") from e
