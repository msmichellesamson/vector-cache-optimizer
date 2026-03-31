"""Health check endpoint for Kubernetes readiness probes."""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Any

from ..core.cache_engine import CacheEngine
from ..metrics.collector import MetricsCollector
from .logger import logger


@dataclass
class HealthStatus:
    """Health check status."""
    healthy: bool
    checks: Dict[str, bool]
    details: Dict[str, Any]
    timestamp: float


class HealthChecker:
    """Performs health checks for readiness probes."""
    
    def __init__(self, cache_engine: CacheEngine, metrics_collector: MetricsCollector):
        self.cache_engine = cache_engine
        self.metrics_collector = metrics_collector
        self._check_timeout = 2.0  # seconds
    
    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        start_time = time.time()
        checks = {}
        details = {}
        
        try:
            # Check Redis connectivity
            checks['redis'] = await self._check_redis_health()
            
            # Check memory usage
            memory_ok, memory_usage = await self._check_memory_health()
            checks['memory'] = memory_ok
            details['memory_usage_mb'] = memory_usage
            
            # Check hit rate
            hit_rate_ok, hit_rate = self._check_hit_rate()
            checks['hit_rate'] = hit_rate_ok
            details['hit_rate'] = hit_rate
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            checks['error'] = False
            details['error'] = str(e)
        
        healthy = all(checks.values())
        
        return HealthStatus(
            healthy=healthy,
            checks=checks,
            details=details,
            timestamp=start_time
        )
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity with timeout."""
        try:
            # Quick ping test
            await asyncio.wait_for(
                self.cache_engine._redis_client.ping(),
                timeout=self._check_timeout
            )
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    async def _check_memory_health(self) -> tuple[bool, float]:
        """Check memory usage is within acceptable limits."""
        try:
            memory_info = await self.cache_engine._redis_client.info('memory')
            used_memory = int(memory_info.get('used_memory', 0))
            max_memory = int(memory_info.get('maxmemory', 0))
            
            memory_mb = used_memory / (1024 * 1024)
            
            if max_memory > 0:
                usage_ratio = used_memory / max_memory
                memory_ok = usage_ratio < 0.9  # Alert if > 90% full
            else:
                memory_ok = memory_mb < 1024  # Alert if > 1GB when no limit
            
            return memory_ok, memory_mb
            
        except Exception as e:
            logger.warning(f"Memory health check failed: {e}")
            return False, 0.0
    
    def _check_hit_rate(self) -> tuple[bool, float]:
        """Check cache hit rate is acceptable."""
        try:
            hit_rate = self.metrics_collector.get_hit_rate()
            # Consider healthy if hit rate > 50% or insufficient data
            hit_rate_ok = hit_rate is None or hit_rate > 0.5
            return hit_rate_ok, hit_rate or 0.0
            
        except Exception as e:
            logger.warning(f"Hit rate health check failed: {e}")
            return True, 0.0  # Don't fail health check for metrics issues
