"""Redis connection health validation utilities."""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError


class ConnectionHealth(Enum):
    """Connection health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Health check result."""
    status: ConnectionHealth
    latency_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = None


class ConnectionValidator:
    """Validates Redis connection health with latency checks."""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    async def validate_connection(self, client: redis.Redis) -> HealthCheck:
        """Validate connection with ping and latency measurement."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test basic connectivity
            await asyncio.wait_for(client.ping(), timeout=self.timeout)
            
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Determine health based on latency
            if latency_ms < 50:
                status = ConnectionHealth.HEALTHY
            elif latency_ms < 200:
                status = ConnectionHealth.DEGRADED
            else:
                status = ConnectionHealth.UNHEALTHY
            
            return HealthCheck(
                status=status,
                latency_ms=latency_ms,
                details={"ping_success": True}
            )
            
        except TimeoutError:
            return HealthCheck(
                status=ConnectionHealth.UNHEALTHY,
                latency_ms=float('inf'),
                error="Connection timeout"
            )
        except ConnectionError as e:
            return HealthCheck(
                status=ConnectionHealth.UNHEALTHY,
                latency_ms=float('inf'),
                error=f"Connection failed: {str(e)}"
            )
        except RedisError as e:
            return HealthCheck(
                status=ConnectionHealth.DEGRADED,
                latency_ms=0.0,
                error=f"Redis error: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in health check: {e}")
            return HealthCheck(
                status=ConnectionHealth.UNHEALTHY,
                latency_ms=0.0,
                error=f"Unexpected error: {str(e)}"
            )
