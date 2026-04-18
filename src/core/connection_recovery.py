"""Connection recovery mechanism for Redis cache engine."""

import asyncio
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .errors import CacheConnectionError, CacheTimeoutError


@dataclass
class RecoveryConfig:
    """Configuration for connection recovery."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    timeout_threshold: float = 5.0
    recovery_window: timedelta = timedelta(minutes=5)


class ConnectionRecoveryManager:
    """Manages connection recovery with exponential backoff and circuit breaker pattern."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._failure_count = 0
        self._last_failure: Optional[datetime] = None
        self._is_recovering = False
    
    async def execute_with_recovery(
        self,
        operation: Callable[[], Any],
        operation_name: str = "redis_operation"
    ) -> Any:
        """Execute operation with automatic recovery on timeout/connection errors."""
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if self._is_recovering:
                    await self._wait_for_recovery()
                
                result = await asyncio.wait_for(
                    operation(),
                    timeout=self.config.timeout_threshold
                )
                
                # Reset failure state on success
                self._reset_failure_state()
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout in {operation_name} (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )
                await self._handle_failure(attempt, "timeout")
                
            except Exception as e:
                self.logger.error(
                    f"Connection error in {operation_name}: {e} (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )
                await self._handle_failure(attempt, "connection_error")
        
        raise CacheConnectionError(f"Failed to execute {operation_name} after {self.config.max_retries} retries")
    
    async def _handle_failure(self, attempt: int, failure_type: str) -> None:
        """Handle operation failure with backoff and recovery logic."""
        self._failure_count += 1
        self._last_failure = datetime.now()
        
        if attempt < self.config.max_retries:
            delay = min(
                self.config.base_delay * (self.config.backoff_multiplier ** attempt),
                self.config.max_delay
            )
            
            self.logger.info(f"Retrying after {delay}s delay (failure type: {failure_type})")
            await asyncio.sleep(delay)
    
    async def _wait_for_recovery(self) -> None:
        """Wait for recovery window to pass."""
        if not self._last_failure:
            return
        
        time_since_failure = datetime.now() - self._last_failure
        if time_since_failure < self.config.recovery_window:
            wait_time = (self.config.recovery_window - time_since_failure).total_seconds()
            self.logger.info(f"Waiting {wait_time:.1f}s for recovery window")
            await asyncio.sleep(wait_time)
        
        self._is_recovering = False
    
    def _reset_failure_state(self) -> None:
        """Reset failure tracking state after successful operation."""
        if self._failure_count > 0:
            self.logger.info("Connection recovered successfully")
        
        self._failure_count = 0
        self._last_failure = None
        self._is_recovering = False
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is considered healthy."""
        if not self._last_failure:
            return True
        
        time_since_failure = datetime.now() - self._last_failure
        return time_since_failure > self.config.recovery_window
    
    def get_health_metrics(self) -> dict[str, Any]:
        """Get current health metrics for monitoring."""
        return {
            "failure_count": self._failure_count,
            "last_failure": self._last_failure.isoformat() if self._last_failure else None,
            "is_recovering": self._is_recovering,
            "is_healthy": self.is_healthy
        }