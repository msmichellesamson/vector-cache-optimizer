from enum import Enum
from typing import Optional, Callable, Any
import time
import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class FailureType(Enum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"

@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    timeout_threshold: float = 10.0
    recovery_timeout: float = 30.0
    max_recovery_timeout: float = 300.0
    backoff_multiplier: float = 2.0
    half_open_max_calls: int = 3

class CircuitBreaker:
    def __init__(self, config: CircuitConfig = None):
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.current_timeout = self.config.recovery_timeout
        self.half_open_calls = 0
        self.failure_types: dict[FailureType, int] = {}
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type for better handling"""
        error_msg = str(exception).lower()
        
        if "timeout" in error_msg:
            return FailureType.TIMEOUT
        elif "connection" in error_msg:
            return FailureType.CONNECTION
        elif "rate limit" in error_msg or "429" in error_msg:
            return FailureType.RATE_LIMIT
        elif hasattr(exception, 'status_code') and 500 <= getattr(exception, 'status_code', 0) < 600:
            return FailureType.SERVER_ERROR
        
        return FailureType.UNKNOWN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset with exponential backoff"""
        if self.last_failure_time is None:
            return True
            
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.current_timeout
    
    def _on_success(self):
        """Reset circuit breaker on successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                logger.info("Circuit breaker closing after successful recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.current_timeout = self.config.recovery_timeout
                self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failure with classification and backoff"""
        failure_type = self._classify_failure(exception)
        self.failure_types[failure_type] = self.failure_types.get(failure_type, 0) + 1
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            # Double the timeout on half-open failure
            self.current_timeout = min(
                self.current_timeout * self.config.backoff_multiplier,
                self.config.max_recovery_timeout
            )
            logger.warning(f"Circuit breaker reopened, next attempt in {self.current_timeout}s")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception(f"Circuit breaker is open, retry in {self.current_timeout - (time.time() - self.last_failure_time):.1f}s")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def get_metrics(self) -> dict:
        """Get circuit breaker metrics"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "current_timeout": self.current_timeout,
            "failure_types": {ft.value: count for ft, count in self.failure_types.items()},
            "last_failure_time": self.last_failure_time
        }