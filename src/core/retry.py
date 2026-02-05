"""Retry mechanism for cache operations with circuit breaker integration."""

import asyncio
import logging
from typing import Callable, TypeVar, Any
from functools import wraps
import random

from .errors import CacheError, CircuitBreakerError
from .circuit_breaker import CircuitBreaker

T = TypeVar('T')

logger = logging.getLogger(__name__)

class RetryConfig:
    """Configuration for retry mechanism."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

def with_retry(config: RetryConfig, circuit_breaker: CircuitBreaker = None):
    """Decorator for adding retry logic to cache operations."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Check circuit breaker before attempt
                    if circuit_breaker and not circuit_breaker.can_execute():
                        raise CircuitBreakerError("Circuit breaker is open")
                    
                    result = await func(*args, **kwargs)
                    
                    # Success - record in circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    if attempt > 0:
                        logger.info(f"Operation succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Record failure in circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    # Don't retry on last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (2 ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
            
            # All attempts failed
            logger.error(f"All {config.max_attempts} attempts failed")
            raise CacheError(f"Operation failed after {config.max_attempts} attempts") from last_exception
        
        return wrapper
    return decorator