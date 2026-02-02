"""Cache-specific error types with metrics integration."""

import time
from enum import Enum
from typing import Optional
from src.metrics.collector import MetricsCollector


class CacheErrorType(Enum):
    """Cache error categories for metrics and monitoring."""
    CONNECTION_FAILED = "connection_failed"
    SERIALIZATION_ERROR = "serialization_error"
    MEMORY_PRESSURE = "memory_pressure"
    EVICTION_FAILED = "eviction_failed"
    CIRCUIT_OPEN = "circuit_open"
    TIMEOUT = "timeout"


class CacheError(Exception):
    """Base cache exception with automatic metrics tracking."""
    
    def __init__(
        self, 
        message: str,
        error_type: CacheErrorType,
        key: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.key = key
        self.timestamp = time.time()
        
        if metrics_collector:
            metrics_collector.increment_counter(
                "cache_errors_total",
                labels={"error_type": error_type.value, "key_provided": key is not None}
            )


class ConnectionError(CacheError):
    """Redis connection failures."""
    
    def __init__(self, message: str, key: Optional[str] = None, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(message, CacheErrorType.CONNECTION_FAILED, key, metrics_collector)


class SerializationError(CacheError):
    """Vector serialization/deserialization failures."""
    
    def __init__(self, message: str, key: Optional[str] = None, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(message, CacheErrorType.SERIALIZATION_ERROR, key, metrics_collector)


class MemoryPressureError(CacheError):
    """Memory constraints preventing cache operations."""
    
    def __init__(self, message: str, key: Optional[str] = None, metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(message, CacheErrorType.MEMORY_PRESSURE, key, metrics_collector)
