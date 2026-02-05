"""Custom exceptions for vector cache optimizer."""


class VectorCacheError(Exception):
    """Base exception for vector cache errors."""
    pass


class CacheConnectionError(VectorCacheError):
    """Raised when cache connection fails."""
    pass


class CacheHealthError(VectorCacheError):
    """Raised when cache health check fails."""
    pass


class EmbeddingError(VectorCacheError):
    """Raised when embedding operations fail."""
    pass


class CircuitBreakerError(VectorCacheError):
    """Raised when circuit breaker is open."""
    pass


class MLPredictionError(VectorCacheError):
    """Raised when ML prediction fails."""
    pass


class RetryExhaustedError(VectorCacheError):
    """Raised when all retry attempts are exhausted."""
    pass


class ConfigurationError(VectorCacheError):
    """Raised when configuration is invalid."""
    pass
