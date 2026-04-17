"""Cache statistics tracker for comprehensive performance monitoring."""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from collections import defaultdict, deque
import threading
from .structured_logger import StructuredLogger


@dataclass
class CacheStats:
    """Cache statistics snapshot."""
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    memory_usage_mb: float
    avg_access_time_ms: float
    total_operations: int
    errors_per_minute: float
    fragmentation_ratio: float


class CacheStatsTracker:
    """Tracks comprehensive cache statistics with sliding windows."""
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.logger = StructuredLogger("cache_stats")
        
        # Sliding windows for metrics
        self._hits = deque(maxlen=window_size)
        self._misses = deque(maxlen=window_size)
        self._evictions = deque(maxlen=window_size)
        self._access_times = deque(maxlen=window_size)
        self._errors = deque(maxlen=window_size)
        self._memory_usage = deque(maxlen=window_size)
        self._timestamps = deque(maxlen=window_size)
        
        # Counters
        self._total_operations = 0
        self._lock = threading.RLock()
        
    def record_hit(self, access_time_ms: float) -> None:
        """Record a cache hit."""
        with self._lock:
            current_time = time.time()
            self._hits.append(1)
            self._misses.append(0)
            self._access_times.append(access_time_ms)
            self._timestamps.append(current_time)
            self._total_operations += 1
            
    def record_miss(self, access_time_ms: float) -> None:
        """Record a cache miss."""
        with self._lock:
            current_time = time.time()
            self._hits.append(0)
            self._misses.append(1)
            self._access_times.append(access_time_ms)
            self._timestamps.append(current_time)
            self._total_operations += 1
            
    def record_eviction(self) -> None:
        """Record a cache eviction."""
        with self._lock:
            self._evictions.append(1)
            
    def record_error(self) -> None:
        """Record a cache error."""
        with self._lock:
            current_time = time.time()
            self._errors.append(current_time)
            
    def record_memory_usage(self, usage_mb: float) -> None:
        """Record current memory usage."""
        with self._lock:
            self._memory_usage.append(usage_mb)
            
    def get_current_stats(self) -> Optional[CacheStats]:
        """Get current cache statistics."""
        with self._lock:
            if not self._hits:
                return None
                
            # Calculate rates
            total_hits = sum(self._hits)
            total_misses = sum(self._misses)
            total_requests = total_hits + total_misses
            
            if total_requests == 0:
                return None
                
            hit_rate = total_hits / total_requests
            miss_rate = total_misses / total_requests
            
            # Calculate eviction rate
            eviction_rate = sum(self._evictions) / len(self._evictions) if self._evictions else 0
            
            # Calculate average access time
            avg_access_time = sum(self._access_times) / len(self._access_times) if self._access_times else 0
            
            # Calculate error rate (errors per minute)
            current_time = time.time()
            recent_errors = [t for t in self._errors if current_time - t <= 60]
            errors_per_minute = len(recent_errors)
            
            # Memory usage
            memory_usage = self._memory_usage[-1] if self._memory_usage else 0
            
            # Fragmentation (simplified calculation)
            fragmentation_ratio = min(eviction_rate * 2, 1.0)  # Rough estimate
            
            return CacheStats(
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                eviction_rate=eviction_rate,
                memory_usage_mb=memory_usage,
                avg_access_time_ms=avg_access_time,
                total_operations=self._total_operations,
                errors_per_minute=errors_per_minute,
                fragmentation_ratio=fragmentation_ratio
            )
            
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for monitoring systems."""
        stats = self.get_current_stats()
        if not stats:
            return {}
            
        return {
            'cache_hit_rate': stats.hit_rate,
            'cache_miss_rate': stats.miss_rate,
            'cache_eviction_rate': stats.eviction_rate,
            'cache_memory_usage_mb': stats.memory_usage_mb,
            'cache_avg_access_time_ms': stats.avg_access_time_ms,
            'cache_total_operations': stats.total_operations,
            'cache_errors_per_minute': stats.errors_per_minute,
            'cache_fragmentation_ratio': stats.fragmentation_ratio,
        }