"""Real-time cache efficiency tracking and analysis."""
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque


@dataclass
class EfficiencyMetrics:
    """Cache efficiency metrics snapshot."""
    timestamp: float
    hit_rate: float
    memory_efficiency: float
    eviction_rate: float
    avg_retrieval_time: float
    efficiency_score: float


class EfficiencyTracker:
    """Tracks and analyzes cache efficiency in real-time."""
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
        
        # Counters for current window
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._retrieval_times = deque(maxlen=100)
        self._window_start = time.time()
        
    def record_hit(self, retrieval_time: float) -> None:
        """Record a cache hit with retrieval time."""
        self._hits += 1
        self._retrieval_times.append(retrieval_time)
        
    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1
        
    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self._evictions += 1
        
    def calculate_efficiency(self, memory_used: int, memory_total: int) -> EfficiencyMetrics:
        """Calculate current efficiency metrics."""
        now = time.time()
        window_duration = now - self._window_start
        
        # Calculate hit rate
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        # Calculate memory efficiency
        memory_efficiency = memory_used / memory_total if memory_total > 0 else 0.0
        
        # Calculate eviction rate (per second)
        eviction_rate = self._evictions / window_duration if window_duration > 0 else 0.0
        
        # Calculate average retrieval time
        avg_retrieval_time = (
            sum(self._retrieval_times) / len(self._retrieval_times)
            if self._retrieval_times else 0.0
        )
        
        # Calculate overall efficiency score (0-100)
        efficiency_score = self._calculate_score(
            hit_rate, memory_efficiency, eviction_rate, avg_retrieval_time
        )
        
        metrics = EfficiencyMetrics(
            timestamp=now,
            hit_rate=hit_rate,
            memory_efficiency=memory_efficiency,
            eviction_rate=eviction_rate,
            avg_retrieval_time=avg_retrieval_time,
            efficiency_score=efficiency_score
        )
        
        self.metrics_history.append(metrics)
        return metrics
        
    def _calculate_score(self, hit_rate: float, memory_eff: float, 
                        eviction_rate: float, avg_time: float) -> float:
        """Calculate composite efficiency score."""
        # Weight factors
        hit_weight = 0.4
        memory_weight = 0.2
        eviction_weight = 0.2
        time_weight = 0.2
        
        # Normalize eviction rate (lower is better)
        eviction_score = max(0, 1 - (eviction_rate / 10))  # Assume 10/sec is very bad
        
        # Normalize retrieval time (lower is better) 
        time_score = max(0, 1 - (avg_time / 0.1))  # Assume 100ms is very bad
        
        score = (
            hit_rate * hit_weight +
            memory_eff * memory_weight +
            eviction_score * eviction_weight +
            time_score * time_weight
        ) * 100
        
        return min(100, max(0, score))
        
    def get_trend(self, lookback: int = 60) -> Optional[str]:
        """Get efficiency trend over lookback period."""
        if len(self.metrics_history) < 2:
            return None
            
        recent_metrics = list(self.metrics_history)[-lookback:]
        if len(recent_metrics) < 2:
            return None
            
        start_score = recent_metrics[0].efficiency_score
        end_score = recent_metrics[-1].efficiency_score
        
        diff = end_score - start_score
        
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "degrading"
        else:
            return "stable"
            
    def reset_window(self) -> None:
        """Reset current tracking window."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._retrieval_times.clear()
        self._window_start = time.time()
