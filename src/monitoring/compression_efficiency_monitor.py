"""Compression efficiency monitoring with adaptive thresholds."""

import asyncio
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CompressionMetrics:
    """Compression efficiency metrics."""
    ratio: float
    bytes_saved: int
    cpu_overhead_ms: float
    memory_pressure: float


class CompressionEfficiencyMonitor:
    """Monitors compression efficiency and adapts thresholds based on memory pressure."""
    
    def __init__(self, memory_threshold: float = 0.8):
        self.logger = logging.getLogger(__name__)
        self.memory_threshold = memory_threshold
        self._metrics_history: Dict[str, list] = {}
        self._adaptive_threshold = 2.0  # Start with 2:1 compression ratio
        
    async def track_compression(self, 
                              original_size: int, 
                              compressed_size: int, 
                              cpu_time_ms: float,
                              memory_pressure: float) -> CompressionMetrics:
        """Track single compression operation."""
        try:
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            bytes_saved = original_size - compressed_size
            
            metrics = CompressionMetrics(
                ratio=ratio,
                bytes_saved=bytes_saved,
                cpu_overhead_ms=cpu_time_ms,
                memory_pressure=memory_pressure
            )
            
            await self._update_history(metrics)
            await self._adapt_threshold(memory_pressure)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to track compression: {e}")
            return CompressionMetrics(1.0, 0, 0.0, memory_pressure)
    
    async def _update_history(self, metrics: CompressionMetrics) -> None:
        """Update metrics history with cleanup."""
        timestamp = datetime.utcnow()
        
        for key in ['ratio', 'bytes_saved', 'cpu_overhead']:
            if key not in self._metrics_history:
                self._metrics_history[key] = []
            
            value = getattr(metrics, key)
            self._metrics_history[key].append((timestamp, value))
            
            # Keep only last 1000 entries
            if len(self._metrics_history[key]) > 1000:
                self._metrics_history[key] = self._metrics_history[key][-1000:]
    
    async def _adapt_threshold(self, memory_pressure: float) -> None:
        """Adapt compression threshold based on memory pressure."""
        if memory_pressure > self.memory_threshold:
            # Under memory pressure - lower threshold to save more memory
            self._adaptive_threshold = max(1.5, self._adaptive_threshold * 0.9)
        elif memory_pressure < 0.5:
            # Plenty of memory - can afford higher threshold
            self._adaptive_threshold = min(4.0, self._adaptive_threshold * 1.1)
            
        self.logger.debug(f"Adaptive threshold: {self._adaptive_threshold:.2f}")
    
    def should_compress(self, estimated_ratio: float, memory_pressure: float) -> bool:
        """Determine if compression is worthwhile."""
        # Always compress under high memory pressure
        if memory_pressure > self.memory_threshold:
            return estimated_ratio > 1.2
            
        return estimated_ratio > self._adaptive_threshold
    
    def get_efficiency_summary(self) -> Dict[str, float]:
        """Get current efficiency summary."""
        if not self._metrics_history.get('ratio'):
            return {'avg_ratio': 1.0, 'total_bytes_saved': 0, 'avg_cpu_overhead': 0.0}
            
        recent_window = datetime.utcnow() - timedelta(minutes=15)
        
        ratios = [r for t, r in self._metrics_history['ratio'] 
                 if t > recent_window]
        bytes_saved = [b for t, b in self._metrics_history['bytes_saved'] 
                      if t > recent_window]
        cpu_times = [c for t, c in self._metrics_history['cpu_overhead'] 
                    if t > recent_window]
        
        return {
            'avg_ratio': sum(ratios) / len(ratios) if ratios else 1.0,
            'total_bytes_saved': sum(bytes_saved),
            'avg_cpu_overhead': sum(cpu_times) / len(cpu_times) if cpu_times else 0.0,
            'adaptive_threshold': self._adaptive_threshold
        }
