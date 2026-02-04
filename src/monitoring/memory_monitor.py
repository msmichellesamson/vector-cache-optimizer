"""Memory pressure monitoring for intelligent cache eviction."""

import asyncio
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MemoryMetrics:
    """System memory metrics."""
    used_percent: float
    available_gb: float
    cache_size_gb: float
    pressure_level: str  # low, medium, high, critical
    timestamp: datetime


class MemoryPressureMonitor:
    """Monitors system memory pressure to inform cache eviction decisions."""
    
    def __init__(self, 
                 check_interval: int = 30,
                 high_threshold: float = 80.0,
                 critical_threshold: float = 90.0):
        self.check_interval = check_interval
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._metrics_history: list[MemoryMetrics] = []
        
    async def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        self._running = True
        self.logger.info("Memory pressure monitoring started")
        
        while self._running:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self._metrics_history) > 100:
                    self._metrics_history.pop(0)
                    
                if metrics.pressure_level in ["high", "critical"]:
                    self.logger.warning(f"Memory pressure: {metrics.pressure_level} - {metrics.used_percent:.1f}% used")
                    
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                
            await asyncio.sleep(self.check_interval)
            
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._running = False
        self.logger.info("Memory pressure monitoring stopped")
        
    def _collect_metrics(self) -> MemoryMetrics:
        """Collect current memory metrics."""
        memory = psutil.virtual_memory()
        
        pressure_level = "low"
        if memory.percent >= self.critical_threshold:
            pressure_level = "critical"
        elif memory.percent >= self.high_threshold:
            pressure_level = "high"
        elif memory.percent >= 60.0:
            pressure_level = "medium"
            
        return MemoryMetrics(
            used_percent=memory.percent,
            available_gb=memory.available / (1024**3),
            cache_size_gb=self._estimate_cache_size(),
            pressure_level=pressure_level,
            timestamp=datetime.utcnow()
        )
        
    def _estimate_cache_size(self) -> float:
        """Rough estimation of cache memory usage."""
        # This would integrate with Redis memory stats in real implementation
        return 0.5  # Placeholder
        
    def get_current_pressure(self) -> Optional[str]:
        """Get current memory pressure level."""
        if not self._metrics_history:
            return None
        return self._metrics_history[-1].pressure_level
        
    def should_aggressive_evict(self) -> bool:
        """Determine if aggressive eviction is needed."""
        current = self.get_current_pressure()
        return current in ["high", "critical"]