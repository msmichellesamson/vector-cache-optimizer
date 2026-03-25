"""Enhanced memory monitor with pressure detection."""
import asyncio
import logging
from typing import Optional, Callable
from .memory_pressure import MemoryPressureDetector, MemoryMetrics, PressureLevel
from ..metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage and trigger scaling actions."""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 check_interval: int = 30,
                 scaling_callback: Optional[Callable[[bool], None]] = None):
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval
        self.scaling_callback = scaling_callback
        self.pressure_detector = MemoryPressureDetector()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start memory monitoring."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Memory monitor started")
        
    async def stop(self):
        """Stop memory monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Memory monitor stopped")
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                cache_size_mb = await self._get_cache_size_mb()
                metrics = self.pressure_detector.get_memory_metrics(cache_size_mb)
                
                await self._record_metrics(metrics)
                await self._handle_pressure(metrics)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        try:
            # This would integrate with your cache engine
            return 0.0  # Placeholder
        except Exception:
            return 0.0
            
    async def _record_metrics(self, metrics: MemoryMetrics):
        """Record memory metrics."""
        self.metrics_collector.record_gauge('memory_available_mb', metrics.available_mb)
        self.metrics_collector.record_gauge('memory_used_percent', metrics.used_percent)
        self.metrics_collector.record_gauge('cache_size_mb', metrics.cache_size_mb)
        self.metrics_collector.record_gauge('memory_pressure_level', 
                                           self._pressure_to_numeric(metrics.pressure_level))
        
    async def _handle_pressure(self, metrics: MemoryMetrics):
        """Handle memory pressure events."""
        if metrics.should_scale and self.scaling_callback:
            logger.warning(f"Memory pressure detected: {metrics.pressure_level.value}, "
                          f"triggering scaling")
            try:
                self.scaling_callback(True)
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")
                
    def _pressure_to_numeric(self, level: PressureLevel) -> float:
        """Convert pressure level to numeric value for metrics."""
        mapping = {
            PressureLevel.LOW: 1.0,
            PressureLevel.MEDIUM: 2.0,
            PressureLevel.HIGH: 3.0,
            PressureLevel.CRITICAL: 4.0
        }
        return mapping.get(level, 0.0)