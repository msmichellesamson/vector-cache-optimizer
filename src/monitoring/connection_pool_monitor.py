import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass
from ..core.connection_pool import ConnectionPool
from .logger import StructuredLogger

@dataclass
class PoolHealthMetrics:
    active_connections: int
    idle_connections: int
    failed_connections: int
    avg_connection_time: float
    pool_exhaustion_count: int
    last_health_check: float

class ConnectionPoolMonitor:
    """Monitors connection pool health and performance."""
    
    def __init__(self, pool: ConnectionPool, check_interval: float = 30.0):
        self.pool = pool
        self.check_interval = check_interval
        self.logger = StructuredLogger("connection_pool_monitor")
        self.metrics_history: Dict[str, list] = {
            "connection_times": [],
            "pool_sizes": [],
            "failures": []
        }
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start continuous pool monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            return
            
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Started connection pool monitoring")
        
    async def stop_monitoring(self):
        """Stop pool monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped connection pool monitoring")
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self._check_pool_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Pool monitoring error", error=str(e))
                await asyncio.sleep(5.0)  # Back off on error
                
    async def _check_pool_health(self):
        """Check and log pool health metrics."""
        try:
            metrics = await self._collect_metrics()
            
            # Log current metrics
            self.logger.info(
                "Pool health check",
                active=metrics.active_connections,
                idle=metrics.idle_connections,
                failed=metrics.failed_connections,
                avg_conn_time=metrics.avg_connection_time,
                exhaustion_count=metrics.pool_exhaustion_count
            )
            
            # Check for alerts
            await self._check_alerts(metrics)
            
        except Exception as e:
            self.logger.error("Failed to collect pool metrics", error=str(e))
            
    async def _collect_metrics(self) -> PoolHealthMetrics:
        """Collect current pool metrics."""
        stats = await self.pool.get_stats()
        
        # Calculate average connection time from recent history
        recent_times = self.metrics_history["connection_times"][-10:]
        avg_time = sum(recent_times) / len(recent_times) if recent_times else 0.0
        
        return PoolHealthMetrics(
            active_connections=stats.get("active", 0),
            idle_connections=stats.get("idle", 0), 
            failed_connections=stats.get("failed", 0),
            avg_connection_time=avg_time,
            pool_exhaustion_count=stats.get("exhausted", 0),
            last_health_check=time.time()
        )
        
    async def _check_alerts(self, metrics: PoolHealthMetrics):
        """Check if metrics trigger any alerts."""
        # Alert on pool exhaustion
        if metrics.pool_exhaustion_count > 0:
            self.logger.warning(
                "Pool exhaustion detected",
                exhaustion_count=metrics.pool_exhaustion_count,
                active=metrics.active_connections
            )
            
        # Alert on high connection times
        if metrics.avg_connection_time > 1.0:  # 1 second threshold
            self.logger.warning(
                "High connection times detected",
                avg_time=metrics.avg_connection_time
            )
            
        # Alert on connection failures
        if metrics.failed_connections > 5:
            self.logger.error(
                "High connection failure rate",
                failed=metrics.failed_connections
            )
    
    def record_connection_time(self, connection_time: float):
        """Record a connection establishment time."""
        self.metrics_history["connection_times"].append(connection_time)
        # Keep only last 100 measurements
        if len(self.metrics_history["connection_times"]) > 100:
            self.metrics_history["connection_times"] = \
                self.metrics_history["connection_times"][-100:]