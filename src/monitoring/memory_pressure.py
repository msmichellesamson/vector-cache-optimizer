import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager


@dataclass
class MemoryPressureEvent:
    pressure_level: float
    available_memory: int
    used_memory: int
    timestamp: float
    threshold_breached: bool


class MemoryPressureMonitor:
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._callbacks: list = []

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            lines = meminfo.split('\n')
            stats = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    if 'kB' in value:
                        stats[key.strip()] = int(value.replace('kB', '').strip()) * 1024
            
            total = stats.get('MemTotal', 0)
            available = stats.get('MemAvailable', 0)
            used = total - available
            pressure = used / total if total > 0 else 0.0
            
            return {
                'total': total,
                'used': used,
                'available': available,
                'pressure': pressure
            }
        except Exception as e:
            self.logger.error(f"Failed to read memory stats: {e}")
            return {'total': 0, 'used': 0, 'available': 0, 'pressure': 0.0}

    async def register_callback(self, callback):
        """Register callback for memory pressure events."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    @asynccontextmanager
    async def monitor(self, interval: float = 1.0):
        """Async context manager for memory pressure monitoring."""
        self._monitoring = True
        monitor_task = None
        
        try:
            monitor_task = asyncio.create_task(self._monitor_loop(interval))
            yield self
        finally:
            self._monitoring = False
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    async def _monitor_loop(self, interval: float):
        """Internal monitoring loop."""
        while self._monitoring:
            try:
                stats = await self.get_memory_stats()
                pressure = stats['pressure']
                
                event = MemoryPressureEvent(
                    pressure_level=pressure,
                    available_memory=stats['available'],
                    used_memory=stats['used'],
                    timestamp=asyncio.get_event_loop().time(),
                    threshold_breached=pressure >= self.warning_threshold
                )
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(interval)
