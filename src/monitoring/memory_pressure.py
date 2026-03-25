"""Memory pressure detection for cache scaling decisions."""
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PressureLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MemoryMetrics:
    available_mb: float
    used_percent: float
    cache_size_mb: float
    pressure_level: PressureLevel
    should_scale: bool

class MemoryPressureDetector:
    """Detects memory pressure and triggers cache scaling decisions."""
    
    def __init__(self, 
                 high_threshold: float = 85.0,
                 critical_threshold: float = 95.0,
                 min_available_mb: float = 512.0):
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.min_available_mb = min_available_mb
        
    def get_memory_metrics(self, cache_size_mb: float) -> MemoryMetrics:
        """Get current memory metrics and pressure level."""
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            used_percent = memory.percent
            
            pressure_level = self._calculate_pressure_level(
                used_percent, available_mb
            )
            
            should_scale = self._should_trigger_scaling(
                pressure_level, available_mb, cache_size_mb
            )
            
            return MemoryMetrics(
                available_mb=available_mb,
                used_percent=used_percent,
                cache_size_mb=cache_size_mb,
                pressure_level=pressure_level,
                should_scale=should_scale
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
            return MemoryMetrics(
                available_mb=0.0,
                used_percent=100.0,
                cache_size_mb=cache_size_mb,
                pressure_level=PressureLevel.CRITICAL,
                should_scale=True
            )
    
    def _calculate_pressure_level(self, used_percent: float, available_mb: float) -> PressureLevel:
        """Calculate memory pressure level based on usage."""
        if used_percent >= self.critical_threshold or available_mb < self.min_available_mb:
            return PressureLevel.CRITICAL
        elif used_percent >= self.high_threshold:
            return PressureLevel.HIGH
        elif used_percent >= 70.0:
            return PressureLevel.MEDIUM
        else:
            return PressureLevel.LOW
    
    def _should_trigger_scaling(self, 
                               pressure_level: PressureLevel,
                               available_mb: float,
                               cache_size_mb: float) -> bool:
        """Determine if we should trigger cache scaling."""
        if pressure_level == PressureLevel.CRITICAL:
            return True
        
        if pressure_level == PressureLevel.HIGH and cache_size_mb > available_mb * 0.8:
            return True
            
        return False