"""Cache size drift detection for memory optimization."""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class DriftPoint:
    """Single drift measurement point."""
    timestamp: float
    size_bytes: int
    hit_rate: float
    eviction_count: int

class DriftDetector:
    """Detects unusual cache size growth patterns."""
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.3):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.measurements: deque[DriftPoint] = deque(maxlen=window_size)
        self._baseline_size: Optional[int] = None
        self._last_alert: float = 0
        self._alert_cooldown = 300  # 5 minutes
        
    def record_measurement(self, size_bytes: int, hit_rate: float, eviction_count: int) -> None:
        """Record a cache measurement point."""
        point = DriftPoint(
            timestamp=time.time(),
            size_bytes=size_bytes,
            hit_rate=hit_rate,
            eviction_count=eviction_count
        )
        
        self.measurements.append(point)
        
        if self._baseline_size is None and len(self.measurements) >= 10:
            self._establish_baseline()
            
    def detect_drift(self) -> Optional[Dict]:
        """Detect if cache size is drifting unusually."""
        if len(self.measurements) < 20 or self._baseline_size is None:
            return None
            
        recent_points = list(self.measurements)[-10:]
        avg_recent_size = sum(p.size_bytes for p in recent_points) / len(recent_points)
        
        drift_ratio = abs(avg_recent_size - self._baseline_size) / self._baseline_size
        
        if drift_ratio > self.drift_threshold:
            return self._create_drift_alert(drift_ratio, avg_recent_size)
            
        return None
        
    def _establish_baseline(self) -> None:
        """Establish baseline cache size from early measurements."""
        baseline_points = list(self.measurements)[:20]
        self._baseline_size = sum(p.size_bytes for p in baseline_points) // len(baseline_points)
        logger.info(f"Established drift baseline: {self._baseline_size} bytes")
        
    def _create_drift_alert(self, drift_ratio: float, current_size: int) -> Optional[Dict]:
        """Create drift alert if not in cooldown."""
        now = time.time()
        if now - self._last_alert < self._alert_cooldown:
            return None
            
        self._last_alert = now
        
        recent_hit_rate = self.measurements[-1].hit_rate
        trend = "growing" if current_size > self._baseline_size else "shrinking"
        
        return {
            "type": "cache_drift",
            "severity": "warning" if drift_ratio < 0.5 else "critical",
            "drift_ratio": drift_ratio,
            "baseline_size": self._baseline_size,
            "current_size": current_size,
            "trend": trend,
            "hit_rate": recent_hit_rate,
            "timestamp": now
        }