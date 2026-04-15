"""Real-time cache compression ratio monitoring and optimization."""

import time
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .structured_logger import StructuredLogger


@dataclass
class CompressionStats:
    """Compression statistics for a time window."""
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    savings_pct: float
    timestamp: float


class CompressionTracker:
    """Tracks compression ratios and identifies optimization opportunities."""
    
    def __init__(self, window_size: int = 100, alert_threshold: float = 0.3):
        self.window_size = window_size
        self.alert_threshold = alert_threshold  # Alert if ratio drops below this
        self.stats_window = deque(maxlen=window_size)
        self.logger = StructuredLogger("compression_tracker")
        self._total_raw = 0
        self._total_compressed = 0
        
    def record_compression(self, raw_size: int, compressed_size: int) -> None:
        """Record a compression event."""
        if raw_size <= 0:
            return
            
        ratio = compressed_size / raw_size if raw_size > 0 else 1.0
        savings_pct = ((raw_size - compressed_size) / raw_size) * 100
        
        stats = CompressionStats(
            raw_bytes=raw_size,
            compressed_bytes=compressed_size,
            ratio=ratio,
            savings_pct=savings_pct,
            timestamp=time.time()
        )
        
        self.stats_window.append(stats)
        self._total_raw += raw_size
        self._total_compressed += compressed_size
        
        # Alert on poor compression
        if ratio > (1.0 - self.alert_threshold):
            self.logger.warning(
                "Poor compression detected",
                ratio=ratio,
                raw_size=raw_size,
                compressed_size=compressed_size
            )
    
    def get_current_ratio(self) -> Optional[float]:
        """Get current average compression ratio."""
        if not self.stats_window:
            return None
            
        total_raw = sum(s.raw_bytes for s in self.stats_window)
        total_compressed = sum(s.compressed_bytes for s in self.stats_window)
        
        return total_compressed / total_raw if total_raw > 0 else None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get compression metrics for monitoring."""
        current_ratio = self.get_current_ratio()
        lifetime_ratio = (
            self._total_compressed / self._total_raw 
            if self._total_raw > 0 else 0.0
        )
        
        recent_savings = 0.0
        if self.stats_window:
            recent_savings = sum(s.savings_pct for s in self.stats_window) / len(self.stats_window)
        
        return {
            "current_compression_ratio": current_ratio or 0.0,
            "lifetime_compression_ratio": lifetime_ratio,
            "recent_savings_percent": recent_savings,
            "total_raw_bytes": float(self._total_raw),
            "total_compressed_bytes": float(self._total_compressed),
            "compression_events": float(len(self.stats_window))
        }
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start periodic compression monitoring."""
        while True:
            try:
                metrics = self.get_metrics()
                self.logger.info("Compression metrics", **metrics)
                
                # Check for degradation
                current_ratio = metrics["current_compression_ratio"]
                if current_ratio > 0.8:  # Very poor compression
                    self.logger.error(
                        "Severe compression degradation detected",
                        ratio=current_ratio
                    )
                    
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error("Compression monitoring error", error=str(e))
                await asyncio.sleep(interval)
