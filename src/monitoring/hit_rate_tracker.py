from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import deque
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class HitRateWindow:
    """Sliding window for hit rate calculation."""
    hits: int = 0
    total: int = 0
    timestamp: datetime = None
    
    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

class HitRateTracker:
    """Real-time cache hit rate monitoring with sliding windows."""
    
    def __init__(self, window_size_minutes: int = 5, max_windows: int = 12):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.max_windows = max_windows
        self.windows: deque[HitRateWindow] = deque(maxlen=max_windows)
        self.current_window = HitRateWindow(timestamp=datetime.utcnow())
        self._lock = threading.RLock()
        
    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._ensure_current_window()
            self.current_window.hits += 1
            self.current_window.total += 1
            
    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._ensure_current_window()
            self.current_window.total += 1
            
    def get_current_rate(self) -> float:
        """Get hit rate for current window."""
        with self._lock:
            self._ensure_current_window()
            return self.current_window.rate
            
    def get_average_rate(self, windows: int = 5) -> float:
        """Get average hit rate over last N windows."""
        with self._lock:
            self._rotate_if_needed()
            
            recent_windows = list(self.windows)[-windows:]
            if self.current_window.total > 0:
                recent_windows.append(self.current_window)
                
            if not recent_windows:
                return 0.0
                
            total_hits = sum(w.hits for w in recent_windows)
            total_requests = sum(w.total for w in recent_windows)
            
            return total_hits / total_requests if total_requests > 0 else 0.0
            
    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive hit rate metrics."""
        with self._lock:
            return {
                'current_rate': self.get_current_rate(),
                'avg_rate_5m': self.get_average_rate(1),
                'avg_rate_30m': self.get_average_rate(6),
                'avg_rate_1h': self.get_average_rate(12)
            }
            
    def _ensure_current_window(self) -> None:
        """Ensure current window is valid for current time."""
        self._rotate_if_needed()
        
    def _rotate_if_needed(self) -> None:
        """Rotate window if current window is too old."""
        now = datetime.utcnow()
        
        if now - self.current_window.timestamp >= self.window_size:
            if self.current_window.total > 0:
                self.windows.append(self.current_window)
                logger.debug(
                    f"Rotated window: {self.current_window.hits}/{self.current_window.total} "
                    f"({self.current_window.rate:.2%})"
                )
                
            self.current_window = HitRateWindow(timestamp=now)