from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .hit_rate_tracker import HitRateTracker
from .alert_engine import AlertEngine, AlertLevel, Alert


@dataclass
class HitRateThresholds:
    """Hit rate alert thresholds."""
    critical: float = 0.70  # Below 70% is critical
    warning: float = 0.85   # Below 85% is warning
    window_minutes: int = 5  # Evaluation window


class HitRateAlerter:
    """Real-time cache hit rate monitoring and alerting."""
    
    def __init__(self, 
                 hit_rate_tracker: HitRateTracker,
                 alert_engine: AlertEngine,
                 thresholds: Optional[HitRateThresholds] = None):
        self.hit_rate_tracker = hit_rate_tracker
        self.alert_engine = alert_engine
        self.thresholds = thresholds or HitRateThresholds()
        self.logger = logging.getLogger(__name__)
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=10)
    
    async def check_hit_rate(self) -> None:
        """Check current hit rate and trigger alerts if needed."""
        try:
            window_start = datetime.utcnow() - timedelta(minutes=self.thresholds.window_minutes)
            current_rate = await self.hit_rate_tracker.get_hit_rate_since(window_start)
            
            if current_rate is None:
                self.logger.warning("No hit rate data available")
                return
            
            await self._evaluate_and_alert(current_rate)
            
        except Exception as e:
            self.logger.error(f"Hit rate check failed: {e}")
    
    async def _evaluate_and_alert(self, hit_rate: float) -> None:
        """Evaluate hit rate and send alerts if thresholds are breached."""
        alert_type = None
        level = None
        
        if hit_rate < self.thresholds.critical:
            alert_type = "critical_hit_rate"
            level = AlertLevel.CRITICAL
        elif hit_rate < self.thresholds.warning:
            alert_type = "low_hit_rate"
            level = AlertLevel.WARNING
        
        if alert_type and self._should_send_alert(alert_type):
            alert = Alert(
                level=level,
                message=f"Cache hit rate dropped to {hit_rate:.2%} (threshold: {self._get_threshold(level):.2%})",
                tags={"component": "cache", "metric": "hit_rate"},
                metadata={"hit_rate": hit_rate, "window_minutes": self.thresholds.window_minutes}
            )
            
            await self.alert_engine.send_alert(alert)
            self._last_alert_time[alert_type] = datetime.utcnow()
            
            self.logger.warning(f"Hit rate alert sent: {alert.message}")
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_time = self._last_alert_time.get(alert_type)
        if last_time is None:
            return True
        return datetime.utcnow() - last_time >= self._alert_cooldown
    
    def _get_threshold(self, level: AlertLevel) -> float:
        """Get threshold value for alert level."""
        return self.thresholds.critical if level == AlertLevel.CRITICAL else self.thresholds.warning
