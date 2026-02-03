from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from src.ml.predictor import CachePredictor
from src.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metrics: Dict[str, float]
    threshold: float
    actual: float

class AlertEngine:
    def __init__(self, predictor: CachePredictor, metrics: MetricsCollector):
        self.predictor = predictor
        self.metrics = metrics
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # ML-driven thresholds (updated by predictor)
        self.dynamic_thresholds = {
            "hit_rate_min": 0.85,  # Updated by ML model
            "latency_max_ms": 50,
            "memory_usage_max": 0.8,
            "prediction_accuracy_min": 0.75
        }
    
    def update_thresholds(self, cache_stats: Dict[str, float]) -> None:
        """Update alert thresholds based on ML predictions"""
        try:
            # Use predictor to adjust hit rate threshold
            predicted_optimal = self.predictor.predict_optimal_hit_rate(cache_stats)
            self.dynamic_thresholds["hit_rate_min"] = max(0.7, predicted_optimal * 0.9)
            
            # Adjust latency threshold based on recent performance
            recent_p95 = cache_stats.get("latency_p95_ms", 50)
            self.dynamic_thresholds["latency_max_ms"] = min(100, recent_p95 * 1.5)
            
            logger.info(f"Updated thresholds: {self.dynamic_thresholds}")
        except Exception as e:
            logger.error(f"Failed to update thresholds: {e}")
    
    def check_alerts(self) -> List[Alert]:
        """Check all alert conditions and return new alerts"""
        new_alerts = []
        current_metrics = self.metrics.get_current_metrics()
        
        # Update thresholds first
        self.update_thresholds(current_metrics)
        
        # Check each alert condition
        alert_checks = [
            self._check_hit_rate,
            self._check_latency,
            self._check_memory_usage,
            self._check_prediction_accuracy
        ]
        
        for check in alert_checks:
            alert = check(current_metrics)
            if alert and alert.name not in self.active_alerts:
                new_alerts.append(alert)
                self.active_alerts[alert.name] = alert
                self.alert_history.append(alert)
        
        return new_alerts
    
    def _check_hit_rate(self, metrics: Dict[str, float]) -> Optional[Alert]:
        hit_rate = metrics.get("hit_rate", 0.0)
        threshold = self.dynamic_thresholds["hit_rate_min"]
        
        if hit_rate < threshold:
            return Alert(
                name="low_hit_rate",
                severity=AlertSeverity.HIGH if hit_rate < 0.6 else AlertSeverity.MEDIUM,
                message=f"Cache hit rate {hit_rate:.2%} below threshold {threshold:.2%}",
                timestamp=datetime.utcnow(),
                metrics=metrics,
                threshold=threshold,
                actual=hit_rate
            )
        return None
    
    def _check_latency(self, metrics: Dict[str, float]) -> Optional[Alert]:
        latency_p95 = metrics.get("latency_p95_ms", 0.0)
        threshold = self.dynamic_thresholds["latency_max_ms"]
        
        if latency_p95 > threshold:
            return Alert(
                name="high_latency",
                severity=AlertSeverity.CRITICAL if latency_p95 > 200 else AlertSeverity.HIGH,
                message=f"P95 latency {latency_p95:.1f}ms exceeds threshold {threshold:.1f}ms",
                timestamp=datetime.utcnow(),
                metrics=metrics,
                threshold=threshold,
                actual=latency_p95
            )
        return None
    
    def _check_memory_usage(self, metrics: Dict[str, float]) -> Optional[Alert]:
        memory_usage = metrics.get("memory_usage_ratio", 0.0)
        threshold = self.dynamic_thresholds["memory_usage_max"]
        
        if memory_usage > threshold:
            return Alert(
                name="high_memory_usage",
                severity=AlertSeverity.CRITICAL if memory_usage > 0.95 else AlertSeverity.HIGH,
                message=f"Memory usage {memory_usage:.1%} exceeds threshold {threshold:.1%}",
                timestamp=datetime.utcnow(),
                metrics=metrics,
                threshold=threshold,
                actual=memory_usage
            )
        return None
    
    def _check_prediction_accuracy(self, metrics: Dict[str, float]) -> Optional[Alert]:
        accuracy = metrics.get("prediction_accuracy", 1.0)
        threshold = self.dynamic_thresholds["prediction_accuracy_min"]
        
        if accuracy < threshold:
            return Alert(
                name="low_prediction_accuracy",
                severity=AlertSeverity.MEDIUM,
                message=f"ML prediction accuracy {accuracy:.1%} below threshold {threshold:.1%}",
                timestamp=datetime.utcnow(),
                metrics=metrics,
                threshold=threshold,
                actual=accuracy
            )
        return None
    
    def resolve_alert(self, alert_name: str) -> bool:
        """Mark an alert as resolved"""
        if alert_name in self.active_alerts:
            del self.active_alerts[alert_name]
            logger.info(f"Resolved alert: {alert_name}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        return list(self.active_alerts.values())