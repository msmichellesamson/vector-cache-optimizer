import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import json

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    severity: str
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0

class AlertEngine:
    def __init__(self, webhook_url: str, retry_config: Optional[RetryConfig] = None):
        self.webhook_url = webhook_url
        self.retry_config = retry_config or RetryConfig()
        self._active_alerts: Dict[str, Alert] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert with exponential backoff retry logic"""
        if not self._session:
            raise RuntimeError("AlertEngine not properly initialized")

        alert_key = f"{alert.metric}_{alert.severity}"
        
        # Deduplicate alerts within 5 minutes
        if alert_key in self._active_alerts:
            last_alert = self._active_alerts[alert_key]
            if datetime.utcnow() - last_alert.timestamp < timedelta(minutes=5):
                logger.debug(f"Suppressing duplicate alert: {alert_key}")
                return True

        payload = {
            "text": f"🚨 {alert.severity.upper()}: {alert.message}",
            "attachments": [{
                "color": "danger" if alert.severity == "critical" else "warning",
                "fields": [
                    {"title": "Metric", "value": alert.metric, "short": True},
                    {"title": "Value", "value": str(alert.value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                ]
            }]
        }

        success = await self._send_with_retry(payload)
        
        if success:
            self._active_alerts[alert_key] = alert
            logger.info(f"Alert sent successfully: {alert_key}")
        else:
            logger.error(f"Failed to send alert after all retries: {alert_key}")
            
        return success

    async def _send_with_retry(self, payload: dict) -> bool:
        """Send webhook with exponential backoff"""
        delay = self.retry_config.base_delay
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                async with self._session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status < 400:
                        return True
                    
                    logger.warning(
                        f"Alert webhook failed (attempt {attempt + 1}): "
                        f"HTTP {response.status}"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Alert webhook error (attempt {attempt + 1}): {e}"
                )
            
            # Don't sleep after the last attempt
            if attempt < self.retry_config.max_attempts - 1:
                await asyncio.sleep(min(delay, self.retry_config.max_delay))
                delay *= self.retry_config.backoff_factor
        
        return False

    def clear_alert(self, metric: str, severity: str):
        """Clear active alert to allow re-sending"""
        alert_key = f"{metric}_{severity}"
        if alert_key in self._active_alerts:
            del self._active_alerts[alert_key]
            logger.info(f"Cleared active alert: {alert_key}")