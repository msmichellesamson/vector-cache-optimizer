import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import signal
import sys


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    name: str
    message: str
    severity: AlertSeverity
    labels: Dict[str, str]
    timestamp: float


class AlertEngine:
    def __init__(self, webhook_url: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.webhook_url = webhook_url
        self._running = False
        self._alert_queue = asyncio.Queue(maxsize=1000)
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self):
        """Start the alert processing loop"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Alert engine started")
        
        try:
            await self._process_alerts()
        except Exception as e:
            self.logger.error(f"Alert engine error: {e}")
            raise
        finally:
            self._running = False
    
    async def shutdown(self):
        """Gracefully shutdown alert engine"""
        if not self._running:
            return
        
        self.logger.info("Shutting down alert engine...")
        self._running = False
        self._shutdown_event.set()
        
        # Process remaining alerts
        remaining_alerts = self._alert_queue.qsize()
        if remaining_alerts > 0:
            self.logger.info(f"Processing {remaining_alerts} remaining alerts")
            timeout = 30  # 30 seconds to drain
            try:
                await asyncio.wait_for(self._drain_queue(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout draining queue, {self._alert_queue.qsize()} alerts lost")
    
    async def _drain_queue(self):
        """Drain remaining alerts in queue"""
        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                await self._send_alert(alert)
                self._alert_queue.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error draining alert: {e}")
    
    async def _process_alerts(self):
        """Main alert processing loop"""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Wait for alert or shutdown
                alert_task = asyncio.create_task(self._alert_queue.get())
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                
                done, pending = await asyncio.wait(
                    [alert_task, shutdown_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Process alert if available
                if alert_task in done:
                    alert = await alert_task
                    await self._send_alert(alert)
                    self._alert_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(1)
    
    async def _send_alert(self, alert: Alert):
        """Send alert with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
                # In real implementation, send to webhook/Slack/PagerDuty
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to send alert after {max_retries} attempts: {e}")
                else:
                    await asyncio.sleep(2 ** attempt)