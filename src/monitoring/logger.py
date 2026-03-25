"""Structured logging with correlation IDs for distributed tracing."""
import logging
import json
import uuid
from contextvars import ContextVar
from typing import Dict, Any, Optional
from datetime import datetime

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

class StructuredLogger:
    """Logger with structured output and correlation ID tracking."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._create_formatter())
            self.logger.addHandler(handler)
    
    def _create_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logs."""
        return logging.Formatter(
            fmt='%(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
    
    def _create_log_entry(self, level: str, message: str, **kwargs) -> str:
        """Create structured log entry as JSON."""
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            'correlation_id': correlation_id.get(),
            'service': 'vector-cache-optimizer',
            **kwargs
        }
        return json.dumps(entry, separators=(',', ':'))
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.logger.info(self._create_log_entry('INFO', message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self.logger.error(self._create_log_entry('ERROR', message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.logger.warning(self._create_log_entry('WARNING', message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.logger.debug(self._create_log_entry('DEBUG', message, **kwargs))

def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    correlation_id.set(cid)
    return cid

def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger(name)
