"""Structured logging with correlation IDs for better observability."""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from ..core.errors import CacheError

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

class StructuredFormatter(logging.Formatter):
    """JSON formatter with correlation IDs and structured fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'correlation_id': correlation_id.get()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info'}:
                log_entry[key] = value
                
        return json.dumps(log_entry)

class StructuredLogger:
    """Logger with structured output and correlation tracking."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add structured handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.propagate = False
    
    def set_correlation_id(self, corr_id: Optional[str] = None) -> str:
        """Set correlation ID for current context."""
        if corr_id is None:
            corr_id = str(uuid.uuid4())[:8]
        correlation_id.set(corr_id)
        return corr_id
    
    def info(self, message: str, **kwargs):
        """Log info with structured data."""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error with structured data."""
        extra = kwargs.copy()
        if error:
            extra['error_type'] = type(error).__name__
            extra['error_message'] = str(error)
        self.logger.error(message, exc_info=error is not None, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning with structured data."""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug with structured data."""
        self.logger.debug(message, extra=kwargs)

# Global instance
logger = StructuredLogger('vector_cache')
