"""Structured logging configuration for the wellness RAG application."""

import logging
import sys
from typing import Any, Dict
import structlog
from structlog.stdlib import LoggerFactory

from backend.config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.debug 
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log an event with structured data."""
        self.logger.info(event, **kwargs)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log an error with context."""
        context = context or {}
        self.logger.error(
            "Error occurred",
            error=str(error),
            error_type=type(error).__name__,
            **context
        )
