import sys
import logging
import structlog
from typing import Any, Dict

def configure_logger(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    Uses structlog for JSON output in production and colored console in dev.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run solely on log entries that originate from structlog.
        foreign_pre_chain=shared_processors,
        # These run on all log entries.
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    # Use standard library logging configuration
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Quiet down some libraries
    logging.getLogger("uvicorn.access").setLevel("WARNING")
    logging.getLogger("urllib3").setLevel("WARNING")

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
