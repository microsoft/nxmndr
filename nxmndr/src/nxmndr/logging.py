# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Structured logging configuration for the inference system.

This module provides consistent logging configuration across all modules,
with structured output and proper formatting.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add process/thread info for debugging
        if hasattr(record, "process") and record.process:
            log_data["process"] = record.process
        if hasattr(record, "thread") and record.thread:
            log_data["thread"] = record.thread

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Include extra fields passed to logger
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        # Format the base message
        base_msg = super().format(record)

        # Append duration if present
        if hasattr(record, "duration_ms"):
            base_msg += f" ({record.duration_ms}ms)"

        return base_msg


def setup_logging(
    level: str = "INFO",
    format_type: str = "human",  # 'human' or 'json'
    log_file: Optional[Path] = None,
    include_extra: bool = True,
) -> None:
    """Setup structured logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format ('human' for console, 'json' for structured)
        log_file: Optional file path to write logs to
        include_extra: Whether to include extra fields in JSON logs
    """
    # Convert provided level to logging constant or integer value
    if isinstance(level, str):
        cleaned_level = level.strip()
        if cleaned_level.isdigit():
            numeric_level = int(cleaned_level)
        else:
            numeric_level = getattr(logging, cleaned_level.upper(), logging.INFO)
    elif isinstance(level, int):
        numeric_level = level
    else:
        numeric_level = logging.INFO

    # Choose formatter
    formatter: logging.Formatter
    if format_type == "json":
        formatter = StructuredFormatter(include_extra=include_extra)
    else:
        formatter = HumanReadableFormatter()

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        # Always use JSON format for files
        file_handler.setFormatter(StructuredFormatter(include_extra=include_extra))
        root_logger.addHandler(file_handler)

    # Set specific logger levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)


def _ensure_logging_configured() -> None:
    """Configure root logger once, honoring PYTHONLOGLEVEL if set."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    env_level = os.getenv("PYTHONLOGLEVEL")
    if env_level:
        setup_logging(level=env_level)
    else:
        setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    _ensure_logging_configured()
    return logging.getLogger(name)


def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics in a structured way."""
    logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "performance_metrics": kwargs,
        },
    )


def log_model_info(logger: logging.Logger, model_spec: Any, **kwargs) -> None:
    """Log model information in a structured way."""
    logger.info(
        f"Model: {model_spec.__class__.__name__}",
        extra={"model_type": model_spec.__class__.__name__, "model_info": kwargs},
    )


# Context manager for performance logging
class PerformanceContext:
    """Context manager for automatic performance logging."""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.extra_info = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            log_performance(self.logger, self.operation, duration, **self.extra_info)
        else:
            self.logger.error(
                f"Failed: {self.operation}",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(exc_val),
                    **self.extra_info,
                },
            )
