"""Structured logging configuration for Aero-Eval."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging with Rich handler."""
    logger = logging.getLogger("aero_eval")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_time=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module."""
    return logging.getLogger(f"aero_eval.{name}")
