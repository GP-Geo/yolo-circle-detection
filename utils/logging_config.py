"""
Centralized logging configuration for YOLO circle detection project.

This module provides consistent logging setup across all scripts with:
- Color-coded console output
- File logging with rotation
- Structured log format with timestamps
- Debug mode support
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


# Color codes for terminal output
class LogColors:
    """ANSI color codes for terminal output."""
    GREY = '\x1b[38;21m'
    BLUE = '\x1b[38;5;39m'
    YELLOW = '\x1b[38;5;226m'
    RED = '\x1b[38;5;196m'
    BOLD_RED = '\x1b[31;1m'
    RESET = '\x1b[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    FORMATS = {
        logging.DEBUG: LogColors.GREY + "%(levelname)-8s" + LogColors.RESET + " - %(name)s - %(message)s",
        logging.INFO: LogColors.BLUE + "%(levelname)-8s" + LogColors.RESET + " - %(name)s - %(message)s",
        logging.WARNING: LogColors.YELLOW + "%(levelname)-8s" + LogColors.RESET + " - %(name)s - %(message)s",
        logging.ERROR: LogColors.RED + "%(levelname)-8s" + LogColors.RESET + " - %(name)s - %(message)s",
        logging.CRITICAL: LogColors.BOLD_RED + "%(levelname)-8s" + LogColors.RESET + " - %(name)s - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Console logging level (default: INFO)
        log_file: Optional path to log file. If provided, logs are written to file
        console: Whether to output to console (default: True)
        file_level: File logging level (default: DEBUG)

    Returns:
        Configured logger instance

    Example:
        >>> from utils import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with color formatting
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    # File handler with rotation (10MB max, keep 5 backups)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)

        # File logs get detailed format without colors
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def enable_debug_mode():
    """Enable debug logging for all loggers in the project."""
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)
