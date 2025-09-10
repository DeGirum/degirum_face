#
# Logging configuration for degirum_face package
# Copyright DeGirum Corp. 2025
#

import logging
import sys
from typing import Optional


# Package logger
logger = logging.getLogger(__name__.split(".")[0])  # 'degirum_face'


def logging_disable():
    """Disable all logging for the package."""
    logger.disabled = True


def set_log_level(level: str) -> None:
    """
    Set the logging level for the package and enable logging.

    Args:
        level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)


def configure_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> None:
    """
    Configure logging for the degirum_face package.

    Args:
        level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        format_str (Optional[str]): Custom format string. If None, uses default format.
        handler (Optional[logging.Handler]): Custom handler. If None, uses StreamHandler to stdout.
    """
    # Remove any existing handlers
    for existing_handler in logger.handlers[:]:
        logger.removeHandler(existing_handler)

    # Set level
    set_log_level(level)

    # Create handler
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    # Set format
    if format_str is None:
        format_str = "[%(asctime)s] %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
