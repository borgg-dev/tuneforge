"""
Logging utilities for TuneForge Subnet.

Provides consistent loguru-based logging setup across all components.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: str = "/tmp/tuneforge",
    component_name: str = "tuneforge",
) -> None:
    """
    Configure loguru logging with stderr output and file rotation.

    Args:
        level: Log level for stderr output (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        component_name: Component name used for the log filename.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        log_path / f"{component_name}.log",
        rotation="100 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )
