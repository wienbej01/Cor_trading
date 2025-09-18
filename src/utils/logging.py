"""
Logging configuration module for FX-Commodity correlation arbitrage strategy.
Provides centralized logging setup and utilities.
"""

import sys
from pathlib import Path
from typing import Optional
import subprocess
import os

from loguru import logger


def get_git_info() -> dict:
    """
    Get git repository information.
    
    Returns:
        Dictionary with git information (commit SHA, branch, etc.)
    """
    try:
        # Get current commit SHA
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get git status
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        is_clean = len(status) == 0
        
        return {
            "commit_sha": commit_sha,
            "branch": branch,
            "is_clean": is_clean
        }
    except Exception:
        return {
            "commit_sha": "unknown",
            "branch": "unknown",
            "is_clean": False
        }


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Path to log file. If None, logs only to console.
        log_format: Custom log format. If None, uses default format.
        rotation: Log file rotation setting.
        retention: Log file retention setting.
    """
    # Remove default logger
    logger.remove()

    # Default log format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console logger
    logger.add(sys.stdout, format=log_format, level=log_level, colorize=True)

    # Add file logger if specified
    if log_file is not None:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (usually __name__).

    Returns:
        Logger instance.
    """
    return logger.bind(name=name)


def log_function_entry_exit(func):
    """
    Decorator to log function entry and exit.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func_name}")
        return result

    return wrapper


def log_performance(func):
    """
    Decorator to log function execution time.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """
    import time

    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func_name} executed in {execution_time:.4f} seconds")
        return result

    return wrapper


class TradingLogger:
    """
    Specialized logger for trading operations.
    """

    def __init__(self, name: str = "Trading"):
        """
        Initialize the trading logger.

        Args:
            name: Name for the logger.
        """
        self.logger = logger.bind(name=name)

    def log_signal(self, signal_type: str, details: dict) -> None:
        """
        Log a trading signal.

        Args:
            signal_type: Type of signal ("entry", "exit", "stop_loss").
            details: Dictionary with signal details.
        """
        self.logger.info(f"Signal: {signal_type.upper()}", details=details)

    def log_trade(self, trade_id: str, action: str, details: dict) -> None:
        """
        Log a trade action.

        Args:
            trade_id: Unique trade identifier.
            action: Trade action ("open", "close", "modify").
            details: Dictionary with trade details.
        """
        self.logger.info(f"Trade {trade_id}: {action.upper()}", details=details)

    def log_performance(self, metrics: dict) -> None:
        """
        Log performance metrics.

        Args:
            metrics: Dictionary with performance metrics.
        """
        self.logger.info("Performance metrics", metrics=metrics)

    def log_error(
        self, error_type: str, error_message: str, details: dict = None
    ) -> None:
        """
        Log an error with context.

        Args:
            error_type: Type of error.
            error_message: Error message.
            details: Additional error context.
        """
        if details is None:
            details = {}

        self.logger.error(f"Error: {error_type} - {error_message}", details=details)

    def log_data_quality(self, data_type: str, quality_metrics: dict) -> None:
        """
        Log data quality metrics.

        Args:
            data_type: Type of data ("fx", "commodity", "spread").
            quality_metrics: Dictionary with quality metrics.
        """
        self.logger.debug(f"Data quality: {data_type}", metrics=quality_metrics)


# Initialize default logging
setup_logging()

# Create default trading logger
trading_logger = TradingLogger()
