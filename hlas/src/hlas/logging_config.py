import os
import sys
import logging
import logging.handlers
from dotenv import load_dotenv

try:
    # Optional import early; Redis is mandatory for the app, but we avoid failing logging if anything changes.
    from .redis_utils import get_redis
except Exception:
    get_redis = None  # type: ignore


def _log_once_global(key: str, level: int, message: str):
    """Log a message only once across workers using Redis SETNX. Falls back to normal log if Redis import fails."""
    logger = logging.getLogger('hlas')
    try:
        if get_redis is None:
            # No Redis import available; log normally
            logger.log(level, message)
            return
        r = get_redis()
        if r.set(f"log_once:{key}", "1", nx=True, ex=3600):
            logger.log(level, message)
    except Exception:
        # As a safety net, still log
        logger.log(level, message)


def setup_logging():
    """
    Configures the 'hlas' package logger to output to the console and a rotating file.
    This setup is controlled by environment variables.
    """
    load_dotenv()

    # Log only when DEBUG=True
    if os.getenv("DEBUG", "false").lower() != "true":
        return

    log_file = os.getenv("LOG_FILE", "logs/app.log")
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # --- Configure only the 'hlas' logger ---
    hlas_logger = logging.getLogger('hlas')
    hlas_logger.setLevel(log_level)
    
    # Prevent log messages from being passed to the root logger
    hlas_logger.propagate = False

    # Clear any existing handlers to prevent duplicates
    if hlas_logger.hasHandlers():
        hlas_logger.handlers.clear()

    # Create a shared formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    hlas_logger.addHandler(console_handler)

    # File-level logging intentionally disabled per requirements; console-only.

    # Gate this info message so only one worker prints it
    _log_once_global(
        key="logging_initialized",
        level=logging.INFO,
        message=f"Logging for 'hlas' package initialized (console-only). Level: {log_level_name}",
    )
