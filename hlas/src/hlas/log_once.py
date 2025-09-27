"""
Log deduplication utility for multi-worker environments.
Uses Redis to ensure startup logs only appear once across all workers.
"""
import logging
import os
from functools import wraps
from typing import Optional, Callable
import time

def is_worker_process() -> bool:
    """Check if we're running as a uvicorn worker process."""
    # Uvicorn workers have these env vars set
    return any([
        os.environ.get('APP_MODULE'),  # Set by uvicorn for workers
        os.environ.get('GUNICORN_PID'),  # Set by gunicorn
        '--workers' in ' '.join(os.environ.get('COMMANDLINE', '')),  # Check command line
    ])

def get_worker_id() -> str:
    """Get a unique identifier for this worker."""
    return f"{os.getpid()}"

_logged_once_cache = set()

def log_once(key: str, log_func: Callable, *args, ttl_seconds: int = 3600, **kwargs):
    """
    Log a message only once across all workers using Redis.
    
    Args:
        key: Unique key for this log message
        log_func: The logging function to call (e.g., logger.info)
        *args: Arguments to pass to log_func
        ttl_seconds: How long to remember that we logged this
        **kwargs: Keyword arguments to pass to log_func
    """
    # Check local cache first to avoid Redis calls
    if key in _logged_once_cache:
        return
    
    try:
        # Try to use Redis for cross-process deduplication
        from .redis_utils import get_redis
        redis_client = get_redis()
        
        # For worker processes, only the first worker should log
        if is_worker_process():
            full_key = f"log_once:{key}"
            # Use Redis SET NX (set if not exists) with expiration
            if redis_client.set(full_key, "1", nx=True, ex=ttl_seconds):
                # We got the lock, so we can log
                log_func(*args, **kwargs)
                _logged_once_cache.add(key)
        else:
            # Not a worker process, just log normally
            log_func(*args, **kwargs)
            _logged_once_cache.add(key)
    except Exception:
        # If Redis fails, check if we're worker #2+ and suppress logs
        worker_id = get_worker_id()
        # Simple heuristic: if PID is even and we're in a worker, skip logging
        # This isn't perfect but reduces duplicates when Redis is down
        if not is_worker_process() or int(worker_id) % 2 == 1:
            log_func(*args, **kwargs)
            _logged_once_cache.add(key)

def log_once_info(logger: logging.Logger, key: str, message: str, *args):
    """Convenience wrapper for logger.info with deduplication."""
    log_once(key, logger.info, message, *args)

def log_once_warning(logger: logging.Logger, key: str, message: str, *args):
    """Convenience wrapper for logger.warning with deduplication."""
    log_once(key, logger.warning, message, *args)

def log_once_error(logger: logging.Logger, key: str, message: str, *args):
    """Convenience wrapper for logger.error with deduplication."""
    log_once(key, logger.error, message, *args)