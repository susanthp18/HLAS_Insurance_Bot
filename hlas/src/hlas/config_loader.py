"""
Centralized configuration loader for HLAS application.
Loads and caches YAML configurations at module import time to avoid repeated file I/O.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
from threading import Lock

logger = logging.getLogger(__name__)

# Optional: gate certain startup logs so they appear only once across workers
try:
    from .redis_utils import get_redis
except Exception:
    get_redis = None  # type: ignore

def _log_once(key: str, level: int, message: str):
    try:
        if get_redis is None:
            logger.log(level, message)
            return
        r = get_redis()
        if r.set(f"log_once:{key}", "1", nx=True, ex=3600):
            logger.log(level, message)
    except Exception:
        logger.log(level, message)


class ConfigLoader:
    """
    Singleton configuration loader that caches YAML configurations.
    Configurations are loaded once at first access and reused for all subsequent requests.
    """
    
    _instance: Optional['ConfigLoader'] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> 'ConfigLoader':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the config loader and load configurations."""
        if self._initialized:
            return
            
        self._initialized = True
        self._agents_spec: Dict[str, Any] = {}
        self._tasks_spec: Dict[str, Any] = {}
        self._config_dir = Path(__file__).parent / "config"
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load YAML configurations from disk."""
        # Load agents configuration
        try:
            agents_path = self._config_dir / "agents.yaml"
            with open(agents_path, "r", encoding="utf-8") as f:
                self._agents_spec = yaml.safe_load(f) or {}
            _log_once(
                key="config_agents_loaded",
                level=logging.INFO,
                message=f"ConfigLoader: Loaded agents.yaml - {len(self._agents_spec)} agents defined",
            )
        except Exception as e:
            logger.error("ConfigLoader: Failed to load agents.yaml - %s", str(e))
            self._agents_spec = {}
        
        # Load tasks configuration
        try:
            tasks_path = self._config_dir / "tasks.yaml"
            with open(tasks_path, "r", encoding="utf-8") as f:
                self._tasks_spec = yaml.safe_load(f) or {}
            _log_once(
                key="config_tasks_loaded",
                level=logging.INFO,
                message=f"ConfigLoader: Loaded tasks.yaml - {len(self._tasks_spec)} tasks defined",
            )
        except Exception as e:
            logger.error("ConfigLoader: Failed to load tasks.yaml - %s", str(e))
            self._tasks_spec = {}
    
    @property
    def agents_spec(self) -> Dict[str, Any]:
        """Get the cached agents specification."""
        return self._agents_spec
    
    @property
    def tasks_spec(self) -> Dict[str, Any]:
        """Get the cached tasks specification."""
        return self._tasks_spec
    
    def reload(self) -> None:
        """
        Reload configurations from disk.
        This should only be called when configurations have changed
        and you need to refresh the cache (e.g., during development).
        """
        logger.info("ConfigLoader: Reloading configurations from disk")
        self._load_configs()
    
    @classmethod
    def get_instance(cls) -> 'ConfigLoader':
        """Get the singleton instance of ConfigLoader."""
        return cls()


# Create a module-level instance for easy import
_config_loader = ConfigLoader()


def get_agents_spec() -> Dict[str, Any]:
    """Get the cached agents specification."""
    return _config_loader.agents_spec


def get_tasks_spec() -> Dict[str, Any]:
    """Get the cached tasks specification."""
    return _config_loader.tasks_spec


def reload_configs() -> None:
    """
    Reload configurations from disk.
    Use this sparingly, only when you need to refresh configs during runtime.
    """
    _config_loader.reload()


# For backward compatibility with prompt_runner.py
AGENTS_SPEC = get_agents_spec()
TASKS_SPEC = get_tasks_spec()