import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

# Prefer Singapore timezone for all session timestamps
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:
    # Fallback if zoneinfo unavailable
    SGT_TZ = timezone(timedelta(hours=8))

# Metrics and Redis-backed cache
from .metrics import SESSION_CACHE_HITS, SESSION_CACHE_MISSES
from .redis_utils import SessionCache
from .mongo_history import log_history

# Idle session reset threshold (seconds). If exceeded, we reset the session state.
SESSION_IDLE_RESET_SECONDS = int(os.getenv("SESSION_IDLE_RESET_SECONDS", os.getenv("SESSION_CACHE_TTL_SECONDS", "900")))

logger = logging.getLogger(__name__)

DEFAULT_SESSION_FIELDS = {
    "product": None,
    "slots": {},
    "recommended_tier": None,
    "live_agent_status": False,
}


class SessionManager:
    """
    Redis-only session manager. Stores the full session state (including a small
    rolling conversation history) in Redis as JSON. No MongoDB dependency.
    """
    _instance = None
    _cache: 'SessionCache' = None  # Set in __new__

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            # Initialize Redis cache (mandatory)
            cls._cache = SessionCache()
            logger.info("Session cache initialized (Redis-only session management)")
        return cls._instance

    def __init__(self):
        pass

    def _new_session(self, session_id: str, now: datetime) -> Dict[str, Any]:
        base = {
            "session_id": session_id,
            "history": [],
            "created_at": now,
            "last_active": now,
        }
        base.update(DEFAULT_SESSION_FIELDS)
        return base

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Fetches a session from Redis; creates one if missing.
        Performs idle reset if last_active is older than SESSION_IDLE_RESET_SECONDS.
        """
        now = datetime.now(SGT_TZ)
        cached = self._cache.get(session_id)
        if cached:
            logger.info("Loaded session %s from cache.", session_id)
            SESSION_CACHE_HITS.inc()
            session_data = cached
        else:
            SESSION_CACHE_MISSES.inc()
            logger.info("No session found for %s. Creating a new one (Redis-only).", session_id)
            session_data = self._new_session(session_id, now)
            self._cache.set(session_id, session_data)

        # Idle reset check
        try:
            last_active = session_data.get("last_active")
            # Normalize last_active to timezone-aware Singapore time
            if isinstance(last_active, str):
                try:
                    last_active = datetime.fromisoformat(last_active)
                except Exception:
                    last_active = None
            if isinstance(last_active, datetime):
                try:
                    if last_active.tzinfo is None:
                        last_active = last_active.replace(tzinfo=SGT_TZ)
                    else:
                        last_active = last_active.astimezone(SGT_TZ)
                except Exception:
                    last_active = None
            if last_active and (now - last_active) > timedelta(seconds=SESSION_IDLE_RESET_SECONDS):
                logger.info(
                    "Idle reset: session %s inactive for > %ds. Resetting state (fresh session, clearing history).",
                    session_id,
                    SESSION_IDLE_RESET_SECONDS,
                )
                # Start a fresh session (do not preserve prior history)
                reset_state = {**DEFAULT_SESSION_FIELDS, "session_id": session_id, "last_active": now}
                to_save = dict(reset_state)
                to_save["history"] = []
                # Preserve original creation time if available
                if session_data.get("created_at"):
                    to_save["created_at"] = session_data["created_at"]
                else:
                    to_save["created_at"] = now
                self.save_session(session_id, to_save)
                return to_save
        except Exception as e:
            logger.warning("Idle reset check failed for %s: %s", session_id, e)

        return session_data

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """
        Persist the session state in Redis. History should be included in the payload.
        """
        if not session_data:
            logger.warning("Attempted to save empty session data for %s.", session_id)
            return
        # Update last_active and persist
        session_state = dict(session_data)
        session_state["last_active"] = datetime.now(SGT_TZ)
        self._cache.set(session_id, session_state)
        logger.info("Saved session state for %s in Redis.", session_id)

    def add_history_entry(self, session_id: str, user_message: str, bot_response: str):
        """
        Adds a new user-bot interaction to the in-session history and updates cache (keep last 5).
        """
        ts = datetime.now(SGT_TZ)
        cached = self._cache.get(session_id) or self._new_session(session_id, ts)
        hist = cached.get("history", [])
        hist.append({
            "session_id": session_id,
            "timestamp": ts.isoformat(),
            "user": user_message,
            "assistant": bot_response,
        })
        if len(hist) > 5:
            hist = hist[-5:]
        cached["history"] = hist
        cached["last_active"] = ts
        self._cache.set(session_id, cached)
        # Append this turn to MongoDB (no-op if Mongo not configured)
        try:
            log_history(session_id=session_id, user_message=user_message, assistant_message=bot_response, ts=ts)
        except Exception as e:
            logger.warning("Mongo history logging failed for session %s: %s", session_id, e)
        logger.info("Added history entry for session %s.", session_id)

    def reset_session(self, session_id: str):
        """
        Reset the session state to defaults and clear conversation history.
        """
        now = datetime.now(SGT_TZ)
        cached = self._cache.get(session_id) or self._new_session(session_id, now)
        new_state = {**DEFAULT_SESSION_FIELDS, "session_id": session_id, "history": [], "created_at": cached.get("created_at", now), "last_active": now}
        self._cache.set(session_id, new_state)
        logger.info("Reset session state for %s with fresh history.", session_id)

    def close_connection(self):
        """
        No-op for Redis-only session manager; included for API compatibility.
        """
        logger.info("SessionManager.close_connection called (no-op for Redis).")

# Backwards-compatible alias to avoid refactoring all imports
MongoSessionManager = SessionManager
