"""
Production-grade session management for the agentic chatbot.
Uses Redis for session state and integrates with MongoDB for history persistence.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

try:
    from zoneinfo import ZoneInfo
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:
    SGT_TZ = timezone(timedelta(hours=8))

from .redis_utils import SessionCache
from .mongo_history import log_history
from .metrics import SESSION_CACHE_HITS, SESSION_CACHE_MISSES

SESSION_IDLE_RESET_SECONDS = int(os.getenv("SESSION_IDLE_RESET_SECONDS", os.getenv("SESSION_CACHE_TTL_SECONDS", "900")))

logger = logging.getLogger(__name__)

DEFAULT_SESSION_FIELDS = {
    "product": None,
    "slots": {},
    "recommended_tier": None,
    "live_agent_status": False,
    "intent": None,
}


class SessionManager:
    """
    Redis-backed session manager for the agentic chatbot.
    Stores session state in Redis and logs conversation history to MongoDB.
    """
    _instance = None
    _cache: Optional[SessionCache] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._cache = SessionCache(prefix="agentic:session")
            logger.info("Agentic SessionManager initialized (Redis-backed)")
        return cls._instance

    def _new_session(self, session_id: str, now: datetime) -> Dict[str, Any]:
        base = {
            "session_id": session_id,
            "history": [],
            "created_at": now.isoformat(),
            "last_active": now.isoformat(),
        }
        base.update(DEFAULT_SESSION_FIELDS)
        return base

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Fetch session from Redis; create if missing.
        Performs idle reset if session exceeds idle timeout.
        """
        now = datetime.now(SGT_TZ)
        cached = self._cache.get(session_id)
        
        if cached:
            SESSION_CACHE_HITS.inc()
            session_data = cached
        else:
            SESSION_CACHE_MISSES.inc()
            logger.info("Creating new agentic session: %s", session_id)
            session_data = self._new_session(session_id, now)
            self._cache.set(session_id, session_data)
            return session_data

        # Idle reset check
        try:
            last_active = session_data.get("last_active")
            if isinstance(last_active, str):
                try:
                    last_active = datetime.fromisoformat(last_active)
                except Exception:
                    last_active = None
            if isinstance(last_active, datetime):
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=SGT_TZ)
                else:
                    last_active = last_active.astimezone(SGT_TZ)
                    
                if (now - last_active) > timedelta(seconds=SESSION_IDLE_RESET_SECONDS):
                    logger.info("Idle reset: session %s inactive > %ds", session_id, SESSION_IDLE_RESET_SECONDS)
                    reset_state = self._new_session(session_id, now)
                    if session_data.get("created_at"):
                        reset_state["created_at"] = session_data["created_at"]
                    self._cache.set(session_id, reset_state)
                    return reset_state
        except Exception as e:
            logger.warning("Idle reset check failed for %s: %s", session_id, e)

        return session_data

    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Persist session state to Redis."""
        if not session_data:
            logger.warning("Attempted to save empty session data for %s", session_id)
            return
        session_state = dict(session_data)
        session_state["last_active"] = datetime.now(SGT_TZ).isoformat()
        self._cache.set(session_id, session_state)
        logger.debug("Saved agentic session: %s", session_id)

    def add_history_entry(
        self, 
        session_id: str, 
        user_message: str, 
        bot_response: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Add conversation turn to in-session history (keep last 5) and persist to MongoDB.
        """
        ts = datetime.now(SGT_TZ)
        cached = self._cache.get(session_id) or self._new_session(session_id, ts)
        
        # Update in-session history (rolling window)
        hist = cached.get("history", [])
        hist.append({
            "timestamp": ts.isoformat(),
            "user": user_message,
            "assistant": bot_response[:200] if len(bot_response) > 200 else bot_response,
        })
        if len(hist) > 5:
            hist = hist[-5:]
        cached["history"] = hist
        cached["last_active"] = ts.isoformat()
        self._cache.set(session_id, cached)
        
        # Persist full conversation to MongoDB
        try:
            log_history(
                session_id=session_id,
                user_message=user_message,
                assistant_message=bot_response,
                ts=ts,
                metadata=metadata
            )
        except Exception as e:
            logger.warning("MongoDB history logging failed for session %s: %s", session_id, e)

    def reset_session(self, session_id: str) -> None:
        """Reset session to defaults and clear history."""
        now = datetime.now(SGT_TZ)
        cached = self._cache.get(session_id)
        new_state = self._new_session(session_id, now)
        if cached and cached.get("created_at"):
            new_state["created_at"] = cached["created_at"]
        self._cache.set(session_id, new_state)
        logger.info("Reset agentic session: %s", session_id)

    def delete_session(self, session_id: str) -> None:
        """Completely remove session from Redis."""
        self._cache.delete(session_id)
        logger.info("Deleted agentic session: %s", session_id)

    def update_field(self, session_id: str, field: str, value: Any) -> None:
        """Update a single field in the session."""
        session = self.get_session(session_id)
        session[field] = value
        self.save_session(session_id, session)

    def is_live_agent_active(self, session_id: str) -> bool:
        """Check if live agent mode is active for this session."""
        session = self.get_session(session_id)
        val = session.get("live_agent_status")
        if isinstance(val, str):
            return val.strip().lower() in ("on", "true", "yes", "1")
        return bool(val)

    def set_live_agent_status(self, session_id: str, status: bool) -> None:
        """Set live agent status for this session."""
        self.update_field(session_id, "live_agent_status", status)
