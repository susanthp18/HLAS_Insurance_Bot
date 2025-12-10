"""
MongoDB conversation history persistence for the agentic chatbot.
Standalone copy for the agentic module.
"""

import os
import logging
from typing import Optional
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:
    SGT_TZ = timezone(timedelta(hours=8))

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "").lower()
COLLECTION_NAME = os.getenv("AGENTIC_HISTORY_COLLECTION", "agentic_conversation_history")

_logger = logging.getLogger(__name__)
_client = None
_db = None
_inited = False


def _init_if_needed() -> None:
    global _inited, _client, _db
    if _inited:
        return
    _inited = True
    
    if MongoClient is None:
        _logger.warning("Agentic Mongo history: pymongo not available; history persistence disabled.")
        return
    if not MONGO_URI or not DB_NAME:
        _logger.warning("Agentic Mongo history: MONGO_URI/DB_NAME not set; history persistence disabled.")
        return
    try:
        _client = MongoClient(MONGO_URI, tz_aware=True)
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        _logger.info("Agentic Mongo history initialized for DB '%s', collection '%s'", DB_NAME, COLLECTION_NAME)
    except Exception as e:
        _logger.error("Agentic Mongo history initialization failed: %s", e)
        _client = None
        _db = None


def log_history(
    session_id: str, 
    user_message: str, 
    assistant_message: str, 
    ts: Optional[datetime] = None,
    metadata: Optional[dict] = None
) -> None:
    """
    Append a conversation turn to MongoDB agentic_conversation_history collection.
    
    Args:
        session_id: Unique session identifier
        user_message: User's message
        assistant_message: Bot's response
        ts: Timestamp (defaults to now in Singapore timezone)
        metadata: Optional dict with additional info (product, intent, etc.)
    """
    _init_if_needed()
    if _db is None:
        return
    try:
        if ts is None:
            ts = datetime.now(SGT_TZ)
        doc = {
            "session_id": session_id,
            "timestamp": ts,
            "user": user_message,
            "assistant": assistant_message,
        }
        if metadata:
            doc["metadata"] = metadata
        _db[COLLECTION_NAME].insert_one(doc)
    except Exception as e:
        _logger.warning("Agentic Mongo history: failed to insert conversation turn: %s", e)


def get_history(session_id: str, limit: int = 20) -> list:
    """
    Retrieve conversation history for a session from MongoDB.
    
    Args:
        session_id: Unique session identifier
        limit: Maximum number of turns to retrieve
        
    Returns:
        List of conversation turns, oldest first
    """
    _init_if_needed()
    if _db is None:
        return []
    try:
        cursor = _db[COLLECTION_NAME].find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit)
        return list(cursor)
    except Exception as e:
        _logger.warning("Agentic Mongo history: failed to retrieve history: %s", e)
        return []


def clear_history(session_id: str) -> None:
    """
    Clear conversation history for a session from MongoDB.
    """
    _init_if_needed()
    if _db is None:
        return
    try:
        _db[COLLECTION_NAME].delete_many({"session_id": session_id})
        _logger.info("Agentic Mongo history: cleared history for session %s", session_id)
    except Exception as e:
        _logger.warning("Agentic Mongo history: failed to clear history: %s", e)
