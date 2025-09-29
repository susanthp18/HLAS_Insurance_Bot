import os
import logging
from typing import Optional
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:  # pragma: no cover
    SGT_TZ = timezone(timedelta(hours=8))

try:
    from pymongo import MongoClient
except Exception as e:  # pragma: no cover
    MongoClient = None  # type: ignore

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME") or ""
DB_NAME = DB_NAME.lower() if isinstance(DB_NAME, str) else ""

_logger = logging.getLogger(__name__)
_client = None
_db = None
_inited = False


def _init_if_needed() -> None:
    global _inited, _client, _db
    if _inited:
        return
    _inited = True
    # Graceful no-op if pymongo or env is not available
    if MongoClient is None:
        _logger.warning("Mongo history writer: pymongo not available; history persistence disabled.")
        return
    if not MONGO_URI or not DB_NAME:
        _logger.warning("Mongo history writer: MONGO_URI/DB_NAME not set; history persistence disabled.")
        return
    try:
        _client = MongoClient(MONGO_URI, tz_aware=True)
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        _logger.info("Mongo history writer initialized for DB '%s'", DB_NAME)
    except Exception as e:
        _logger.error("Mongo history writer initialization failed: %s", e)
        _client = None
        _db = None


def log_history(session_id: str, user_message: str, assistant_message: str, ts: Optional[datetime] = None) -> None:
    """Append a conversation turn to MongoDB conversation_history collection.

    - No-op if Mongo is not configured or unavailable.
    - Uses 'timestamp' as a tz-aware datetime (Asia/Singapore).
    - This function is append-only and does not mutate session state.
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
        _db["conversation_history"].insert_one(doc)
    except Exception as e:  # pragma: no cover
        _logger.warning("Mongo history writer: failed to insert conversation turn: %s", e)
