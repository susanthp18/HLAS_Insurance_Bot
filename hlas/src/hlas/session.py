import os
import logging
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
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

# Load environment variables for MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
DB_NAME = DB_NAME.lower()

# Idle session reset threshold (seconds). If exceeded, we reset the session state.
SESSION_IDLE_RESET_SECONDS = int(os.getenv("SESSION_IDLE_RESET_SECONDS", os.getenv("SESSION_CACHE_TTL_SECONDS", "900")))

logger = logging.getLogger(__name__)

class MongoSessionManager:
    """
    Manages session and conversation history data in MongoDB with Redis caching.
    """
    _instance = None
    _client = None
    _db = None
    _cache: 'SessionCache' = None  # Set in __new__

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoSessionManager, cls).__new__(cls)
            try:
                # Use tz_aware=True so Mongo returns timezone-aware datetimes
                cls._client = MongoClient(MONGO_URI, tz_aware=True)
                # The ismaster command is cheap and does not require auth.
                cls._client.admin.command('ismaster')
                cls._db = cls._client[DB_NAME]
                logger.info("Successfully connected to MongoDB.")
            except ConnectionFailure as e:
                logger.error("Could not connect to MongoDB: %s", e)
                raise
            # Initialize Redis cache (mandatory)
            cls._cache = SessionCache()
            logger.info("Session cache initialized (Redis)")
        return cls._instance

    def __init__(self):
        """
        Initializes the session manager. The connection is established in __new__.
        """
        pass

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Fetches a session and its conversation history from cache or database.

        If the session does not exist, it returns a new, empty session structure.
        Also performs idle reset if last_active is older than SESSION_IDLE_RESET_SECONDS.
        """
        try:
            now = datetime.now(SGT_TZ)

            # Try cache first (do not return early; we may need to idle-reset)
            cached = self._cache.get(session_id)
            if cached:
                logger.info("Loaded session %s from cache.", session_id)
                SESSION_CACHE_HITS.inc()
                session_data = cached
            else:
                # Cache miss -> load from DB
                SESSION_CACHE_MISSES.inc()
                session_data = self._db.sessions.find_one({"session_id": session_id})
                
                history_cursor = self._db.conversation_history.find(
                    {"session_id": session_id},
                    {"_id": 0}
                ).sort("timestamp", -1).limit(5)

                history = list(history_cursor)
                history.reverse()

                if session_data:
                    session_data.pop("_id", None)
                    session_data['history'] = history
                    logger.info("Loaded session %s from DB.", session_id)
                else:
                    logger.info("No session found for %s. Creating a new one.", session_id)
                    session_data = {
                        "session_id": session_id,
                        "product": None,
                        "slots": {},
                        "recommended_tier": None,
                        "history": history,
                        "created_at": now,
                        "last_active": now,
                    }
                # Store initial in cache
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
                    logger.info("Idle reset: session %s inactive for > %ds. Resetting state.", session_id, SESSION_IDLE_RESET_SECONDS)
                    # Preserve history
                    history = session_data.get("history", [])
                    # Reset fields
                    reset_state = {
                        "session_id": session_id,
                        "product": None,
                        "slots": {},
                        "recommended_tier": None,
                        "last_active": now,
                    }
                    # Save and cache
                    to_save = dict(reset_state)
                    to_save["history"] = history
                    self.save_session(session_id, to_save)
                    # Return fully reset session with history
                    return to_save
            except Exception as e:
                logger.warning("Idle reset check failed for %s: %s", session_id, e)

            return session_data
        except OperationFailure as e:
            logger.error("Error fetching session %s: %s", session_id, e)
            return {
                "session_id": session_id,
                "history": [],
                "error": str(e)
            }

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """
        Saves the session state to the database and updates cache.
        The history is saved via add_history_entry.
        """
        if not session_data:
            logger.warning("Attempted to save empty session data for %s.", session_id)
            return

        try:
            import time
            start = time.time()
            
            # Keep history out of DB 'sessions' document, but preserve it for cache
            history = session_data.pop("history", [])
            
            session_state = session_data.copy()
            session_state["last_active"] = datetime.now(SGT_TZ)

            # The 'created_at' field should only be set when the document is inserted.
            # Remove it from the session_state to avoid a conflict with $setOnInsert.
            session_state.pop('created_at', None)

            self._db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": session_state,
                    "$setOnInsert": {"created_at": datetime.now(SGT_TZ)}
                },
                upsert=True,
                hint="session_id_1"  # Use index hint if available
            )
            
            elapsed = time.time() - start
            logger.info("Saved session state for %s in %.2fs.", session_id, elapsed)

            # Update cache copy
            cached = self._cache.get(session_id) or {}
            cached.update(session_state)
            if history:
                cached["history"] = history
            else:
                # Preserve existing cached history
                cached.setdefault("history", [])
            self._cache.set(session_id, cached)
        except OperationFailure as e:
            logger.error("Error saving session %s: %s", session_id, e)
            raise

    def add_history_entry(self, session_id: str, user_message: str, bot_response: str):
        """
        Adds a new user-bot interaction to the conversation history and updates cached history.
        """
        try:
            import time
            start = time.time()
            ts = datetime.now(SGT_TZ)
            history_entry = {
                "session_id": session_id,
                "timestamp": ts,
                "user": user_message,
                "assistant": bot_response
            }
            
            # Batch both operations to reduce round-trips
            from pymongo import InsertOne, UpdateOne
            operations = [
                InsertOne(history_entry),
            ]
            self._db.conversation_history.bulk_write(operations, ordered=False)
            
            # Update last_active separately (lighter operation)
            self._db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"last_active": ts}},
                hint="session_id_1"  # Use index hint if available
            )
            
            elapsed = time.time() - start
            logger.info("Added history entry for session %s in %.2fs.", session_id, elapsed)

            # Update cached history (keep last 5)
            cached = self._cache.get(session_id)
            if cached is not None:
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
                self._cache.set(session_id, cached)
        except OperationFailure as e:
            logger.error("Error adding history for session %s: %s", session_id, e)
            raise

    def reset_session(self, session_id: str):
        """
        Reset the session state to defaults while preserving the conversation history
        stored in the `conversation_history` collection. Also invalidates cache.
        """
        try:
            existing = self._db.sessions.find_one({"session_id": session_id}, {"created_at": 1})
            created_at = existing.get("created_at") if existing else datetime.now(SGT_TZ)

            fields_to_unset = {
                "comparison_status": "",
                "summary_status": "",
                "comparison_slot": "",
                "comparison_history": "",
                "summary_slot": "",
                "summary_history": "",
                "recommendation_status": "",
                "last_question": "",
                "_last_info_prod_q": "",
                "_last_info_user_msg": "",
                "_fu_query": "",
                "pending_slot": "",
                "last_completed": ""
            }

            update_body = {
                "$set": {
                    "product": None,
                    "slots": {},
                    "recommended_tier": None,
                    "last_active": datetime.now(SGT_TZ)
                },
                "$unset": fields_to_unset
            }

            if not existing:
                update_body["$setOnInsert"] = {
                    "session_id": session_id,
                    "created_at": created_at
                }

            self._db.sessions.update_one({"session_id": session_id}, update_body, upsert=True)
            logger.info("Reset session state for %s while preserving conversation history.", session_id)

            # Invalidate cache
            self._cache.invalidate(session_id)
        except OperationFailure as e:
            logger.error("Error resetting session %s: %s", session_id, e)
            raise

    def close_connection(self):
        """
        Closes the MongoDB connection.
        """
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed.")