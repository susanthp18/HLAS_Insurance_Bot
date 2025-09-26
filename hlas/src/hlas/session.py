import os
import logging
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

# Load environment variables for MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
DB_NAME = DB_NAME.lower()

logger = logging.getLogger(__name__)

class MongoSessionManager:
    """
    Manages session and conversation history data in MongoDB.
    """
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoSessionManager, cls).__new__(cls)
            try:
                cls._client = MongoClient(MONGO_URI)
                # The ismaster command is cheap and does not require auth.
                cls._client.admin.command('ismaster')
                cls._db = cls._client[DB_NAME]
                logger.info("Successfully connected to MongoDB.")
            except ConnectionFailure as e:
                logger.error("Could not connect to MongoDB: %s", e)
                raise
        return cls._instance

    def __init__(self):
        """
        Initializes the session manager. The connection is established in __new__.
        """
        pass

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Fetches a session and its conversation history from the database.

        If the session does not exist, it returns a new, empty session structure.
        """
        try:
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
                return session_data
            else:
                logger.info("No session found for %s. Creating a new one.", session_id)
                return {
                    "session_id": session_id,
            "product": None,
            "slots": {},
            "recommended_tier": None,
                    "history": history,
                    "created_at": datetime.now(timezone.utc),
                    "last_active": datetime.now(timezone.utc)
                }
        except OperationFailure as e:
            logger.error("Error fetching session %s: %s", session_id, e)
            return {
                "session_id": session_id,
                "history": [],
                "error": str(e)
            }

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """
        Saves the session state to the database.
        The history is saved via add_history_entry.
        """
        if not session_data:
            logger.warning("Attempted to save empty session data for %s.", session_id)
            return

        try:
            session_data.pop("history", [])
            
            session_state = session_data.copy()
            session_state["last_active"] = datetime.now(timezone.utc)

            # The 'created_at' field should only be set when the document is inserted.
            # Remove it from the session_state to avoid a conflict with $setOnInsert.
            session_state.pop('created_at', None)

            self._db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": session_state,
                    "$setOnInsert": {"created_at": datetime.now(timezone.utc)}
                },
                upsert=True
            )
            logger.info("Saved session state for %s.", session_id)
        except OperationFailure as e:
            logger.error("Error saving session %s: %s", session_id, e)
            raise

    def add_history_entry(self, session_id: str, user_message: str, bot_response: str):
        """
        Adds a new user-bot interaction to the conversation history.
        """
        try:
            history_entry = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc),
                "user": user_message,
                "assistant": bot_response
            }
            self._db.conversation_history.insert_one(history_entry)
            logger.info("Added history entry for session %s.", session_id)
            
            self._db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"last_active": datetime.now(timezone.utc)}}
            )
        except OperationFailure as e:
            logger.error("Error adding history for session %s: %s", session_id, e)
            raise

    def reset_session(self, session_id: str):
        """
        Reset the session state to defaults while preserving the conversation history
        stored in the `conversation_history` collection.
        """
        try:
            existing = self._db.sessions.find_one({"session_id": session_id}, {"created_at": 1})
            created_at = existing.get("created_at") if existing else datetime.now(timezone.utc)

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
                    "last_active": datetime.now(timezone.utc)
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