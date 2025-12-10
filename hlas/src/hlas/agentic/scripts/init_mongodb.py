#!/usr/bin/env python3
"""
MongoDB Initialization Script for Agentic Chatbot
=================================================

This script initializes the MongoDB database with required collections
and indexes for the agentic chatbot conversation history.

Usage:
    python -m hlas.agentic.scripts.init_mongodb

Or directly:
    python init_mongodb.py
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import CollectionInvalid
except ImportError:
    logger.error("pymongo is required. Install with: pip install pymongo")
    sys.exit(1)


def init_mongodb():
    """Initialize MongoDB collections and indexes for the agentic chatbot."""
    
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("DB_NAME", "hlas").lower()
    history_collection = os.getenv("AGENTIC_HISTORY_COLLECTION", "agentic_conversation_history")
    
    logger.info("Connecting to MongoDB: %s", mongo_uri)
    
    try:
        client = MongoClient(mongo_uri, tz_aware=True)
        client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error("Failed to connect to MongoDB: %s", e)
        sys.exit(1)
    
    db = client[db_name]
    logger.info("Using database: %s", db_name)
    
    # 1. Create conversation history collection
    logger.info("Setting up collection: %s", history_collection)
    
    try:
        db.create_collection(history_collection)
        logger.info("Created collection: %s", history_collection)
    except CollectionInvalid:
        logger.info("Collection already exists: %s", history_collection)
    
    collection = db[history_collection]
    
    # 2. Create indexes
    logger.info("Creating indexes...")
    
    # Index on session_id for fast lookups
    collection.create_index(
        [("session_id", ASCENDING)],
        name="idx_session_id"
    )
    logger.info("Created index: idx_session_id")
    
    # Compound index on session_id + timestamp for retrieving history in order
    collection.create_index(
        [("session_id", ASCENDING), ("timestamp", DESCENDING)],
        name="idx_session_timestamp"
    )
    logger.info("Created index: idx_session_timestamp")
    
    # Index on timestamp for cleanup/archival operations
    collection.create_index(
        [("timestamp", DESCENDING)],
        name="idx_timestamp"
    )
    logger.info("Created index: idx_timestamp")
    
    # Optional: TTL index to auto-expire old conversations (90 days)
    ttl_days = int(os.getenv("AGENTIC_HISTORY_TTL_DAYS", "90"))
    if ttl_days > 0:
        collection.create_index(
            [("timestamp", ASCENDING)],
            name="idx_ttl",
            expireAfterSeconds=ttl_days * 24 * 60 * 60
        )
        logger.info("Created TTL index: idx_ttl (expires after %d days)", ttl_days)
    
    # 3. Print collection stats
    stats = db.command("collstats", history_collection)
    logger.info("Collection stats: documents=%d, size=%d bytes", 
                stats.get("count", 0), stats.get("size", 0))
    
    # 4. Insert a test document to verify write access
    test_doc = {
        "session_id": "_init_test_",
        "timestamp": datetime.utcnow(),
        "user": "MongoDB initialization test",
        "assistant": "Initialization successful",
        "metadata": {"script": "init_mongodb.py"}
    }
    result = collection.insert_one(test_doc)
    logger.info("Test document inserted: %s", result.inserted_id)
    
    # Clean up test document
    collection.delete_one({"session_id": "_init_test_"})
    logger.info("Test document cleaned up")
    
    logger.info("=" * 50)
    logger.info("MongoDB initialization completed successfully!")
    logger.info("Database: %s", db_name)
    logger.info("Collection: %s", history_collection)
    logger.info("=" * 50)
    
    client.close()


if __name__ == "__main__":
    init_mongodb()
