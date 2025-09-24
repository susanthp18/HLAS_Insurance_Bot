import os
import sys
import argparse
import logging
from datetime import datetime
from pymongo import MongoClient, errors
from dotenv import load_dotenv

def setup_logging(log_level="INFO"):
    """
    Sets up production-level logging with both file and console handlers.
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logger
    logger = logging.getLogger('mongo_init')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(logs_dir, f'mongo_init_{timestamp}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Create error file handler for errors and above
    error_log_filename = os.path.join(logs_dir, f'mongo_init_errors_{timestamp}.log')
    error_handler = logging.FileHandler(error_log_filename)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized. Log file: {log_filename}")
    logger.info(f"Error log file: {error_log_filename}")
    
    return logger

def validate_environment(logger):
    """
    Validates required environment variables and configuration.
    
    Args:
        logger: Configured logger instance
        
    Returns:
        tuple: (mongo_uri, db_name) if valid, (None, None) if invalid
    """
    logger.info("Starting environment validation")
    
    # Load environment variables
    try:
        load_dotenv()
        logger.debug("Environment variables loaded from .env file")
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
    
    # Validate MONGO_URI
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        logger.error("MONGO_URI environment variable not set")
        return None, None
    
    logger.info("MONGO_URI found in environment")
    # Log masked URI for security (show only the protocol and host)
    try:
        if mongo_uri.startswith("mongodb"):
            uri_parts = mongo_uri.split("@")
            if len(uri_parts) > 1:
                masked_uri = f"mongodb://***@{uri_parts[-1]}"
            else:
                masked_uri = mongo_uri.split("://")[0] + "://***"
            logger.debug(f"Using MongoDB URI: {masked_uri}")
    except Exception as e:
        logger.warning(f"Could not mask URI for logging: {e}")
    
    # Get database name
    db_name = os.getenv("DB_NAME")
    db_name = db_name.lower()
    logger.info(f"Using database name: '{db_name}'")
    
    return mongo_uri, db_name

def test_mongodb_connection(mongo_uri, logger):
    """
    Tests MongoDB connection and returns client if successful.
    
    Args:
        mongo_uri: MongoDB connection string
        logger: Configured logger instance
        
    Returns:
        MongoClient instance or None if connection fails
    """
    logger.info("Testing MongoDB connection...")
    
    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=20000           # 20 second socket timeout
        )
        
        # Test connection by pinging the server
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB server")
        
        # Log server info
        server_info = client.server_info()
        logger.info(f"MongoDB server version: {server_info.get('version', 'Unknown')}")
        
        return client
        
    except errors.ServerSelectionTimeoutError as e:
        logger.error(f"Could not connect to MongoDB server (timeout): {e}")
        return None
    except errors.ConnectionFailure as e:
        logger.error(f"MongoDB connection failure: {e}")
        return None
    except errors.ConfigurationError as e:
        logger.error(f"MongoDB configuration error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to MongoDB: {e}")
        return None

def handle_reset_collections(db, logger):
    """
    Handles the collection reset process with user confirmation.
    
    Args:
        db: MongoDB database instance
        logger: Configured logger instance
        
    Returns:
        bool: True if reset was performed, False if cancelled
    """
    logger.warning("Reset flag detected - collections will be dropped")
    
    try:
        # Get collection stats before dropping
        sessions_collection = db["sessions"]
        conversation_history_collection = db["conversation_history"]
        
        sessions_count = sessions_collection.count_documents({})
        history_count = conversation_history_collection.count_documents({})
        
        logger.info(f"Current 'sessions' collection document count: {sessions_count}")
        logger.info(f"Current 'conversation_history' collection document count: {history_count}")
        
        # Get user confirmation
        print(f"\n⚠️  WARNING: This will permanently delete:")
        print(f"   - {sessions_count} documents from 'sessions' collection")
        print(f"   - {history_count} documents from 'conversation_history' collection")
        
        confirm = input("\nAre you sure you want to drop these collections? (yes/no): ")
        
        if confirm.lower() == 'yes':
            logger.info("User confirmed collection reset")
            
            # Drop sessions collection
            logger.info("Dropping 'sessions' collection...")
            sessions_collection.drop()
            logger.info("Successfully dropped 'sessions' collection")
            
            # Drop conversation_history collection
            logger.info("Dropping 'conversation_history' collection...")
            conversation_history_collection.drop()
            logger.info("Successfully dropped 'conversation_history' collection")
            
            return True
        else:
            logger.info("User cancelled reset operation")
            print("Reset operation cancelled.")
            return False
            
    except Exception as e:
        logger.error(f"Error during collection reset: {e}")
        raise

def create_collection_index(collection, index_spec, index_name, unique=False, logger=None):
    """
    Creates an index on a collection with proper error handling and logging.
    
    Args:
        collection: MongoDB collection instance
        index_spec: Index specification (field name or complex spec)
        index_name: Human-readable name for logging
        unique: Whether the index should be unique
        logger: Configured logger instance
        
    Returns:
        bool: True if index was created successfully, False otherwise
    """
    try:
        if unique:
            result = collection.create_index(index_spec, unique=True)
            logger.info(f"Created unique index on '{index_name}': {result}")
        else:
            result = collection.create_index(index_spec)
            logger.info(f"Created index on '{index_name}': {result}")
        return True
        
    except errors.OperationFailure as e:
        if "already exists" in str(e).lower():
            logger.debug(f"Index on '{index_name}' already exists")
        else:
            logger.warning(f"Could not create index on '{index_name}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating index on '{index_name}': {e}")
        return False

def initialize_collections(db, logger):
    """
    Initializes collections and creates necessary indexes.
    
    Args:
        db: MongoDB database instance
        logger: Configured logger instance
    """
    logger.info("Starting collection initialization")
    
    sessions_collection = db["sessions"]
    conversation_history_collection = db["conversation_history"]
    
    # Initialize sessions collection
    logger.info("Initializing 'sessions' collection...")
    
    sessions_success = 0
    sessions_success += create_collection_index(
        sessions_collection, "session_id", "session_id", unique=True, logger=logger
    )
    sessions_success += create_collection_index(
        sessions_collection, "last_active", "last_active", logger=logger
    )
    
    logger.info(f"'sessions' collection initialized ({sessions_success}/2 indexes created)")
    
    # Initialize conversation_history collection
    logger.info("Initializing 'conversation_history' collection...")
    
    history_success = 0
    history_success += create_collection_index(
        conversation_history_collection, "session_id", "session_id", logger=logger
    )
    history_success += create_collection_index(
        conversation_history_collection, "timestamp", "timestamp", logger=logger
    )
    
    logger.info(f"'conversation_history' collection initialized ({history_success}/2 indexes created)")
    
    # Log collection information
    try:
        db_stats = db.command("dbStats")
        logger.info(f"Database '{db.name}' statistics:")
        logger.info(f"  - Collections: {db_stats.get('collections', 'Unknown')}")
        logger.info(f"  - Data size: {db_stats.get('dataSize', 'Unknown')} bytes")
        logger.info(f"  - Index size: {db_stats.get('indexSize', 'Unknown')} bytes")
    except Exception as e:
        logger.warning(f"Could not retrieve database statistics: {e}")

def main():
    """
    Main function that orchestrates the MongoDB initialization process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize or reset MongoDB collections.")
    parser.add_argument("--reset", action="store_true", help="Drop existing collections before initializing.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level (default: INFO)")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("="*60)
    logger.info("MongoDB Initialization Script Started")
    logger.info("="*60)
    logger.info(f"Script arguments: reset={args.reset}, log_level={args.log_level}")
    
    client = None
    exit_code = 0
    
    try:
        # Validate environment
        mongo_uri, db_name = validate_environment(logger)
        if not mongo_uri or not db_name:
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Test MongoDB connection
        client = test_mongodb_connection(mongo_uri, logger)
        if not client:
            logger.error("Could not establish MongoDB connection")
            sys.exit(1)
        
        # Get database
        db = client[db_name]
        logger.info(f"Connected to database: '{db_name}'")
        
        # Handle reset if requested
        if args.reset:
            if not handle_reset_collections(db, logger):
                logger.info("Script terminated by user")
                return
        
        # Initialize collections
        initialize_collections(db, logger)
        
        logger.info("MongoDB initialization completed successfully")
        print("\n✅ MongoDB initialization completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user (Ctrl+C)")
        print("\n⚠️ Script interrupted by user")
        exit_code = 130
        
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Check the log files for detailed information.")
        exit_code = 1
        
    finally:
        if client:
            try:
                client.close()
                logger.info("MongoDB connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")
        
        logger.info("="*60)
        logger.info(f"MongoDB Initialization Script Completed (exit code: {exit_code})")
        logger.info("="*60)
        
        if exit_code != 0:
            sys.exit(exit_code)

if __name__ == "__main__":
    main()