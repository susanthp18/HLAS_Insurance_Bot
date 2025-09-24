import os
import sys
import logging
import logging.handlers
from dotenv import load_dotenv

def setup_logging():
    """
    Configures the 'hlas' package logger to output to the console and a rotating file.
    This setup is controlled by environment variables.
    """
    load_dotenv()

    if os.getenv("LOGGING_ENABLED", "false").lower() != "true":
        return

    log_file = os.getenv("LOG_FILE", "logs/app.log")
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # --- Configure only the 'hlas' logger ---
    hlas_logger = logging.getLogger('hlas')
    hlas_logger.setLevel(log_level)
    
    # Prevent log messages from being passed to the root logger
    hlas_logger.propagate = False

    # Clear any existing handlers to prevent duplicates
    if hlas_logger.hasHandlers():
        hlas_logger.handlers.clear()

    # Create a shared formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    hlas_logger.addHandler(console_handler)

    # --- Rotating File Handler ---
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        hlas_logger.addHandler(file_handler)
    except Exception as e:
        hlas_logger.error(f"Failed to set up file logging for '{log_file}': {e}", exc_info=True)

    hlas_logger.info(f"Logging for 'hlas' package initialized. Level: {log_level_name}. Log file: {log_file}")
