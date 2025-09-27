import os
import logging
from langchain_openai import AzureOpenAIEmbeddings
from crewai import LLM
from dotenv import load_dotenv, find_dotenv

# Load .env from project tree (searches upward from CWD)
load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)

# Strict, centralized Azure OpenAI config (no fallbacks)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
# Optional model behavior tuning
AZURE_OPENAI_TEMPERATURE_STR = os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2")
# Response generation model configuration (can be set to different deployment)
AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_OPENAI_RESPONSE_TEMPERATURE_STR = os.environ.get("AZURE_OPENAI_RESPONSE_TEMPERATURE", "0.3")


# Initialize as None at the module level
azure_llm = None
azure_embeddings = None
azure_response_llm = None  # Separate LLM for response generation


def initialize_models():
    """
    Initializes and configures the LLM and embedding models.
    This function is idempotent - it will only initialize once.
    """
    global azure_llm, azure_embeddings, azure_response_llm
    
    # Skip if already initialized (idempotent)
    if azure_llm is not None and azure_embeddings is not None and azure_response_llm is not None:
        logger.debug("Models already initialized, skipping re-initialization")
        return
    
    # Use log_once to prevent duplicate logs across workers
    try:
        from .log_once import log_once_info
        log_once_info(logger, "llm_init_start", "Initializing LLM and embedding models...")
    except (ImportError, Exception):
        logger.info("Initializing LLM and embedding models...")

    # Check required environment variables
    missing_vars = []
    if not AZURE_OPENAI_ENDPOINT:
        missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing_vars.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
        missing_vars.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
        missing_vars.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {missing_vars}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Resolve temperature
        try:
            _temperature = float(AZURE_OPENAI_TEMPERATURE_STR)
        except (ValueError, TypeError):
            _temperature = 0.2
            logger.warning("Invalid or missing AZURE_OPENAI_TEMPERATURE, defaulting to %.1f", _temperature)

        # Create LLM instance
        globals()["azure_llm"] = LLM(
            model=f"azure/{AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}",
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_ENDPOINT.rstrip("/"),
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=_temperature,
        )
        # Use log_once to prevent duplicate logs across workers
        try:
            from .log_once import log_once_info
            log_once_info(logger, "llm_init_success", "Azure LLM initialized successfully.")
        except (ImportError, Exception):
            logger.info("Azure LLM initialized successfully.")

        # Create Response Generation LLM instance (GPT-4-mini)
        try:
            _response_temperature = float(AZURE_OPENAI_RESPONSE_TEMPERATURE_STR)
        except (ValueError, TypeError):
            _response_temperature = 0.3
            logger.warning("Invalid or missing AZURE_OPENAI_RESPONSE_TEMPERATURE, defaulting to %.1f", _response_temperature)
        
        globals()["azure_response_llm"] = LLM(
            model=f"azure/{AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME}",
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_ENDPOINT.rstrip("/"),
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=_response_temperature,
        )
        # Use log_once to prevent duplicate logs across workers
        try:
            from .log_once import log_once_info
            log_once_info(logger, "response_llm_init_success", f"Azure Response LLM ({AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME}) initialized successfully.")
        except (ImportError, Exception):
            logger.info(f"Azure Response LLM ({AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME}) initialized successfully.")

        # Create Embeddings instance
        globals()["azure_embeddings"] = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )
        # Use log_once to prevent duplicate logs across workers
        try:
            from .log_once import log_once_info
            log_once_info(logger, "embeddings_init_success", "Azure Embeddings initialized successfully.")
        except (ImportError, Exception):
            logger.info("Azure Embeddings initialized successfully.")

    except Exception as e:
        logger.error("Failed to initialize models: %s", e, exc_info=True)
        raise  # Re-raise the exception to halt application startup


__all__ = ["azure_llm", "azure_embeddings", "azure_response_llm", "initialize_models"]
