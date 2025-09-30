import os
import logging
from langchain_openai import AzureOpenAIEmbeddings
from crewai import LLM
from dotenv import load_dotenv, find_dotenv

# Load .env from project tree (searches upward from CWD)
load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)

# Provider toggle (default: gpt)
LLM_PROVIDER = (os.environ.get("LLM_PROVIDER", "gpt") or "gpt").strip().lower()

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

# Grok (OpenAI-compatible) config
GROK_OPENAI_ENDPOINT = os.environ.get(
    "GROK_OPENAI_ENDPOINT",
    "https://grokmodl-resource.services.ai.azure.com/openai/v1/",
)
GROK_OPENAI_API_KEY = os.environ.get("GROK_OPENAI_API_KEY")
GROK_OPENAI_DEPLOYMENT_NAME = os.environ.get(
    "GROK_OPENAI_DEPLOYMENT_NAME",
    "grok-4-fast-non-reasoning",
)
GROK_OPENAI_RESPONSE_DEPLOYMENT_NAME = os.environ.get(
    "GROK_OPENAI_RESPONSE_DEPLOYMENT_NAME",
    GROK_OPENAI_DEPLOYMENT_NAME,
)
GROK_OPENAI_TEMPERATURE_STR = os.environ.get("GROK_OPENAI_TEMPERATURE", "0.2")
GROK_OPENAI_RESPONSE_TEMPERATURE_STR = os.environ.get("GROK_OPENAI_RESPONSE_TEMPERATURE", "0.3")


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

    # Check required environment variables depending on provider
    missing_vars = []
    if LLM_PROVIDER == "grok":
        # Grok chat LLM requirements
        if not GROK_OPENAI_ENDPOINT:
            missing_vars.append("GROK_OPENAI_ENDPOINT")
        if not GROK_OPENAI_API_KEY:
            missing_vars.append("GROK_OPENAI_API_KEY")
        if not GROK_OPENAI_DEPLOYMENT_NAME:
            missing_vars.append("GROK_OPENAI_DEPLOYMENT_NAME")
        # We still initialize Azure embeddings for retrieval; require Azure embedding env too
        if not AZURE_OPENAI_ENDPOINT:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_KEY:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_API_VERSION:
            missing_vars.append("AZURE_OPENAI_API_VERSION")
        if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
            missing_vars.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    else:
        # Default GPT/Azure requirements (chat + embeddings)
        if not AZURE_OPENAI_ENDPOINT:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_KEY:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_API_VERSION:
            missing_vars.append("AZURE_OPENAI_API_VERSION")
        if not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
            missing_vars.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
            missing_vars.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    
    if missing_vars:
        error_msg = f"Missing required environment variables ({LLM_PROVIDER}): {missing_vars}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        if LLM_PROVIDER == "grok":
            # Resolve temperatures
            try:
                _temperature = float(GROK_OPENAI_TEMPERATURE_STR)
            except (ValueError, TypeError):
                _temperature = 0.2
                logger.warning("Invalid or missing GROK_OPENAI_TEMPERATURE, defaulting to %.1f", _temperature)
            try:
                _response_temperature = float(GROK_OPENAI_RESPONSE_TEMPERATURE_STR)
            except (ValueError, TypeError):
                _response_temperature = 0.3
                logger.warning("Invalid or missing GROK_OPENAI_RESPONSE_TEMPERATURE, defaulting to %.1f", _response_temperature)

            # Initialize chat LLMs using Grok/OpenAI-compatible endpoint
            globals()["azure_llm"] = LLM(
                model=f"openai/{GROK_OPENAI_DEPLOYMENT_NAME}",
                api_key=GROK_OPENAI_API_KEY,
                base_url=GROK_OPENAI_ENDPOINT.rstrip("/"),
                temperature=_temperature,
            )
            try:
                from .log_once import log_once_info
                log_once_info(logger, "llm_init_success", f"Grok LLM initialized: {GROK_OPENAI_DEPLOYMENT_NAME}")
            except (ImportError, Exception):
                logger.info("Grok LLM initialized: %s", GROK_OPENAI_DEPLOYMENT_NAME)

            globals()["azure_response_llm"] = LLM(
                model=f"openai/{GROK_OPENAI_RESPONSE_DEPLOYMENT_NAME}",
                api_key=GROK_OPENAI_API_KEY,
                base_url=GROK_OPENAI_ENDPOINT.rstrip("/"),
                temperature=_response_temperature,
            )
            try:
                from .log_once import log_once_info
                log_once_info(logger, "response_llm_init_success", f"Grok Response LLM initialized: {GROK_OPENAI_RESPONSE_DEPLOYMENT_NAME}")
            except (ImportError, Exception):
                logger.info("Grok Response LLM initialized: %s", GROK_OPENAI_RESPONSE_DEPLOYMENT_NAME)

            # Always initialize Azure embeddings (used by retrieval)
            globals()["azure_embeddings"] = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                openai_api_version=AZURE_OPENAI_API_VERSION,
            )
            try:
                from .log_once import log_once_info
                log_once_info(logger, "embeddings_init_success", "Azure Embeddings initialized successfully (Grok provider).")
            except (ImportError, Exception):
                logger.info("Azure Embeddings initialized successfully (Grok provider).")
        else:
            # Resolve temperature
            try:
                _temperature = float(AZURE_OPENAI_TEMPERATURE_STR)
            except (ValueError, TypeError):
                _temperature = 0.2
                logger.warning("Invalid or missing AZURE_OPENAI_TEMPERATURE, defaulting to %.1f", _temperature)

            # Create LLM instance (Azure GPT)
            globals()["azure_llm"] = LLM(
                model=f"azure/{AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}",
                api_key=AZURE_OPENAI_API_KEY,
                base_url=AZURE_OPENAI_ENDPOINT.rstrip("/"),
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=_temperature,
            )
            try:
                from .log_once import log_once_info
                log_once_info(logger, "llm_init_success", "Azure LLM initialized successfully.")
            except (ImportError, Exception):
                logger.info("Azure LLM initialized successfully.")

            # Create Response Generation LLM instance
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
            try:
                from .log_once import log_once_info
                log_once_info(logger, "response_llm_init_success", f"Azure Response LLM ({AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME}) initialized successfully.")
            except (ImportError, Exception):
                logger.info(f"Azure Response LLM ({AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME}) initialized successfully.")

            # Create Embeddings instance (Azure)
            globals()["azure_embeddings"] = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                openai_api_version=AZURE_OPENAI_API_VERSION,
            )
            try:
                from .log_once import log_once_info
                log_once_info(logger, "embeddings_init_success", "Azure Embeddings initialized successfully.")
            except (ImportError, Exception):
                logger.info("Azure Embeddings initialized successfully.")

    except Exception as e:
        logger.error("Failed to initialize models: %s", e, exc_info=True)
        raise  # Re-raise the exception to halt application startup


__all__ = ["azure_llm", "azure_embeddings", "azure_response_llm", "initialize_models"]
