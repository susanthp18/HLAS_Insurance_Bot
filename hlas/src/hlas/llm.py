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


# Initialize as None at the module level
azure_llm = None
azure_embeddings = None


def initialize_models():
    """
    Initializes and configures the LLM and embedding models.
    """
    global azure_llm, azure_embeddings
    
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
        azure_llm = LLM(
            model=f"azure/{AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}",
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_ENDPOINT.rstrip("/"),
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=_temperature,
        )
        logger.info("Azure LLM initialized successfully.")

        # Create Embeddings instance
        azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )
        logger.info("Azure Embeddings initialized successfully.")

    except Exception as e:
        logger.error("Failed to initialize models: %s", e, exc_info=True)
        raise  # Re-raise the exception to halt application startup


__all__ = ["azure_llm", "azure_embeddings", "initialize_models"]