"""
LLM Configuration and Initialization for Agentic Chatbot
========================================================

Production-optimized LLM management with:
- Thread-safe singleton initialization
- Connection pooling via httpx
- Embedding cache for repeated queries
- Lazy initialization on first access
"""

import os
import logging
import threading
from typing import Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import httpx

load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)

# ============================================
# Configuration (loaded once at module import)
# ============================================

# Provider toggle (default: azure)
LLM_PROVIDER = (os.environ.get("LLM_PROVIDER", "azure") or "azure").strip().lower()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")

# Embeddings Configuration
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT") or AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_EMBEDDING_API_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY") or AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBEDDING_API_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION") or AZURE_OPENAI_API_VERSION
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

# Response LLM Configuration
AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME", "gpt-4o-mini")

# Temperature settings
AZURE_OPENAI_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2"))
AZURE_OPENAI_RESPONSE_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_RESPONSE_TEMPERATURE", "0.3"))

# Connection pool settings
HTTP_POOL_SIZE = int(os.environ.get("AGENTIC_HTTP_POOL_SIZE", "100"))
HTTP_TIMEOUT = float(os.environ.get("AGENTIC_HTTP_TIMEOUT", "30.0"))



# ============================================
# Thread-safe Singleton Management
# ============================================

_init_lock = threading.Lock()
_chat_llm: Optional[AzureChatOpenAI] = None
_response_llm: Optional[AzureChatOpenAI] = None
_embeddings: Optional[AzureOpenAIEmbeddings] = None
_http_client: Optional[httpx.Client] = None
_async_http_client: Optional[httpx.AsyncClient] = None
_initialized = False


def _get_http_client() -> httpx.Client:
    """Get shared sync HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=HTTP_POOL_SIZE,
                max_keepalive_connections=HTTP_POOL_SIZE // 2,
            ),
            timeout=httpx.Timeout(HTTP_TIMEOUT),
        )
    return _http_client


def _get_async_http_client() -> httpx.AsyncClient:
    """Get shared async HTTP client with connection pooling."""
    global _async_http_client
    if _async_http_client is None:
        _async_http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=HTTP_POOL_SIZE,
                max_keepalive_connections=HTTP_POOL_SIZE // 2,
            ),
            timeout=httpx.Timeout(HTTP_TIMEOUT),
        )
    return _async_http_client


def _validate_config() -> None:
    """Validate required environment variables."""
    missing = []
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
        missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
        missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")


def initialize_models() -> None:
    """
    Initialize LLM and embedding models. Thread-safe and idempotent.
    Uses double-checked locking for optimal performance.
    """
    global _chat_llm, _response_llm, _embeddings, _initialized
    
    # Fast path: already initialized
    if _initialized:
        return
    
    # Slow path: acquire lock and initialize
    with _init_lock:
        # Double-check after acquiring lock
        if _initialized:
            return
        
        _validate_config()
        
        try:
            # Chat LLM (for routing, intent detection)
            # Using max_retries and request_timeout for resilience
            _chat_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                temperature=AZURE_OPENAI_TEMPERATURE,
                max_retries=2,
                request_timeout=HTTP_TIMEOUT,
            )
            logger.info("Agentic Chat LLM initialized: %s", AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
            
            # Response LLM (for generating user-facing responses)
            _response_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME,
                temperature=AZURE_OPENAI_RESPONSE_TEMPERATURE,
                max_retries=2,
                request_timeout=HTTP_TIMEOUT,
            )
            logger.info("Agentic Response LLM initialized: %s", AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME)
            
            # Embeddings with chunking for batch efficiency
            _embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
                api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
                api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
                azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                chunk_size=16,  # Batch embeddings for efficiency
            )
            logger.info("Agentic Embeddings initialized: %s", AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)
            
            _initialized = True
            logger.info("All agentic LLM models initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize agentic LLM models: %s", e)
            raise


def get_chat_llm() -> AzureChatOpenAI:
    """Get the chat LLM instance. Thread-safe lazy initialization."""
    if not _initialized:
        initialize_models()
    return _chat_llm


def get_response_llm() -> AzureChatOpenAI:
    """Get the response LLM instance. Thread-safe lazy initialization."""
    if not _initialized:
        initialize_models()
    return _response_llm


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Get the embeddings instance. Thread-safe lazy initialization."""
    if not _initialized:
        initialize_models()
    return _embeddings


# ============================================
# Cleanup
# ============================================

def cleanup() -> None:
    """Cleanup resources on shutdown."""
    global _http_client, _async_http_client
    
    if _http_client:
        _http_client.close()
        _http_client = None
    
    if _async_http_client:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_async_http_client.aclose())
            else:
                loop.run_until_complete(_async_http_client.aclose())
        except Exception:
            pass
        _async_http_client = None
    
    logger.info("LLM resources cleaned up")


__all__ = [
    "initialize_models",
    "get_chat_llm",
    "get_response_llm", 
    "get_embeddings",
    "cleanup",
]
