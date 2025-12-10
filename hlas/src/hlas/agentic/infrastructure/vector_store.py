"""
Vector Store (Weaviate) Client for Agentic Chatbot
==================================================

Provides Weaviate client for semantic search / RAG operations.
This is a standalone copy to avoid external dependencies.
"""

import os
import logging
from typing import Optional
from urllib.parse import urlparse

try:
    import weaviate
    from weaviate.auth import AuthApiKey
    import weaviate.classes as wvc
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None
    AuthApiKey = None
    wvc = None

logger = logging.getLogger(__name__)

# Global Weaviate client instance
_weaviate_client = None


def get_weaviate_client():
    """
    Get a singleton Weaviate client instance.
    Suppresses noisy logs and disables version checks.
    """
    global _weaviate_client
    
    if not WEAVIATE_AVAILABLE:
        logger.warning("Weaviate package not installed. RAG features disabled.")
        return None
    
    if _weaviate_client is not None:
        return _weaviate_client
    
    try:
        # Suppress httpx INFO logs
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.WARNING)
        
        # Disable version check
        os.environ["WEAVIATE_SKIP_INIT_CHECKS"] = "true"
        
        weaviate_url = os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_ENDPOINT") or "http://localhost:8080"
        parsed_url = urlparse(weaviate_url)
        
        auth_credentials = None
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        if weaviate_api_key:
            auth_credentials = AuthApiKey(api_key=weaviate_api_key)
        
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        _weaviate_client = weaviate.connect_to_custom(
            http_host=parsed_url.hostname,
            http_port=parsed_url.port or 8080,
            http_secure=parsed_url.scheme == "https",
            grpc_host=parsed_url.hostname,
            grpc_port=grpc_port,
            grpc_secure=False,
            auth_credentials=auth_credentials,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=30),
            ),
            skip_init_checks=True,
        )
        logger.info("Agentic Weaviate client connected: %s", weaviate_url)
        
    except Exception as e:
        logger.error("Failed to connect to Weaviate: %s", e)
        raise
    
    return _weaviate_client


def close_weaviate_client():
    """Close the Weaviate client connection."""
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
            logger.info("Weaviate client connection closed.")
        except Exception as e:
            logger.error("Error closing Weaviate client: %s", e)
        finally:
            _weaviate_client = None


__all__ = ["get_weaviate_client", "close_weaviate_client", "WEAVIATE_AVAILABLE"]
