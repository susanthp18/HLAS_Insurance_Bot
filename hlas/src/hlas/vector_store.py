import os
import logging
import weaviate
from typing import Optional
from urllib.parse import urlparse
from weaviate.auth import AuthApiKey
import weaviate.classes as wvc

logger = logging.getLogger(__name__)

# Global Weaviate client instance
_weaviate_client = None

def get_weaviate_client():
    """
    Get a singleton Weaviate client instance.
    Suppresses noisy httpx logs and disables Weaviate version check to avoid PyPI calls.
    """
    global _weaviate_client
    if _weaviate_client is None:
        try:
            # Suppress httpx INFO logs to reduce noise
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)
            
            # Disable Weaviate version check to avoid unnecessary PyPI calls
            os.environ["WEAVIATE_SKIP_INIT_CHECKS"] = "true"
            
            parsed_url = urlparse(os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_ENDPOINT") or "http://localhost:8080")
            auth_credentials = None
            if os.getenv("WEAVIATE_API_KEY"):
                auth_credentials = AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

            # Use the correct gRPC port that matches your Docker container
            _weaviate_client = weaviate.connect_to_custom(
                http_host=parsed_url.hostname,
                http_port=parsed_url.port or 8080,
                http_secure=parsed_url.scheme == "https",
                grpc_host=parsed_url.hostname,
                grpc_port=50051,  # This matches your Docker container's exposed gRPC port
                grpc_secure=False,
                auth_credentials=auth_credentials,
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=30),  # Increase timeout to 30 seconds
                ),
                skip_init_checks=True  # Skip initialization checks including version validation
            )            
            logger.info("Successfully connected to Weaviate (logs suppressed, version check disabled).")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    return _weaviate_client

def close_weaviate_client():
    """
    Close the Weaviate client connection.
    """
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
            logger.info("Weaviate client connection closed.")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
        finally:
            _weaviate_client = None