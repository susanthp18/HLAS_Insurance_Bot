"""
Infrastructure module for production-grade agentic chatbot.
Provides Redis session management, MongoDB history persistence, metrics, LangGraph checkpointer,
LLM initialization, and vector store client.
"""

from .redis_utils import (
    get_redis,
    RedisLock,
    SessionCache,
    RateLimiter,
    Deduplicator,
    OrderGuard,
    session_lock_key,
)
from .session import SessionManager
from .mongo_history import log_history, get_history, clear_history
from .redis_checkpointer import RedisCheckpointer
from .llm import (
    initialize_models,
    get_chat_llm,
    get_response_llm,
    get_embeddings,
    cleanup as llm_cleanup,
)
from .vector_store import get_weaviate_client, close_weaviate_client, WEAVIATE_AVAILABLE
from .metrics import (
    AGENTIC_MESSAGES_TOTAL,
    AGENTIC_LATENCY,
    SESSION_CACHE_HITS,
    SESSION_CACHE_MISSES,
    WA_MESSAGES_PROCESSED_TOTAL,
    LIVE_AGENT_HANDOFFS,
    POLICY_VIOLATIONS,
    REDIS_LOCK_TIMEOUTS,
)

__all__ = [
    # Redis utilities
    "get_redis",
    "RedisLock",
    "SessionCache",
    "RateLimiter",
    "Deduplicator",
    "OrderGuard",
    "session_lock_key",
    # Session management
    "SessionManager",
    # MongoDB history
    "log_history",
    "get_history",
    "clear_history",
    # LangGraph checkpointer
    "RedisCheckpointer",
    # LLM (thread-safe singletons)
    "initialize_models",
    "get_chat_llm",
    "get_response_llm",
    "get_embeddings",
    "llm_cleanup",
    # Vector store
    "get_weaviate_client",
    "close_weaviate_client",
    "WEAVIATE_AVAILABLE",
    # Metrics
    "AGENTIC_MESSAGES_TOTAL",
    "AGENTIC_LATENCY",
    "SESSION_CACHE_HITS",
    "SESSION_CACHE_MISSES",
    "WA_MESSAGES_PROCESSED_TOTAL",
    "LIVE_AGENT_HANDOFFS",
    "POLICY_VIOLATIONS",
    "REDIS_LOCK_TIMEOUTS",
]
