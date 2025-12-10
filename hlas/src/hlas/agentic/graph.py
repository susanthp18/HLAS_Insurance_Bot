from __future__ import annotations

import os
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes.memory_nodes import _compress_memory_node
from .nodes.master import master_agent_node

logger = logging.getLogger(__name__)

_graph_builder: StateGraph[AgentState] = StateGraph(AgentState)

# Nodes
_graph_builder.add_node("compress", _compress_memory_node)
_graph_builder.add_node("master_agent", master_agent_node)

# Edges
_graph_builder.add_edge(START, "compress")
_graph_builder.add_edge("compress", "master_agent")
_graph_builder.add_edge("master_agent", END)

# Persistence - Use Redis checkpointer in production, fallback to MemorySaver
_checkpointer = None

def _get_checkpointer():
    """Get the appropriate checkpointer based on environment."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    
    use_redis = os.getenv("AGENTIC_USE_REDIS_CHECKPOINTER", "true").lower() in ("true", "1", "yes")
    
    if use_redis:
        try:
            from .infrastructure.redis_checkpointer import RedisCheckpointer
            _checkpointer = RedisCheckpointer()
            logger.info("Agentic using Redis checkpointer for conversation persistence")
            return _checkpointer
        except Exception as e:
            logger.warning(f"Failed to initialize Redis checkpointer, falling back to MemorySaver: {e}")
    
    _checkpointer = MemorySaver()
    logger.info("Agentic using in-memory checkpointer (MemorySaver)")
    return _checkpointer

# Lazy initialization - compile graph when first accessed
_agent_graph = None
_memory_saver = None  # Keep for backwards compatibility

def get_agent_graph():
    """Get the compiled agent graph with appropriate checkpointer."""
    global _agent_graph, _memory_saver
    if _agent_graph is None:
        _memory_saver = _get_checkpointer()
        _agent_graph = _graph_builder.compile(checkpointer=_memory_saver)
    return _agent_graph
