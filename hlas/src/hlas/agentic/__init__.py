from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage

from .state import AgentState
from .graph import get_agent_graph, _memory_saver

# Local greeting (avoid cross-module dependency)
GREETING_MESSAGE = (
    "Hello! I'm the HLAS Smart Bot. I'm here to guide you through our insurance products and services, "
    "answer your questions instantly, and make things easier for you. How can I help you today?"
)

from .utils.slots import _detect_product_llm

logger = logging.getLogger(__name__)

# Response pattern that indicates bot is handing off to live agent
# The master prompt instructs the bot to say this exact phrase
LIVE_AGENT_HANDOFF_PHRASE = "connect you with a live agent"

def _is_live_agent_response(response: str) -> bool:
    """Check if bot response indicates live agent handoff."""
    if not response:
        return False
    return LIVE_AGENT_HANDOFF_PHRASE in response.lower()


import asyncio
from .nodes.policy_validator import check_policy

# ...

async def agentic_chat(session_id: str, message: str) -> Dict[str, Any]:
    """Entrypoint for the new LangGraph-based agentic chatbot.

    This does not touch the legacy /chat or WhatsApp flows and maintains its
    own LangGraph state keyed by session_id (thread_id).
    """

    # Entry log with truncated message preview
    msg_preview = (message or "").replace("\n", " ")[:160]
    logger.info(
        "Agentic.chat.start: session=%s msg='%s'",
        session_id,
        msg_preview,
    )

    # Special-case "hi" greeting to reset conversation state, mirroring /chat.
    if (message or "").strip().lower() == "hi":
        # Reset logic
        try:
            # Try standard clear if supported
            # If delete_thread exists, it likely takes config. But if it fails with unhashable dict,
            # it implies implementation differences. We'll try update_state to clear messages instead.
            
            # Attempt to clear history by updating state with empty messages list
            # This is the standard LangGraph way to "reset" or modify past state.
            # Note: This appends/overwrites depending on reducer. For 'messages' (add_messages), 
            # we usually can't easily clear without a custom reducer or checkpoint hack.
            # But for this specific user request ("reset"), we can just generate a new thread ID 
            # effectively by handling it at the client/session manager level (which we do in /chat).
            # Here in /agent-chat, we rely on session_id.
            
            # Let's try passing the string session_id directly to delete_thread if config dict failed.
            if hasattr(_memory_saver, "delete_thread"):
                 try:
                     _memory_saver.delete_thread({"configurable": {"thread_id": session_id}})
                 except TypeError:
                     # Fallback: try passing session_id as string
                     _memory_saver.delete_thread(session_id)
            else:
                # If we can't delete, we just proceed. The ReAct agent handles context windowing anyway.
                pass
        except Exception as e:
            logger.warning("Agentic.chat: failed to reset LangGraph thread for 'hi' - %s", e)
        
        logger.info("Agentic.chat.hi_reset: session=%s", session_id)
        return {"response": GREETING_MESSAGE, "sources": "", "debug_state": {}}

    # Run Policy Check AND Graph Execution in Parallel
    config = {"configurable": {"thread_id": session_id}}
    agent_graph = get_agent_graph()
    
    try:
        # Get history for context (optional, but helpful for policy check)
        history_snapshot = await agent_graph.aget_state(config)
        history_msgs = history_snapshot.values.get("messages", [])
        historical_product = history_snapshot.values.get("product")

        # Prefer product detected from the current message; fall back to history if needed.
        detected_product = None
        try:
            detected_product = _detect_product_llm(message)
        except Exception:
            detected_product = None
        product = detected_product or historical_product

        logger.debug(
            "Agentic.chat.history: session=%s messages=%d product=%s",
            session_id,
            len(history_msgs or []),
            product,
        )
        # Simple string history for query generator
        history_str = "\n".join([f"{m.type}: {m.content}" for m in history_msgs[-4:]])

        # Launch both tasks concurrently
        policy_task = asyncio.create_task(check_policy(message, history_str, product))
        graph_task = asyncio.create_task(agent_graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        ))
        
        # Wait for both (but we can process policy result immediately if needed)
        is_violation, policy_reply = await policy_task
        
        if is_violation:
            # Cancel graph task if it's still running, as we don't need its output
            graph_task.cancel()
            reply_preview = (policy_reply or "").replace("\n", " ")[:200]
            logger.warning(
                "Agentic.chat.policy_block: session=%s preview='%s'",
                session_id,
                reply_preview,
            )
            return {"response": policy_reply, "sources": "", "debug_state": {"violation": True}}
            
        # If no violation, wait for graph result
        result: AgentState = await graph_task
        
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Agentic.chat: graph invocation failed, returning fallback")
        fallback = (
            "Something went wrong while processing your request. "
            "Please try rephrasing your question or ask about a specific "
            "HLAS product such as Travel, Maid, Car, Personal Accident, "
            "Home, Fraud, Hospital or Early."
        )
        return {"response": fallback, "sources": "", "debug_state": {}}
    
    # ... existing result processing
    


    messages = result.get("messages", []) or []
    reply = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            reply = str(getattr(m, "content", "") or "").strip()
            if reply:
                break
    if not reply:
        reply = (
            "I’m not sure I understood that. Could you clarify what you’d like "
            "to know about our insurance plans?"
        )

    sources_val = result.get("sources") or []
    if isinstance(sources_val, str):
        sources_str = sources_val
    else:
        sources_str = "\n".join(str(s) for s in sources_val if s)

    # Detect live agent handoff from bot response
    live_agent_requested = _is_live_agent_response(reply)
    
    debug_state = {
        "intent": result.get("intent"),
        "product": result.get("product"),
        "rec_ready": result.get("rec_ready", False),
        "live_agent_requested": live_agent_requested,
    }

    reply_preview = reply.replace("\n", " ")[:200]
    logger.info(
        "Agentic.chat.completed: intent=%s product=%s live_agent=%s reply_len=%d preview='%s'",
        debug_state.get("intent"),
        debug_state.get("product"),
        live_agent_requested,
        len(reply),
        reply_preview,
    )
    return {"response": reply, "sources": sources_str, "debug_state": debug_state}

__all__ = ["agentic_chat"]
