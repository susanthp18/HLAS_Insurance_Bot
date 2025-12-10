from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..state import AgentState
from ..config import _router_model
from ..utils.memory import _build_history_context_from_messages

logger = logging.getLogger(__name__)


def _compress_memory_node(state: AgentState) -> AgentState:
    """Optional memory compression node.

    When many messages accumulate and no summary has been added yet, append a
    brief AIMessage summarising earlier conversation. This keeps context
    focused while preserving the full history.
    """

    if state.get("has_summary"):
        return {}
    messages = list(state.get("messages", []) or [])
    if len(messages) <= 10:
        return {}

    # Summarize all but the last few messages to keep recent turns verbatim.
    head = messages[:-4]
    if not head:
        return {}

    history_text = _build_history_context_from_messages(head, max_pairs=5)
    if not history_text:
        return {}

    sys_msg = (
        "You summarize earlier parts of a conversation so that the assistant "
        "can stay focused. Write a concise summary capturing key facts, "
        "user preferences, and decisions."
    )
    user_msg = "Here is the earlier conversation to summarize:\n" + history_text
    try:
        s_msg = _router_model.invoke(
            [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]
        )
        summary = str(getattr(s_msg, "content", "") or "").strip()
    except Exception as e:
        logger.warning("Agentic.compress: summarization failed: %s", e)
        summary = ""

    if not summary:
        return {}

    logger.info(
        "Agentic.memory.compress: added summary for %d earlier messages (chars=%d)",
        len(head),
        len(summary),
    )

    summary_message = AIMessage(
        content="Summary of earlier conversation: " + summary
    )
    return {"messages": [summary_message], "has_summary": True}
