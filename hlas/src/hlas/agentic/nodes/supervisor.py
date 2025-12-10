from __future__ import annotations

from langchain_core.messages import AIMessage

from ..state import AgentState
from .feedback import _classify_feedback_from_messages, _self_critique_and_rewrite_from_messages
from .intent import _classify_intent_from_messages

def _supervisor_node(state: AgentState) -> AgentState:
    """Supervisor node: intent routing + negative feedback handling."""

    messages = list(state.get("messages", []) or [])
    if not messages:
        return {}

    known_product = state.get("product")

    # Negative feedback handling / reflection
    feedback = _classify_feedback_from_messages(messages)
    if feedback and feedback.category == "negative_feedback":
        revised = _self_critique_and_rewrite_from_messages(messages)
        if revised:
            return {
                "messages": [AIMessage(content=revised)],
                "feedback": "negative_feedback",
                "sources": [],
                "intent": "reflect_done",
            }

    # Intent classification
    intent_pred = _classify_intent_from_messages(messages, known_product)
    raw_intent = (intent_pred.intent or "").strip().lower()
    allowed_intents = {
        "info",
        "summary",
        "compare",
        "recommend",
        "purchase",
        "capabilities",
        "greet",
        "chat",
        "other",
    }
    if raw_intent not in allowed_intents:
        normalized_intent = "info"
    else:
        normalized_intent = raw_intent

    product = intent_pred.product or known_product

    return {
        "intent": normalized_intent,
        "product": product,
    }
