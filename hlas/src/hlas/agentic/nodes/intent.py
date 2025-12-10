from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import AgentState, IntentPrediction
from ..config import _router_model
from ..utils.memory import _build_history_context_from_messages, _get_last_user_message

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are an intent router for the HLAS insurance chatbot.

You will see the latest user message plus a short summary of recent turns and
the current known product (if any).

Decide what the assistant should do NEXT, without generating the final user
reply. You only choose a high-level intent and (optionally) a product.

INTENT OPTIONS
- info: user asks about coverage, benefits, limits, exclusions, claims, or
  any other factual detail for one product.
- summary: user wants a concise overview of what a product or plan covers.
- compare: user wants differences between tiers/plans of the same product.
- recommend: user asks which plan/tier is suitable for them, or shares life
  events that clearly create an insurance need (for example: upcoming travel,
  hiring a maid, buying a car or house, having a baby) where it is natural for
  the assistant to suggest a suitable plan.
- purchase: user expresses intent to buy, get a quote, or asks for a link to
  purchase a plan now.
- capabilities: user asks what this bot can do, what products it supports, or
  meta-questions about the assistant.
- greet: very short greeting like "hi", "hello", "hey", "good morning" when
  there is no clear insurance question yet.
- chat: small-talk or open conversation such as "how are you", "I am going on
  holiday", or users sharing life context without a clear factual question.
  In chat, the assistant should respond in a warm, human way, and can gently
  offer relevant insurance help if appropriate.
- other: anything that does not fit the above.

Always be conservative: if you are unsure between summary/compare/info, pick
"info". If the message is only a greeting or thanks, choose "greet". If the
user is mainly making small-talk or sharing personal context, choose "chat".
If they mention a life event that obviously benefits from insurance (like
international travel or a new house) and seem open to help, choose
"recommend" with the most relevant product.
"""

_router_structured = _router_model.with_structured_output(IntentPrediction)

def _classify_intent_from_messages(
    messages: List[BaseMessage],
    known_product: Optional[str],
) -> IntentPrediction:
    """Use AzureChatOpenAI + structured output to classify the next action.

    This is a small LangChain-style runnable: ChatPromptTemplate | structured LLM.
    """

    message = _get_last_user_message(messages)
    history_ctx = _build_history_context_from_messages(messages[:-1], max_pairs=2)
    product = (known_product or "").strip()

    ctx_lines: List[str] = []
    if product:
        ctx_lines.append(f"CURRENT_PRODUCT: {product}")
    if history_ctx:
        ctx_lines.append("RECENT_TURNS (most recent last):")
        ctx_lines.append(history_ctx)
    ctx_lines.append("")
    ctx_lines.append(f"LATEST_MESSAGE: {message}")
    context_text = "\n".join(ctx_lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("user", "{context}"),
        ]
    )

    try:
        result = (prompt | _router_structured).invoke({"context": context_text})
        if isinstance(result, IntentPrediction):
            return result
        # Fallback: best-effort cast
        return IntentPrediction.model_validate(result)
    except Exception as e:
        logger.warning("Agentic intent routing failed, falling back to 'info': %s", e)
        return IntentPrediction(intent="info", product=product or None, reason="fallback_error")


def _route_from_intent(state: AgentState) -> str:
    """Routing function for conditional edges from supervisor."""

    intent = (state.get("intent") or "info").strip().lower()
    if intent == "reflect_done":
        return "reflect_done"
    if intent == "greet":
        return "greet"
    if intent == "capabilities":
        return "capabilities"
    if intent == "chat":
        return "chat"
    if intent == "summary":
        return "summary"
    if intent == "compare":
        return "compare"
    if intent == "recommend":
        return "recommend"
    if intent == "purchase":
        return "purchase"
    if intent == "other":
        return "info"
    return "info"
