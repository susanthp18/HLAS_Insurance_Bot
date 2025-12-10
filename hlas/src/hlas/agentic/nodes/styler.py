from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from ..state import AgentState
from ..config import _router_model, _load_knowledge_base
from ..utils.memory import _build_history_context_from_messages, _get_last_user_message

logger = logging.getLogger(__name__)


def _style_reply_node(state: AgentState) -> AgentState:
    """Final styling/orchestration node to make replies feel more autonomous.

    It takes the draft reply from previous agents/tools plus a short history
    summary and rewrites the answer in a more human, conversational and
    capability-aware way.
    """

    messages = list(state.get("messages", []) or [])
    if not messages:
        return {}

    # Find the latest assistant reply to rewrite/polish.
    draft = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            draft = m
            break
    if not draft:
        return {}

    user_text = _get_last_user_message(messages) or ""
    history_ctx = _build_history_context_from_messages(messages[:-1], max_pairs=3)
    kb_text = _load_knowledge_base() or ""

    intent = (state.get("intent") or "").strip().lower()
    product = (state.get("product") or "").strip()

    sys_prompt = """You are HLAS's autonomous digital insurance assistant.

You see the recent conversation, the latest user message, your own draft
reply (generated from internal tools/flows), and a short description of what
you are capable of.

Your goals:
- Respond in a warm, natural, human way – never sound like a rigid script.
- Be explicitly aware of your capabilities: you can explain, summarise and
  compare HLAS Travel, Maid, Car, Home, Personal Accident, Early Critical
  Illness, Fraud (Protect360) and Hospital Cash plans; you can recommend
  suitable plans; and you can share purchase/next-step guidance.
- If the user shares life context that naturally links to insurance (e.g.
  travelling abroad, buying a car or home, starting a family, worrying about
  scams or hospital costs), gently highlight relevant insurance options and
  ask if they would like help choosing a plan.
- If the user asks about something outside your knowledge, be honest about
  limits and steer the conversation back to how you can help with HLAS
  products.
- Keep replies concise, friendly and focused on actually helping the user.
"""

    kb_section = kb_text.strip()
    if kb_section:
        kb_section = "\n\nCapabilities / Knowledge Base (for your own reference):\n" + kb_section

    user_prompt = f"""Conversation so far (most recent last):
{history_ctx}

Latest user message:
{user_text}

Current intent: {intent or 'unknown'}
Current product focus (if any): {product or 'none'}
{kb_section}

Draft assistant reply from internal tools/flows:
{draft.content}

Please rewrite or improve this reply following the goals above. It is ok to
add a gentle follow-up question or suggestion (for example, offering travel
insurance when the user mentions an overseas trip), but do not be pushy.
"""

    try:
        out_msg = _router_model.invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        )
        final_text = str(getattr(out_msg, "content", "") or "").strip()
    except Exception as e:
        logger.warning("Agentic.styler: styling failed, keeping draft reply: %s", e)
        final_text = ""

    if not final_text:
        # Fall back to the original draft if styling fails.
        final_text = str(getattr(draft, "content", "") or "").strip()
        if not final_text:
            return {}

    return {"messages": [AIMessage(content=final_text)]}
