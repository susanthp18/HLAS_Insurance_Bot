from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import FeedbackPrediction
from ..config import _router_model
from ..utils.memory import _get_last_turn_from_messages

logger = logging.getLogger(__name__)

FEEDBACK_SYSTEM_PROMPT = """You help the HLAS insurance chatbot understand how the user
is reacting to its PREVIOUS answer.

You will see the last assistant reply and the user's latest message.

Label the user's latest message as:
- negative_feedback: they say the previous answer was wrong, incomplete,
  off-topic, confusing, or not what they wanted.
- ack: they acknowledge or close (e.g. 'ok', 'thanks', 'great', 'bye') without
  asking for more.
- clarification: they ask you to clarify, simplify, or expand on the last
  answer (e.g. 'can you explain in simpler terms?', 'I don't understand').
- new_question: they move to a new substantive question or different topic.
- other: anything else.

Be strict: only use negative_feedback when the user is clearly unhappy with or
disagreeing with the last answer.
"""

_feedback_structured = _router_model.with_structured_output(FeedbackPrediction)


def _classify_feedback_from_messages(
    messages: List[BaseMessage],
) -> Optional[FeedbackPrediction]:
    """Classify whether the user is giving negative feedback about the last reply.

    Returns None if there is no usable history or classification fails.
    """

    last_turn = _get_last_turn_from_messages(messages)
    if not last_turn:
        return None
    last_answer = (last_turn.get("assistant") or "").strip()
    last_user = (last_turn.get("user") or "").strip()
    if not last_answer or not last_user:
        return None

    ctx = (
        f"LAST_ANSWER: {last_answer}\n"
        f"USER_MESSAGE: {last_user}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FEEDBACK_SYSTEM_PROMPT),
            ("user", "{context}"),
        ]
    )
    try:
        result = (prompt | _feedback_structured).invoke({"context": ctx})
        if isinstance(result, FeedbackPrediction):
            return result
        return FeedbackPrediction.model_validate(result)
    except Exception as e:
        logger.warning("Agentic feedback classification failed: %s", e)
        return None


def _self_critique_and_rewrite_from_messages(
    messages: List[BaseMessage],
) -> Optional[str]:
    """Use the router model to improve or correct the last answer based on feedback.

    Returns a revised answer, or None on failure.
    """

    last_turn = _get_last_turn_from_messages(messages)
    if not last_turn:
        return None
    last_answer = (last_turn.get("assistant") or "").strip()
    last_user = (last_turn.get("user") or "").strip()
    if not last_answer or not last_user:
        return None

    sys_msg = (
        "You are a self-critique helper for the HLAS insurance chatbot. "
        "You see the bot's previous answer and the user's feedback. "
        "If the answer was wrong, incomplete, or off-topic, correct it and "
        "provide a clearer, more accurate response. If the user is mainly "
        "asking for clarification, restate the key points more simply. "
        "Stay within HLAS insurance products and do not invent new products."
    )
    user_content = (
        "[Previous answer]\n" + last_answer + "\n\n" +
        "[User feedback]\n" + last_user
    )
    try:
        msg = _router_model.invoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=user_content),
            ]
        )
        revised = str(getattr(msg, "content", "") or "").strip()
        return revised or None
    except Exception as e:
        logger.warning("Agentic self-critique failed: %s", e)
        return None
