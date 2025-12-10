from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import AgentState
from ..utils.memory import _get_last_user_message
from ..tools.info import _info_tool
from ..tools.summary import _summary_tool
from ..tools.compare import _compare_tool
from ..tools.purchase import _purchase_tool
from ..tools.capabilities import _capabilities_tool


def _greet_agent_node(state: AgentState) -> AgentState:
    reply = (
        "Hello! I’m the HLAS Smart Bot. I can help you with information, "
        "summaries, comparisons and recommendations for our insurance "
        "plans. How can I help you today?"
    )
    return {"messages": [AIMessage(content=reply)], "sources": []}


def _capabilities_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    reply = _capabilities_tool(user_text)
    return {"messages": [AIMessage(content=reply)], "sources": []}


def _chat_agent_node(state: AgentState) -> AgentState:
    """General small-talk / open conversation agent.

    Uses the main Azure model to answer like a human while gently steering
    towards helpful insurance support when appropriate.
    """

    user_text = _get_last_user_message(state.get("messages", []) or [])

    system_prompt = """You are HLAS's friendly digital insurance assistant.

You should sound warm, natural and human in short conversations.

Behaviour guidelines:
- Answer the user's message directly as a person would in a chat (e.g. if
  they say "how are you", you can say you are doing well and ask how their
  day is going).
- Briefly remind them you are an insurance assistant and can help with
  HLAS products like Travel, Maid, Car, Home, Personal Accident, Early
  Critical Illness, Fraud (Protect360) and Hospital Cash.
- If the user shares life context that naturally connects to insurance
  (for example: travelling overseas, buying a car or house, planning a
  family, worrying about scams or hospital costs), gently highlight how
  HLAS insurance could help and ask if they would like a recommendation.
- Keep replies concise and conversational; avoid sounding like a script.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{user_text}"),
        ]
    )

    # Reuse the router model for chat-style responses to keep configuration
    # simple and fully LLM-driven.
    from ..config import _router_model

    chain = prompt | _router_model
    ai_msg = chain.invoke({"user_text": user_text or ""})

    content = getattr(ai_msg, "content", None) or str(ai_msg)
    return {"messages": [AIMessage(content=content)], "sources": []}


def _info_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    answer, srcs = _info_tool(state.get("product"), user_text)
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _summary_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    answer, srcs = _summary_tool(state.get("product"), state.get("tiers") or [], user_text)
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _compare_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    answer, srcs = _compare_tool(state.get("product"), state.get("tiers") or [], user_text)
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _purchase_agent_node(state: AgentState) -> AgentState:
    reply = _purchase_tool(state.get("product"))
    return {"messages": [AIMessage(content=reply)], "sources": []}
