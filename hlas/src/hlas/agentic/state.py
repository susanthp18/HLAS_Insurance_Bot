from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class IntentPrediction(BaseModel):
    """High-level routing decision for the experimental agent."""

    intent: Literal[
        "info",
        "summary",
        "compare",
        "recommend",
        "purchase",
        "capabilities",
        "greet",
        "chat",
        "other",
    ] = Field(
        description=(
            "One of: 'info', 'summary', 'compare', 'recommend', 'purchase', "
            "'capabilities', 'greet', 'chat', 'other'. Use 'info' for general questions "
            "about benefits/coverage, 'summary' for high-level overviews of a "
            "product/tiers, 'compare' for differences between plans/tiers, "
            "'recommend' when the user wants a personalised plan suggestion, "
            "and 'purchase' when they clearly want to buy or get a link. "
            "Use 'capabilities' for questions about what the bot can do; use 'greet' "
            "for very short greetings like 'hi' or 'hello'; use 'chat' for small-talk "
            "or open conversation where the user is not yet asking a concrete "
            "insurance question but may be sharing life context (e.g. travel plans, "
            "new house, new car, family, health)."
        )
    )
    product: Optional[str] = Field(
        default=None,
        description=(
            "Normalized product name if clearly specified (e.g. Travel, Maid, "
            "Car, PersonalAccident, Home, Early, Fraud, Hospital). Leave empty "
            "if ambiguous."
        ),
    )
    reason: str = Field(
        default="",
        description="Short natural-language explanation for the chosen intent.",
    )


class FeedbackPrediction(BaseModel):
    """Classifier for user reactions to the last answer.

    This is used for negative feedback handling and self-correction.
    """

    category: Literal[
        "negative_feedback",
        "ack",
        "clarification",
        "new_question",
        "other",
    ] = Field(
        description=(
            "How the user is reacting to the PREVIOUS answer. "
            "'negative_feedback' = complaining, saying it was wrong, not helpful, "
            "or off-topic. 'ack' = simple thanks/ok/got it/bye. 'clarification' = "
            "they say answer was unclear and ask you to clarify or simplify it. "
            "'new_question' = they move on to a new substantive topic."
        )
    )
    reason: str = Field(
        default="",
        description="Short natural-language explanation of why this category was chosen.",
    )


class AgentState(MessagesState):
    """LangGraph state for the /agent-chat agent.

    messages: list[BaseMessage] is inherited from MessagesState.
    """

    intent: Optional[str]
    product: Optional[str]
    tiers: List[str]
    slots: Dict[str, Any]
    rec_ready: bool
    sources: List[str]
    feedback: Optional[str]
    pending_slot: Optional[str]
    has_summary: bool
