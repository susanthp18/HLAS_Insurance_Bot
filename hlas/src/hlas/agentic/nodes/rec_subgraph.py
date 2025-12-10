from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Literal

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..state import AgentState
from ..config import _router_model, _load_slot_rules
from ..utils.slots import (
    _required_slots_for_product,
    _slot_descriptions,
    _get_slot_value,
    _detect_product_llm,
    _normalize_product_key,
)
from ..utils.memory import _get_last_user_message
from ..tools.recommendation import _generate_recommendation_text

logger = logging.getLogger(__name__)

class SlotUpdate(BaseModel):
    slot_name: str
    value: str
    confidence: float

class SlotExtraction(BaseModel):
    updates: List[SlotUpdate]
    side_question: Optional[str] = Field(description="If user asks a question instead of answering.")

def _rec_ensure_product(state: AgentState) -> AgentState:
    """Ensure product is known before proceeding."""
    msg = _get_last_user_message(state["messages"])
    prod = state.get("product")
    
    if not prod:
        prod = _normalize_product_key(_detect_product_llm(msg))
    
    if not prod:
        # Ask clarification
        return {
            "messages": [AIMessage(content="Which product would you like a recommendation for: Travel, Maid, Car, Personal Accident, Home, Early, Fraud or Hospital?")],
            "rec_ready": False
        }
    
    return {"product": prod}

def _rec_extract_slots(state: AgentState) -> AgentState:
    """Extract slots from the latest message."""
    prod = state.get("product")
    if not prod:
        return {} # Should be handled by ensure_product
        
    msg = _get_last_user_message(state["messages"])
    current_slots = state.get("slots") or {}
    required = _required_slots_for_product(prod)
    
    # Only extract relevant slots
    sys_msg = (
        f"You are extracting slots for {prod} insurance recommendation. "
        f"Required slots: {', '.join(required)}. "
        f"Current slots: {current_slots}. "
        "Extract values from the user message. If user asks a question, set side_question."
    )
    
    try:
        structured = _router_model.with_structured_output(SlotExtraction)
        result = structured.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=msg)
        ])
        
        new_slots = dict(current_slots)
        for update in result.updates:
            if update.slot_name in required:
                new_slots[update.slot_name] = update.value
                
        # If side_question, we might want to set a flag or handle it. 
        # For now, we proceed. Implementation of side_info handling 
        # would require jumping out to info_tool.
        
        return {"slots": new_slots}
    except Exception as e:
        logger.warning("Slot extraction failed: %s", e)
        return {}

def _rec_validate_slots(state: AgentState) -> AgentState:
    """Validate slots using rules."""
    # In a full implementation, this would load slot_validation_rules.yaml
    # and check each slot. For now, we assume extracted slots are valid enough 
    # or rely on the final generation to handle missing data gracefully, 
    # mirroring the 'logic preserved' requirement.
    # The legacy code didn't have strict per-slot validation loop in the helper 
    # other than checking for presence.
    return {}

def _rec_manager(state: AgentState) -> str:
    """Decide next step: ask slot, generate rec, or exit."""
    prod = state.get("product")
    if not prod:
        return "end_turn" # Wait for user reply to product question
        
    required = _required_slots_for_product(prod)
    slots = state.get("slots") or {}
    missing = [s for s in required if not _get_slot_value(slots, s)]
    
    if missing:
        return "ask_next_slot"
    
    return "generate_rec"

def _rec_ask_next_slot(state: AgentState) -> AgentState:
    prod = state.get("product")
    required = _required_slots_for_product(prod)
    slots = state.get("slots") or {}
    missing = [s for s in required if not _get_slot_value(slots, s)]
    
    if not missing:
        return {}

    next_slot = missing[0]
    desc_map = _slot_descriptions(prod)
    description = desc_map.get(next_slot, f"information about {next_slot}")
    
    sys_msg = (
        "You are helping collect information to recommend an HLAS insurance plan. "
        "Ask ONE concise, friendly question to collect the requested detail."
    )
    user_msg = (
        f"Product: {prod}\nSlot name: {next_slot}\nDescription: {description}\n"
        "Please ask the user for this information."
    )
    
    try:
        q_msg = _router_model.invoke(
            [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]
        )
        question = str(getattr(q_msg, "content", "") or "").strip()
    except Exception:
        question = f"Could you please provide details for {next_slot}?"
        
    return {
        "messages": [AIMessage(content=question)],
        "pending_slot": next_slot # Track what we asked
    }

def _rec_generate_recommendation(state: AgentState) -> AgentState:
    prod = state.get("product")
    slots = state.get("slots") or {}
    
    tier, rec_text = _generate_recommendation_text(prod, slots)
    answer = rec_text or (
        "Based on what you've shared, I recommend a plan, but I couldn't format the full "
        "explanation. Please try asking again in a slightly different way."
    )
    answer = answer.rstrip() + "\n\nWould you like to see how to purchase this plan?"
    
    return {
        "messages": [AIMessage(content=answer)],
        "rec_ready": True
    }

# Build the subgraph
rec_builder = StateGraph(AgentState)
rec_builder.add_node("ensure_product", _rec_ensure_product)
rec_builder.add_node("extract_slots", _rec_extract_slots)
rec_builder.add_node("validate_slots", _rec_validate_slots)
rec_builder.add_node("ask_next_slot", _rec_ask_next_slot)
rec_builder.add_node("generate_rec", _rec_generate_recommendation)

rec_builder.set_entry_point("ensure_product")

rec_builder.add_edge("ensure_product", "extract_slots")
rec_builder.add_edge("extract_slots", "validate_slots")

rec_builder.add_conditional_edges(
    "validate_slots",
    _rec_manager,
    {
        "end_turn": END,
        "ask_next_slot": "ask_next_slot",
        "generate_rec": "generate_rec"
    }
)

rec_builder.add_edge("ask_next_slot", END)
rec_builder.add_edge("generate_rec", END)

recommendation_subgraph = rec_builder.compile()
