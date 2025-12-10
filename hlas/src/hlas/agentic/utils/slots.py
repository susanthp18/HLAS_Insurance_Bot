from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from ..config import _router_model
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ProductDetection(BaseModel):
    product: Optional[str] = Field(
        description="The specific insurance product detected from the text (Travel, Maid, Car, PersonalAccident, Home, Early, Fraud, Hospital). None if unclear."
    )

def _normalize_product_key(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return str(name).strip().lower()

def _detect_product_llm(message: str) -> Optional[str]:
    """Detect product using LLM only, no regex heuristics."""
    if not message:
        return None
        
    sys_msg = (
        "You are an expert insurance product classifier for HLAS. "
        "Identify which product the user is interested in from: "
        "Travel, Maid, Car, PersonalAccident, Home, Early (Critical Illness), Fraud (Protect360), Hospital (Protect360). "
        "\n\nIMPORTANT ALIASES:"
        "\n- 'Family Protect360', 'Family Protect 360', 'Family Protect', 'PA insurance' = PersonalAccident"
        "\n- 'Travel Protect360', 'Travel Protect 360' = Travel"
        "\n- 'Maid Protect360', 'Maid Protect 360', 'helper insurance' = Maid"
        "\n- 'Car Protect360', 'Car Protect 360', 'motor insurance' = Car"
        "\n- 'Home Protect360', 'Home Protect 360' = Home"
        "\n- 'Early Protect360', 'critical illness' = Early"
        "\n- 'Fraud Protect360', 'scam protection' = Fraud"
        "\n- 'Hospital Protect360', 'hospital cash' = Hospital"
        "\n\nReturn null if not clearly about one of these."
    )
    
    try:
        structured = _router_model.with_structured_output(ProductDetection)
        result = structured.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=message)
        ])
        prod = _normalize_product_key(result.product)
        if not prod:
            logger.debug("Agentic.slots.detect_product_llm: no product detected")
            return None

        # Normalize aliases to canonical product keys
        alias_map = {
            "familyprotect360": "personalaccident",
            "familyprotect": "personalaccident",
            "family protect 360": "personalaccident",
            "family protect": "personalaccident",
            "pa": "personalaccident",
        }
        if prod in alias_map:
            logger.info(
                "Agentic.slots.detect_product_llm: alias '%s' mapped to personalaccident",
                prod,
            )
            return alias_map[prod]

        logger.debug("Agentic.slots.detect_product_llm: detected product=%s", prod)
        return prod
    except Exception as e:
        logger.warning("Agentic.slots.detect_product_llm failed: %s", e)
        return None

def _get_slot_value(slots: Dict[str, Any], slot_name: str) -> str:
    """Get a simple string value from a slot container."""

    slot_data = slots.get(slot_name)
    if isinstance(slot_data, dict):
        return str(slot_data.get("value") or "")
    return str(slot_data or "")


def _required_slots_for_product(product: Optional[str]) -> List[str]:
    """Return the list of slots required for a recommendation per product.

    This mirrors RecFlowHelper._required_slots_for_product but is kept local
    to the agentic layer.
    """

    if not product:
        return []
    p = (product or "").lower()
    if p == "travel":
        return ["coverage_scope", "destination"]
    if p == "maid":
        return [
            "duration_of_insurance",
            "maid_country",
            "coverage_above_mom_minimum",
            "add_ons",
        ]
    if p == "personalaccident":
        return ["coverage_scope", "desired_amount"]
    if p in ("home", "homeprotect360"):
        return ["risk_concerns", "coverage_amount"]
    if p == "early":
        return ["existing_cover", "dependants"]
    if p == "car":
        return []
    if p == "fraud":
        return ["purchase_frequency", "scam_exp"]
    if p == "hospital":
        return ["age", "support", "coverage"]
    return []


def _slot_descriptions(product: Optional[str]) -> Dict[str, str]:
    """Short descriptions per slot to help question generation."""

    descriptions = {
        "travel": {
            "coverage_scope": (
                "Coverage for self, family, a group of adults, or a group of families. "
                "The validator will enforce any headcount limits."
            ),
            "destination": "Country the user is travelling to (country name only).",
        },
        "maid": {
            "duration_of_insurance": "Policy duration (14 or 26 months).",
            "maid_country": "Helper's country of origin (country name only).",
            "coverage_above_mom_minimum": (
                "Whether user wants coverage beyond MOM minimum (yes/no). MOM minimum is $60,000 medical "
                "coverage and a $5,000 security bond."
            ),
            "add_ons": "Whether the user wants optional add-on coverages (required/not_required).",
        },
        "personalaccident": {
            "coverage_scope": "Coverage for yourself or your family.",
            "desired_amount": (
                "Desired coverage amount between $500 and $3,500. Phrases like 'highest' can map to 3500; "
                "'minimum' can map to 500."
            ),
        },
        "home": {
            "risk_concerns": (
                "Specific worries such as fire, water damage, or theft (single, multiple, or 'all')."
            ),
            "coverage_amount": "Estimated total value of renovations, contents and valuables (numeric amount).",
        },
        "early": {
            "existing_cover": "Whether the user already has critical illness coverage (yes/no).",
            "dependants": "Whether family members rely on the user's income or care (yes/no).",
        },
        "fraud": {
            "purchase_frequency": "How often the user shops online (daily, weekly, monthly).",
            "scam_exp": "Whether the user has experienced or almost fallen for an online scam.",
        },
        "hospital": {
            "age": "User age or age band (below 25, 25-35, 36-45, above 45).",
            "support": "Whether the user supports anyone financially (yes/no).",
            "coverage": "Desired daily hospital cash (100, 200, or 300).",
        },
    }
    p = (product or "").lower()
    if p == "homeprotect360":
        p = "home"
    return descriptions.get(p, {})
