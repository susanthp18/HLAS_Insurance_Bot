from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage

# Local infrastructure imports (lazy singleton - no explicit init needed)
from ..infrastructure import get_response_llm
from .benefits import get_product_benefits
from ..config import _load_rec_templates
from ..utils.slots import (
    _get_slot_value,
)

logger = logging.getLogger(__name__)

def _get_llm():
    """Get response LLM from local infrastructure (thread-safe singleton)."""
    return get_response_llm()

def _generate_recommendation_text(
    product: str, slots: Dict[str, Any]
) -> tuple[Optional[str], str]:
    """Generate final recommendation text and tier, preserving existing logic.

    This closely mirrors RecFlowHelper._generate_recommendation but is scoped
    locally for the agentic runtime.
    """

    p = (product or "").lower()
    tier: Optional[str] = None
    if p == "travel":
        tier = "Gold"
    elif p == "maid":
        coverage_above_mom = (_get_slot_value(slots, "coverage_above_mom_minimum") or "").strip().lower()
        if coverage_above_mom == "yes":
            tier = "Premier"
        elif coverage_above_mom == "no":
            tier = "Enhanced"
    elif p == "personalaccident":
        try:
            amount = int(_get_slot_value(slots, "desired_amount"))
            if 500 <= amount <= 1000:
                tier = "Silver"
            elif 1001 <= amount <= 2500:
                tier = "Premier"
            elif 2501 <= amount <= 3500:
                tier = "Platinum"
        except (ValueError, TypeError):
            tier = None
    elif p == "home":
        try:
            amount = int(_get_slot_value(slots, "coverage_amount"))
            if amount <= 100000:
                tier = "Silver"
            elif amount <= 200000:
                tier = "Gold"
            else:
                tier = "Platinum"
        except (ValueError, TypeError):
            tier = None
    elif p == "early":
        tier = None
    elif p == "fraud":
        freq = _get_slot_value(slots, "purchase_frequency").strip().lower()
        if freq in ("daily", "everyday", "every day"):
            tier = "Platinum"
        else:
            tier = "Gold"
    elif p == "hospital":
        raw = _get_slot_value(slots, "coverage") or ""
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        try:
            val = int(digits) if digits else 0
        except Exception:
            val = 0
        choices = [100, 200, 300]
        if val <= 0:
            sel = 200
        else:
            sel = min(choices, key=lambda x: abs(x - val))
        tier = {100: "Silver", 200: "Premier", 300: "Titanium"}.get(sel, "Premier")

    # Benefits text
    benefits_text = ""
    try:
        benefits_text = get_product_benefits(product)
    except Exception as e:
        logger.error("Agentic.recommendation: benefits retrieval failed - %s", e)

    rec_templates = _load_rec_templates()
    product_key = p
    tpl = rec_templates.get(product_key, {}) if rec_templates else {}

    if product_key == "maid":
        add_ons_pref = _get_slot_value(slots, "add_ons") or "not_required"
        sys_t = (tpl.get("system") or "").format(tier=tier or "", add_ons=add_ons_pref)
        usr_t = (tpl.get("user") or "").format(
            tier=tier or "",
            add_ons=add_ons_pref,
            benefits=benefits_text or "",
        )
    elif product_key == "travel":
        destination = (_get_slot_value(slots, "destination") or "").strip()
        if destination:
            advisory = (
                f"Medical treatment in {destination} is very good, but can be very expensive. "
                f"Some foreign visitors who cannot cover their medical costs may face restrictions in the future."
            )
        else:
            advisory = (
                "Medical treatment abroad is very good, but can be very expensive. "
                "Some foreign visitors who cannot cover their medical costs may face restrictions in the future."
            )
        sys_t = (tpl.get("system") or "").format(tier=tier or "", destination=destination or "")
        usr_t = (tpl.get("user") or "").format(
            tier=tier or "",
            benefits=benefits_text or "",
            advisory=advisory or "",
            destination=destination or "",
        )
    else:
        sys_t = (tpl.get("system") or "").format(tier=tier or "")
        usr_t = (tpl.get("user") or "").format(tier=tier or "", benefits=benefits_text or "")

    response = ""
    if product_key == "early":
        # Early CI has its own fixed messaging
        try:
            tpl_e = rec_templates.get("early") or {}
            sys_e = (tpl_e.get("system") or "")
            usr_e = (tpl_e.get("user") or "").format(benefits=benefits_text or "")
            if sys_e and usr_e:
                llm = _get_llm()
                if llm:
                    txt = llm.call(
                        messages=[
                            {"role": "system", "content": sys_e},
                            {"role": "user", "content": usr_e},
                        ]
                    )
                    response = str(txt).strip()
                else:
                    logger.error("Agentic.recommendation: LLM not initialized")
        except Exception as e:
            logger.error("Agentic.recommendation: Early LLM call failed - %s", e)
    elif sys_t and usr_t:
        try:
            llm = _get_llm()
            if llm:
                txt = llm.call(
                    messages=[
                        {"role": "system", "content": sys_t},
                        {"role": "user", "content": usr_t},
                    ]
                )
                response = str(txt).strip()
            else:
                logger.error("Agentic.recommendation: LLM not initialized")
        except Exception as e:
            logger.error("Agentic.recommendation: LLM call failed - %s", e)

    return tier, response

