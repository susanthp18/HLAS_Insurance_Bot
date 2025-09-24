from typing import Dict, Any, Optional
import logging
from pathlib import Path
import yaml
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from ..tasks import (
    questionnaire_ask_next_slot_task,
    questionnaire_capture_pending_slot_task,
    validate_slot_task,
    provide_recommendation_task,
)
from ..tools.benefits_tool import benefits_tool
from ..prompt_runner import run_direct_task
from ..agents import recommendation_responder


class RecommendationFlowHelper:
    """One-turn recommendation handler. Decides whether to ask or capture, validates, and advances.

    This helper updates the provided `state` in-place and returns "__done__" to signal termination.
    Expected state fields: session, product, message, reply, last_question, pending_slot.
    """

    @staticmethod
    def _required_slots_for_product(product: Optional[str]) -> list[str]:
        if not product:
            return []
        p = (product or "").lower()
        if p == "travel":
            return [
                "destination",
                "travel_duration",
                "pre_existing_medical_condition",
                "plan_preference",
            ]
        if p == "maid":
            return [
                "duration_of_insurance",
                "maid_country",
                "coverage_above_mom_minimum",
                "add_ons",
            ]
        return []

    @classmethod
    def _first_missing_slot(cls, product: Optional[str], slots: Dict[str, Any]) -> Optional[str]:
        for s in cls._required_slots_for_product(product):
            if s not in slots or slots.get(s) in (None, ""):
                return s
        return None

    @classmethod
    def handle(cls, state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        directive = decision.get("directive")
        # Ensure product is available from state, session, or decision
        product = state.product or state.session.get("product") or decision.get("product")
        if not state.product and product:
            state.product = product
        slots = state.session.get("slots", {}) or {}
        pending = state.session.get("pending_slot")

        # Enforce product-specific slots only; drop leaked slots from other products
        allowed_slots = cls._required_slots_for_product(product)
        try:
            if slots:
                filtered_slots = {k: v for k, v in slots.items() if k in allowed_slots}
                if filtered_slots != slots:
                    dropped = [k for k in slots.keys() if k not in allowed_slots]
                    logger.info("Slots.filter: product=%s dropped=%s", product, dropped)
                    state.session["slots"] = filtered_slots
                    slots = filtered_slots
            if pending and pending not in allowed_slots:
                logger.info("Pending_slot.mismatch: product=%s clearing pending_slot=%s", product, pending)
                state.session.pop("pending_slot", None)
                state.pending_slot = None
                pending = None
        except Exception:
            # Never fail flow due to filtering
            pass

        logger.info("RecommendationFlow.handle: directive=%s product=%s pending=%s slots=%s", 
                   directive, product, pending, list(slots.keys()))

        # Ask for next required slot
        if directive == "ask_slot":
            next_required = cls._first_missing_slot(product, slots)
            ctx = f"Product: {product}\nKnown slots: {slots}\nNext slot to ask: {next_required}"
            qa = run_direct_task(
                agent_obj=questionnaire_ask_next_slot_task.agent,
                agent_key="questionnaire_agent",
                task_key="questionnaire_ask_next_slot",
                context_text=ctx,
                logger=logger,
                label="questionnaire_agent.ask_next_slot",
            ) or {}
            slot_to_ask = next_required
            question = qa.get("question") or state.question or "Please provide the next detail."
            if slot_to_ask:
                state.session["pending_slot"] = slot_to_ask
                state.pending_slot = slot_to_ask
            state.reply = question
            state.last_question = question
            logger.info("RecommendationFlow.ask_slot: slot=%s", slot_to_ask)
            return "__done__"

        # Capture a value for the (pending or routed) slot, validate, and advance
        if directive == "capture_slot":
            target_slot = pending or decision.get("slot_name") or state.slot_name
            ctx = (
                f"Product: {product}\n"
                f"Known slots: {slots}\n"
                f"Target slot: {target_slot or ''}\n"
                f"User message: {state.message}"
            )
            qa = run_direct_task(
                agent_obj=questionnaire_capture_pending_slot_task.agent,
                agent_key="questionnaire_agent",
                task_key="questionnaire_capture_pending_slot",
                context_text=ctx,
                logger=logger,
                label="questionnaire_agent.capture_pending_slot",
            ) or {}
            slot_name = target_slot or qa.get("slot_name")
            slot_value = qa.get("value") or state.slot_value or state.message
            logger.info("RecommendationFlow.capture: slot=%s value='%s'", slot_name, slot_value)
            if slot_name and slot_value and product:

                # Get current date for context
                try:
                    now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
                    date_str = f"Current date (Asia/Singapore): {now_sg.strftime('%d %B %Y')}"
                except Exception:
                    date_str = ""

                # Load slot validation rules from YAML
                rules_block = ""
                try:
                    base_dir = Path(__file__).resolve().parent.parent
                    with open(base_dir / "config" / "slot_validation_rules.yaml", "r", encoding="utf-8") as rf:
                        rules_yaml = yaml.safe_load(rf) or {}
                    product_key = (product or "").lower()
                    slot_key = (slot_name or "").lower()
                    lines = rules_yaml.get(product_key, {}).get(slot_key, [])
                    if lines:
                        rules_block = "Validation rules:\n" + "\n".join(lines)
                except Exception:
                    rules_block = ""

                v_ctx = (
                    f"Product: {product}\n"
                    f"Slot: {slot_name}\n"
                    f"Value: {slot_value}\n"
                    f"User message: {state.message}\n"
                    f"{date_str}\n"
                    f"{rules_block}"
                ).strip()
                # Always use unified slot validator with dynamic rules
                from ..tasks import validate_slot_task as _vts
                v_agent = _vts.agent
                v_task_key = "validate_slot"
                vr = run_direct_task(
                    agent_obj=v_agent,
                    agent_key="slot_validator",
                    task_key=v_task_key,
                    context_text=v_ctx,
                    logger=logger,
                    label=f"{v_task_key}",
                ) or {"valid": False, "question": "Could you clarify that value?"}
                logger.info("RecommendationFlow.validate: slot=%s valid=%s", slot_name, vr.get("valid"))

                if vr.get("valid") and vr.get("normalized_value"):
                    # Save the validated slot
                    new_slots = dict(slots)
                    new_slots[slot_name] = vr["normalized_value"]
                    state.session["slots"] = new_slots
                    # Clear pending slot immediately after successful validation
                    state.session.pop("pending_slot", None)
                    state.pending_slot = None
                    state.last_question = None
                    logger.info("RecommendationFlow.slot_saved: %s=%s pending_slot_cleared=True", slot_name, vr["normalized_value"])

                    # Greedy multi-extraction: try to fill other missing required slots from the same message
                    # Skip multi-extraction for add_ons to ensure it's always asked explicitly
                    for s in cls._required_slots_for_product(product):
                        if s in new_slots and new_slots.get(s) not in (None, ""):
                            continue
                        if s == "add_ons":
                            # Skip multi-extraction for add_ons - always ask explicitly
                            logger.info("RecommendationFlow.multi_extraction: skipping add_ons - will ask explicitly")
                            continue
                        m_ctx = (
                            f"Product: {product}\n"
                            f"Known slots: {new_slots}\n"
                            f"Target slot: {s}\n"
                            f"User message: {state.message}"
                        )
                        m_qa = run_direct_task(
                            agent_obj=questionnaire_capture_pending_slot_task.agent,
                            agent_key="questionnaire_agent",
                            task_key="questionnaire_capture_pending_slot",
                            context_text=m_ctx,
                            logger=logger,
                            label=f"questionnaire_agent.multi_capture.{s}",
                        ) or {}
                        m_val = m_qa.get("value")
                        if not m_val:
                            continue


                        # Get current date for context
                        try:
                            now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
                            date_str = f"Current date (Asia/Singapore): {now_sg.strftime('%d %B %Y')}"
                        except Exception:
                            date_str = ""

                        # Load dynamic rules for greedy slot from YAML as well
                        try:
                            base_dir = Path(__file__).resolve().parent.parent
                            with open(base_dir / "config" / "slot_validation_rules.yaml", "r", encoding="utf-8") as rf:
                                rules_yaml = yaml.safe_load(rf) or {}
                            product_key = (product or "").lower()
                            slot_key = (s or "").lower()
                            lines = rules_yaml.get(product_key, {}).get(slot_key, [])
                            rules_block = ("Validation rules:\n" + "\n".join(lines)) if lines else ""
                        except Exception:
                            rules_block = ""
                        v2_ctx = (
                            f"Product: {product}\n"
                            f"Slot: {s}\n"
                            f"Value: {m_val}\n"
                            f"User message: {state.message}\n"
                            f"{date_str}\n"
                            f"{rules_block}"
                        ).strip()
                        from ..tasks import validate_slot_task as _vts2
                        v2 = run_direct_task(
                            agent_obj=_vts2.agent,
                            agent_key="slot_validator",
                            task_key="validate_slot",
                            context_text=v2_ctx,
                            logger=logger,
                            label=f"validate_slot.multi_validate.{s}",
                        ) or {"valid": False}
                        if v2.get("valid") and v2.get("normalized_value"):
                            new_slots[s] = v2["normalized_value"]
                            state.session["slots"] = new_slots
                            logger.info("RecommendationFlow.multi_slot_saved: %s=%s", s, v2["normalized_value"])
                        else:
                            # Ask clarifying for this slot and stop greedy fill
                            question = v2.get("question") or f"Please provide a valid value for {s}."
                            state.session["pending_slot"] = s
                            state.pending_slot = s
                            state.reply = question
                            state.last_question = question
                            logger.info("RecommendationFlow.multi_validation_failed: slot=%s", s)
                            return "__done__"

                    # After greedy fill, proceed to next missing or recommend
                    next_required = cls._first_missing_slot(product, new_slots)
                    logger.info("RecommendationFlow.after_multi_extraction: next_required=%s all_slots=%s", 
                               next_required, list(new_slots.keys()))
                    if next_required:
                        n_ctx = f"Product: {product}\nKnown slots: {new_slots}\nNext slot to ask: {next_required}"
                        qa_next = run_direct_task(
                            agent_obj=questionnaire_ask_next_slot_task.agent,
                            agent_key="questionnaire_agent",
                            task_key="questionnaire_ask_next_slot",
                            context_text=n_ctx,
                            logger=logger,
                            label="questionnaire_agent.ask_next_after_capture",
                        ) or {}
                        next_slot = next_required
                        question = qa_next.get("question") or "Please provide the next detail."
                        state.session["pending_slot"] = next_slot
                        state.pending_slot = next_slot
                        state.reply = question
                        state.last_question = question
                        logger.info("RecommendationFlow.next_slot: %s pending_slot_set=True", next_slot)
                        return "__done__"
                    else:
                        # All slots available: build recommendation, fetch benefits by product only, and synthesize final reply via product-specific template
                        # Determine tier based on plan preference (hardcoded policy)
                        pref = (new_slots.get("plan_preference") or "").strip().lower()
                        tier = None
                        if (product or "").lower() == "travel":
                            if pref == "budget":
                                tier = "Silver"
                            elif pref == "comprehensive":
                                tier = "Gold"
                        elif (product or "").lower() == "maid":
                            coverage_above_mom = (new_slots.get("coverage_above_mom_minimum") or "").strip().lower()
                            if coverage_above_mom == "yes":
                                tier = "Gold"
                            elif coverage_above_mom == "no":
                                tier = "Silver"
                        # Fallbacks if something is off
                        tier = tier or new_slots.get("recommended_tier") or ""
                        # Fetch all benefits for product (no tier filter)
                        benefits_text = benefits_tool.run(product=product)
                        # Load product-specific recommendation templates
                        try:
                            base_dir = Path(__file__).resolve().parent.parent
                            with open(base_dir / "config" / "recommendation_response.yaml", "r", encoding="utf-8") as rf:
                                rec_templates = yaml.safe_load(rf) or {}
                        except Exception:
                            rec_templates = {}

                        product_key = (product or "").lower()
                        tpl = rec_templates.get(product_key) or {}
                        sys_t = (tpl.get("system") or "").format(tier=tier or "")
                        
                        if product_key == "maid":
                            # Include add_ons information for Maid
                            add_ons_pref = new_slots.get("add_ons", "not_required")
                            usr_t = (tpl.get("user") or "").format(tier=tier or "", add_ons=add_ons_pref, benefits=benefits_text or "")
                        elif product_key == "car":
                            # No tiers: ignore tier fields
                            sys_t = (rec_templates.get("car", {}).get("system") or "")
                            usr_t = (rec_templates.get("car", {}).get("user") or "").format(benefits=benefits_text or "")
                        else:
                            # Travel and other products
                            usr_t = (tpl.get("user") or "").format(tier=tier or "", benefits=benefits_text or "")

                        if sys_t and usr_t:
                            final = run_direct_task(
                                agent_obj=recommendation_responder,
                                agent_key="recommendation_responder",
                                task_key="synthesize_response",
                                context_text=f"[System]\n{sys_t}\n\n[User]\n{usr_t}",
                                logger=logger,
                                label="recommendation.response_synthesis",
                            ) or {}
                            # final may be raw text if model returns a string; safeguard
                            state.reply = final.get("response") if isinstance(final, dict) else str(final)
                        else:
                            # Fallback to simple synthesis
                            state.reply = (
                                f"We recommend {tier}.\n\n"
                                f"Here are key benefits:\n{benefits_text[:1500]}"
                            )
                        # Log before clearing for debugging
                        cleared_slot = state.session.get("pending_slot")
                        cleared_question = state.last_question
                        logger.info("RecommendationFlow.final_cleanup: clearing pending_slot='%s' last_question='%s'", 
                                   cleared_slot, cleared_question)
                        # Ensure we clear any pending slot and last question after finalizing recommendation
                        state.session.pop("pending_slot", None)
                        state.pending_slot = None
                        state.last_question = None
                        # Mark completion and ensure comparison pending is cleared to avoid unintended bypass
                        try:
                            state.session["last_completed"] = "recommendation"
                        except Exception:
                            pass
                        try:
                            state.session.pop("compare_pending", None)
                        except Exception:
                            pass
                        logger.info("Recommendation.completed: cleared pending_slot='%s' last_question='%s' reply_len=%d", 
                                   cleared_slot, cleared_question, len(str(state.reply or "")))
                        return "__done__"
                else:
                    # Validation failed → keep the same pending slot and ask the clarifying question or generate one
                    if vr.get("question"):
                        state.reply = vr.get("question")
                    else:
                        g_ctx = f"Product: {product}\nKnown slots: {slots}\nSlot to ask: {slot_name}"
                        g = run_direct_task(
                            agent_obj=questionnaire_ask_next_slot_task.agent,
                            agent_key="questionnaire_agent",
                            task_key="questionnaire_ask_next_slot",
                            context_text=g_ctx,
                            logger=logger,
                            label="questionnaire_agent.guidance_after_validation_fail",
                        ) or {}
                        state.reply = g.get("question") or f"Please provide a valid value for {slot_name}."
                    state.session["pending_slot"] = slot_name
                    state.pending_slot = slot_name
                    state.last_question = state.reply
                    logger.info("RecommendationFlow.validation_failed: slot=%s", slot_name)
                    return "__done__"

        # Fallback: ask next slot if unclear
        next_required = cls._first_missing_slot(product, slots)
        if next_required:
            ctx = f"Product: {product}\nKnown slots: {slots}\nNext slot to ask: {next_required}"
            qa = run_direct_task(
                agent_obj=questionnaire_ask_next_slot_task.agent,
                agent_key="questionnaire_agent",
                task_key="questionnaire_ask_next_slot",
                context_text=ctx,
                logger=logger,
                label="questionnaire_agent.fallback_ask_next",
            ) or {}
            slot_to_ask = next_required
            question = qa.get("question") or "Please provide the next detail."
            state.session["pending_slot"] = slot_to_ask
            state.pending_slot = slot_to_ask
            state.reply = question
            state.last_question = question
            logger.info("RecommendationFlow.fallback_ask: slot=%s", slot_to_ask)
            return "__done__"
        state.reply = state.reply or "How can I help you further?"
        return "__done__"



