from typing import Dict, Any, Optional
import logging
from pathlib import Path
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import re

from ..prompt_runner import run_direct_task
from ..tools.benefits_tool import benefits_tool
from ..agents import recommendation_responder
from ..llm import azure_response_llm
from ..utils.product_lex import lexical_product_hint, lexical_product_candidates


class RecFlowHelper:
    """Simplified recommendation flow with clear separation of concerns.
    
    Architecture:
    1. slot_extractor: Extracts all possible slots from user message
    2. question_asker: Generates the next question for a missing slot
    3. Clear state management with recommendation_status flag
    """

    @staticmethod
    def _required_slots_for_product(product: Optional[str]) -> list[str]:
        """Get required slots for each product."""
        if not product:
            return []
        p = (product or "").lower()
        if p == "travel":
            return [
                "coverage_scope",
                "destination",
            ]
        if p == "maid":
            return [
                "duration_of_insurance",
                "maid_country",
                "coverage_above_mom_minimum", 
                "add_ons",
            ]
        if p == "personalaccident":
            return ["coverage_scope", "risk_level", "desired_amount"]
        if p == "home" or p == "homeprotect360":
            return ["risk_concerns", "coverage_amount"]
        if p == "early":
            return ["existing_cover", "dependants"]
        if p == "car":
            return []  # Car insurance has no slots to collect
        if p == "fraud":
            return [
                "purchase_frequency",
                "scam_exp",
            ]
        if p == "hospital":
            return [
                "age",
                "occupation",
                "support",
                "coverage",
            ]
        return []

    @staticmethod
    def _get_slot_descriptions(product: Optional[str]) -> Dict[str, str]:
        """Get descriptions for each slot to help extraction."""
        descriptions = {
            "travel": {
                "coverage_scope": "Coverage for self (e.g., myself/just me/me), family, a group of adults, or a group of families. Use general phrases; if a headcount is given, the validator will enforce limits (adults ≤ 20, families ≤ 10).",
                "destination": "Country the user is travelling to (country name only)",
            },
            "maid": {
                "duration_of_insurance": "Policy duration (12 or 24 months)",
                "maid_country": "Helper's country of origin (country name only)",
                "coverage_above_mom_minimum": "Whether user wants coverage beyond MOM minimum (yes/no)",
                "add_ons": "Whether user wants additional add-on coverages (required/not_required)"
            },
            "personalaccident": {
                "coverage_scope": "Coverage for yourself or your family",
                "risk_level": "Occupational risk level: high or low",
                "desired_amount": "Desired coverage amount between $500 and $3,500 (accept 'the higher the better' → 3500; 'the lower the better' → 500)"
            },
            "home": {
                "risk_concerns": "Specific worries such as fire, water damage, or theft (single, multiple, or 'all' to mean everything). Synonyms: burglary/break-in/stolen → theft; flood/leak/water/pipe burst → water damage.",
                "coverage_amount": "Estimated total value of renovations, home contents, and valuables (numeric amount)"
            },
            "early": {
                "existing_cover": "Whether the user already has CI insurance that pays a lump sum (yes/no)",
                "dependants": "Whether family members rely on the user's income or care (yes/no)"
            },
            "car": {},  # Car insurance has no slots
            "fraud": {
                "purchase_frequency": "How often the user shops online (daily, weekly, monthly)",
                "scam_exp": "Whether the user has experienced or almost fallen for an online scam (yes, almost, no)",
            },
            "hospital": {
                "age": "User age or range (below 25, 25-35, 36-45, above 45) or a reasonable age number",
                "occupation": "Free-text occupation (e.g., teacher, engineer, driver)",
                "support": "Whether the user supports anyone financially (Yes/No)",
                "coverage": "Desired daily hospital cash (e.g., $100/day, $200/day, $300/day)",
            },
        }
        p = (product or "").lower()
        if p == "homeprotect360":
            p = "home"
        return descriptions.get(p, {})

    @staticmethod
    def _get_slot_value(slots_dict: Dict[str, Any], slot_name: str) -> str:
        """Get the value from a slot structure."""
        slot_data = slots_dict.get(slot_name)
        if isinstance(slot_data, dict):
            return slot_data.get("value", "")
        # Handle legacy format (simple string values)
        return str(slot_data) if slot_data else ""

    @staticmethod
    def _set_slot_value(slots_dict: Dict[str, Any], slot_name: str, value: str, valid: bool = True) -> None:
        """Set a slot value with validation status."""
        slots_dict[slot_name] = {"value": value, "valid": valid}

    @staticmethod
    def _is_slot_valid(slots_dict: Dict[str, Any], slot_name: str) -> bool:
        """Check if a slot is already validated."""
        slot_data = slots_dict.get(slot_name)
        if isinstance(slot_data, dict):
            return slot_data.get("valid", False)
        # Legacy format is considered unvalidated
        return False

    @staticmethod
    def _get_missing_slots(slots_dict: Dict[str, Any], required_slots: list[str]) -> list[str]:
        """Get slots that are missing or have invalid values."""
        missing = []
        for slot_name in required_slots:
            slot_data = slots_dict.get(slot_name)
            if isinstance(slot_data, dict):
                # New format: check if value exists and is valid
                if not slot_data.get("value") or not slot_data.get("valid", False):
                    missing.append(slot_name)
            else:
                # Legacy format or missing: consider as missing
                if not slot_data:
                    missing.append(slot_name)
        return missing

    @staticmethod
    def _slot_specs(product: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Return product-specific slot metadata (type and options) for dynamic prompting."""
        p = (product or "").lower()
        if p == "travel":
            return {
                "coverage_scope": {"type": "choice", "options": ["self", "family", "group of adults", "group of families"]},
                "destination": {"type": "value", "format": "country"},
            }
        if p == "maid":
            return {
                "duration_of_insurance": {"type": "choice", "options": ["12", "24"]},
                "maid_country": {"type": "value", "format": "country"},
                "coverage_above_mom_minimum": {"type": "yesno"},
                "add_ons": {"type": "choice", "options": ["required", "not_required"]},
            }
        if p == "personalaccident":
            return {
                "coverage_scope": {"type": "choice", "options": ["self or me", "family"]},
                "risk_level": {"type": "choice", "options": ["high", "low"]},
                "desired_amount": {
                    "type": "value",
                    "format": "amount:int",
                    "hints": {
                        "preference_phrases": {
                            "higher": [
                                "the higher the better",
                                "highest",
                                "max",
                                "maximum",
                                "as high as possible",
                                "best coverage"
                            ],
                            "lower": [
                                "the lower the better",
                                "lowest",
                                "min",
                                "minimum",
                                "as low as possible",
                                "budget"
                            ]
                        },
                        "normalize_to": {"higher": "3500", "lower": "500"}
                    }
                },
            }
        if p in ("home", "homeprotect360"):
            return {
                "risk_concerns": {
                    "type": "value",
                    "options": ["fire", "water damage", "theft"],
                    "hints": {
                        "accept_all_phrases": ["all", "everything", "both"],
                        "synonyms": {
                            "theft": ["burglary", "break-in", "stolen"],
                            "water damage": ["flood", "leak", "water", "pipe burst"],
                            "fire": ["fire", "fires"]
                        },
                        "output_format": "comma-separated in order: fire, water damage, theft"
                    }
                },
                "coverage_amount": {"type": "value", "format": "amount:int"},
            }
        if p == "early":
            return {
                "existing_cover": {"type": "yesno"},
                "dependants": {"type": "yesno"},
            }
        if p == "fraud":
            return {
                "purchase_frequency": {"type": "choice", "options": ["daily", "weekly", "monthly"]},
                "scam_exp": {"type": "choice", "options": ["yes", "almost", "no"]},
            }
        if p == "hospital":
            return {
                "age": {"type": "value", "format": "age_or_range"},
                "occupation": {"type": "value", "format": "text"},
                "support": {"type": "yesno"},
                "coverage": {"type": "choice", "options": ["$100/day", "$200/day", "$300/day"]},
            }
        return {}

    @classmethod
    def _extract_slots(cls, state: Any, product: str, logger: logging.Logger) -> Dict[str, Any]:
        """Extract product-specific slots from the current user message with context awareness."""
        required_slots = cls._required_slots_for_product(product)
        slot_descriptions = cls._get_slot_descriptions(product)
        current_slots = state.session.get("slots", {}) or {}
        specs = cls._slot_specs(product)
        
        # Build product-specific slot context - only include slots that need extraction
        slot_info = []
        missing_slots = cls._get_missing_slots(current_slots, required_slots)
        
        # Only include information about missing or invalid slots to reduce prompt size
        for slot in missing_slots:
            description = slot_descriptions.get(slot, f"Information about {slot}")
            slot_info.append(f"- {slot}: {description} (current: not filled)")
        
        # If no missing slots, still check if user is updating existing slots
        if not missing_slots:
            for slot in required_slots:
                current_value = cls._get_slot_value(current_slots, slot)
                if current_value:
                    description = slot_descriptions.get(slot, f"Information about {slot}")
                    slot_info.append(f"- {slot}: {description} (current: {current_value})")
        
        # Include last bot question ONLY when it is a real slot question
        last_slot_name_ctx = state.session.get('_last_slot_name')
        last_bot_question = state.session.get('_last_slot_question', 'None') if last_slot_name_ctx else 'None'

        targets = missing_slots or required_slots
        valid_slots_str = ", ".join(targets)
        slot_meta_json = json.dumps({s: specs.get(s, {}) for s in targets})
        
        context = (
            f"Product: {product}\n"
            f"User message: {state.message}\n"
            f"Last bot question: {last_bot_question}\n"
            f"Valid slots: {valid_slots_str}\n"
            f"Slot meta (JSON): {slot_meta_json}\n\n"
            f"Slots to extract/update (focus on these only):\n" + "\n".join(slot_info)
        )
        
        logger.info("RecFlow.extract_slots: Starting extraction - product=%s, required_slots=%d, last_question='%s'", 
                   product, len(required_slots), last_bot_question[:100])
        
        # Use the slot extractor task
        from ..tasks import extract_slots_task
        extraction_result = run_direct_task(
            agent_obj=extract_slots_task.agent,
            agent_key="slot_extractor",
            task_key="extract_slots",
            context_text=context,
            logger=logger,
            label="slot_extractor.extract_slots",
        ) or {}
        
        logger.info("RecFlow.extract_slots: API output - keys=%s, user_needs_explanation=%s", 
                   list(extraction_result.keys()), extraction_result.get("user_needs_explanation"))
        
        # Check if user needs explanation
        if extraction_result.get("user_needs_explanation") and extraction_result.get("explanation"):
            logger.info("RecFlow.extract_slots: User needs explanation - length=%d", len(extraction_result.get("explanation", "")))
            return {"explanation_needed": extraction_result.get("explanation")}
        
        # Filter to only include product-specific slots
        filtered_result = {}
        for slot_name, slot_value in extraction_result.items():
            if slot_name in required_slots and slot_value and str(slot_value).strip():
                filtered_result[slot_name] = slot_value
        
        logger.info("RecFlow.extract_slots: Completed extraction - extracted_slots=%s", list(filtered_result.keys()))
        return filtered_result

    @classmethod  
    def _validate_slot(cls, slot_name: str, slot_value: str, product: str, state: Any, logger: logging.Logger) -> Dict[str, Any]:
        """Validate a single slot value."""
        logger.info("RecFlow.validate_slot: Starting validation - slot=%s, value='%s'", slot_name, slot_value)
        
        # Get current date for validation context
        date_str = ""
        try:
            now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
            date_str = f"Current date (Asia/Singapore): {now_sg.strftime('%d %B %Y')}"
        except Exception as e:
            logger.warning("RecFlow.validate_slot: Date generation failed - %s", str(e))

        # Load validation rules
        rules_block = ""
        try:
            base_dir = Path(__file__).resolve().parent.parent
            with open(base_dir / "config" / "slot_validation_rules.yaml", "r", encoding="utf-8") as rf:
                rules_yaml = yaml.safe_load(rf) or {}
            product_key = (product or "").lower()
            if product_key == "homeprotect360":
                product_key = "home"
            slot_key = (slot_name or "").lower()
            lines = rules_yaml.get(product_key, {}).get(slot_key, [])
            if lines:
                rules_block = "Validation rules:\n" + "\n".join(lines)
                logger.debug("RecFlow.validate_slot: Loaded %d validation rules for %s.%s", len(lines), product_key, slot_key)
        except Exception as e:
            logger.error("RecFlow.validate_slot: FAILED TO LOAD OR PARSE slot_validation_rules.yaml. Error: %s", str(e), exc_info=True)

        v_ctx = (
            f"Product: {product}\n"
            f"Slot: {slot_name}\n"
            f"Value: {slot_value}\n"
            f"User message: {state.message}\n"
            f"{date_str}\n"
            f"{rules_block}"
        ).strip()

        # Use existing slot validator
        from ..tasks import validate_slot_task as _vts
        validation_result = run_direct_task(
            agent_obj=_vts.agent,
            agent_key="slot_validator",
            task_key="validate_slot",
            context_text=v_ctx,
            logger=logger,
            label=f"validate_slot.{slot_name}",
        ) or {}
        
        logger.info("RecFlow.validate_slot: API output - valid=%s, has_normalized=%s, has_question=%s", 
                   validation_result.get("valid"), bool(validation_result.get("normalized_value")), 
                   bool(validation_result.get("question")))
        
        logger.info("RecFlow.validate_slot: Completed validation - slot=%s, valid=%s, has_normalized=%s", 
                   slot_name, validation_result.get("valid"), bool(validation_result.get("normalized_value")))
        return validation_result

    @classmethod
    def _ask_next_question(cls, product: str, missing_slot: str, current_slots: Dict[str, Any], 
                          user_wants_details: bool, state: Any, logger: logging.Logger) -> str:
        """Ask question for the next missing slot."""
        slot_descriptions = cls._get_slot_descriptions(product)
        description = slot_descriptions.get(missing_slot, f"information about {missing_slot}")
        
        # Generate custom question using question_asker agent with dynamic slot metadata
        specs = cls._slot_specs(product)
        slot_meta = specs.get(missing_slot, {})
        slot_type = slot_meta.get("type", "value")
        options = slot_meta.get("options", [])
        
        context = (
            f"Product: {product}\n"
            f"Missing slot: {missing_slot}\n"
            f"Slot type: {slot_type}\n"
            f"Options: {', '.join(options) if options else ''}\n"
            f"Slot description: {description}\n"
            f"Current slots: {current_slots}\n"
            f"User wants detailed explanations: {user_wants_details}"
        )
        
        logger.info("RecFlow.ask_question: Generating custom question for slot=%s, user_wants_details=%s", 
                   missing_slot, user_wants_details)
        
        from ..tasks import ask_question_task
        question_result = run_direct_task(
            agent_obj=ask_question_task.agent,
            agent_key="question_asker",
            task_key="ask_question",
            context_text=context,
            logger=logger,
            label="question_asker.ask_question",
        ) or {}
        
        logger.info("RecFlow.ask_question: API output - has_question=%s, keys=%s", 
                   bool(question_result.get("question")), list(question_result.keys()))
        
        question = question_result.get("question") or f"Could you please provide {missing_slot}?"
        # Mark last slot asked for yes/no disambiguation
        state.session["_last_slot_name"] = missing_slot
        state.session["_last_slot_question"] = question
        logger.info("RecFlow.ask_question: Generated question for slot=%s, length=%d", missing_slot, len(question))
        return question

    @classmethod
    def _generate_recommendation(cls, product: str, slots: Dict[str, Any], state: Any, logger: logging.Logger) -> str:
        """Generate final recommendation response."""
        logger.info("RecFlow.generate_recommendation: Starting recommendation generation - product=%s, slots_count=%d", 
                   product, len(slots))
        
        # Determine tier based on slots
        tier = None
        if (product or "").lower() == "travel":
            # Always start with Gold plan for Travel
            tier = "Gold"
        elif (product or "").lower() == "maid":
            coverage_above_mom = (cls._get_slot_value(slots, "coverage_above_mom_minimum") or "").strip().lower()
            if coverage_above_mom == "yes":
                tier = "Premier"
            elif coverage_above_mom == "no":
                tier = "Enhanced"
        elif (product or "").lower() == "personalaccident":
            try:
                amount = int(cls._get_slot_value(slots, "desired_amount"))
                if 500 <= amount <= 1000:
                    tier = "Silver"
                elif 1001 <= amount <= 2500:
                    tier = "Premier"
                elif 2501 <= amount <= 3500:
                    tier = "Platinum"
            except (ValueError, TypeError):
                tier = None # Should not happen if validation is correct
        elif (product or "").lower() == "home":
            try:
                amount = int(cls._get_slot_value(slots, "coverage_amount"))
                if amount <= 100000:
                    tier = "Silver"
                elif amount <= 200000:
                    tier = "Gold"
                else:
                    tier = "Platinum"
            except (ValueError, TypeError):
                tier = None
        elif (product or "").lower() == "early":
            # No tiers for Early CI product
            tier = None
        elif (product or "").lower() == "fraud":
            # Choose tier based on purchase frequency
            freq = cls._get_slot_value(slots, "purchase_frequency").strip().lower()
            if freq in ("daily", "everyday", "every day"):
                tier = "Platinum"
            else:
                tier = "Gold"
        elif (product or "").lower() == "hospital":
            # Choose tier based on desired daily coverage (100/200/300)
            raw = cls._get_slot_value(slots, "coverage") or ""
            digits = "".join(ch for ch in str(raw) if ch.isdigit())
            try:
                val = int(digits) if digits else 0
            except Exception:
                val = 0
            # Snap to nearest of 100, 200, 300
            choices = [100, 200, 300]
            if val <= 0:
                sel = 200
            else:
                sel = min(choices, key=lambda x: abs(x - val))
            tier = {100: "Silver", 200: "Premier", 300: "Titanium"}.get(sel, "Premier")
        
        logger.info("RecFlow.generate_recommendation: Determined tier=%s for product=%s", tier, product)
        
        # Get benefits
        benefits_text = ""
        try:
            benefits_text = benefits_tool.run(product=product)
            logger.info("RecFlow.generate_recommendation: Benefits tool output - length=%d, has_content=%s", 
                       len(benefits_text), bool(benefits_text.strip()))
            logger.info("RecFlow.generate_recommendation: Retrieved benefits - length=%d", len(benefits_text))
        except Exception as e:
            logger.error("RecFlow.generate_recommendation: Benefits retrieval failed - %s", str(e))
        
        # Load recommendation templates
        rec_templates = {}
        try:
            base_dir = Path(__file__).resolve().parent.parent
            with open(base_dir / "config" / "recommendation_response.yaml", "r", encoding="utf-8") as rf:
                rec_templates = yaml.safe_load(rf) or {}
            logger.debug("RecFlow.generate_recommendation: Loaded templates for products: %s", list(rec_templates.keys()))
        except Exception as e:
            logger.warning("RecFlow.generate_recommendation: Template loading failed - %s", str(e))

        product_key = (product or "").lower()
        tpl = rec_templates.get(product_key) or {}
        if product_key == "maid":
            add_ons_pref = cls._get_slot_value(slots, "add_ons") or "not_required"
            sys_t = (tpl.get("system") or "").format(tier=tier or "", add_ons=add_ons_pref)
            usr_t = (tpl.get("user") or "").format(tier=tier or "", add_ons=add_ons_pref, benefits=benefits_text or "")
        elif product_key == "travel":
            # Build generic travel advisory using destination (no RAG retrieval to avoid irrelevant content)
            destination = (cls._get_slot_value(slots, "destination") or "").strip()
            # Use simple, generic medical cost advisory for all destinations
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
            logger.info("RecFlow.generate_recommendation: Using generic travel advisory for destination=%s", destination)
            sys_t = (tpl.get("system") or "").format(tier=tier or "", destination=destination or "")
            usr_t = (tpl.get("user") or "").format(tier=tier or "", benefits=benefits_text or "", advisory=advisory or "", destination=destination or "")
        else:
            sys_t = (tpl.get("system") or "").format(tier=tier or "")
            usr_t = (tpl.get("user") or "").format(tier=tier or "", benefits=benefits_text or "")

        response = ""
        if (product or "").lower() == "early":
            # Special handling: Early has no tiers and a fixed coverage suggestion
            try:
                base_dir = Path(__file__).resolve().parent.parent
                with open(base_dir / "config" / "recommendation_response.yaml", "r", encoding="utf-8") as rf:
                    rec_templates = yaml.safe_load(rf) or {}
            except Exception:
                rec_templates = {}
            tpl_e = rec_templates.get("early") or {}
            sys_e = (tpl_e.get("system") or "")
            usr_e = (tpl_e.get("user") or "").format(benefits=benefits_text or "")
            if sys_e and usr_e:
                try:
                    txt = azure_response_llm.call(messages=[
                        {"role": "system", "content": sys_e},
                        {"role": "user", "content": usr_e},
                    ])
                    response = str(txt).strip()
                except Exception as e:
                    logger.error("RecFlow.generate_recommendation: Early LLM call failed - %s", str(e))
        elif sys_t and usr_t:
            logger.info("RecFlow.generate_recommendation: Calling LLM with templates - system_len=%d, user_len=%d", 
                       len(sys_t), len(usr_t))
            try:
                txt = azure_response_llm.call(messages=[
                    {"role": "system", "content": sys_t},
                    {"role": "user", "content": usr_t},
                ])
                response = str(txt).strip()
                logger.info("RecFlow.generate_recommendation: LLM response generated - length=%d", len(response))
            except Exception as e:
                logger.error("RecFlow.generate_recommendation: LLM call failed - %s", str(e))
        
        logger.info("RecFlow.generate_recommendation: Completed - tier=%s, response_len=%d", tier, len(response))
        return response

    @classmethod
    def _handle_fraud_intro(cls, state: Any, logger: logging.Logger) -> Optional[str]:
        """Handle Fraud guided intro funnel. Returns "__done__" when a reply was sent, or None to continue."""
        msg_low = (state.message or "").strip().lower()
        stage = state.session.get("fraud_stage")

        def _yn(s: str) -> str:
            # Normalize simple chat acknowledgements, tolerate punctuation and elongations.
            s = (s or "").strip().lower()
            # Trim trailing punctuation/ellipsis
            s = re.sub(r"[\s\.\!\?\,\;\:…]+$", "", s)
            # Collapse internal multiple spaces
            s = re.sub(r"\s+", " ", s)

            yes = {
                # Core
                "yes", "y", "yeah", "yep", "yup", "ya", "yah",
                # Elongations
                "yess", "yesss", "yessss", "yupp", "yuppp", "yass", "yasss",
                # Polite/ack
                "sure", "ok", "okay", "k", "kk", "alright", "please", "pls", "plz",
                # Phrases
                "go ahead", "go for it", "do it", "proceed", "continue", "sounds good", "yes please", "yes pls", "ok please", "ok pls",
            }
            no = {
                # Core
                "no", "n", "nope", "nah",
                # Phrases / defer
                "not now", "later", "maybe later", "no thanks", "no thank you", "not interested", "no need", "no need thanks",
                # Negations
                "don't", "dont", "do not", "stop", "cancel", "skip",
                # Soft declines
                "i'm fine", "im fine", "all good", "no pls", "no please",
            }
            if s in yes:
                return "yes"
            if s in no:
                return "no"
            return "other"

        # Bootstrap stage only if intro not started and recommendation hasn't begun
        if not stage:
            if (state.session.get("recommendation_status") or "").strip().lower() == "in_progress":
                return None
            state.session["fraud_stage"] = "await_learn_more"
            # Mark recommendation as in-progress to guarantee orchestrator bypass on next turn
            state.session["recommendation_status"] = "in_progress"
            q = "A great choice! Would you like to learn more about our Fraud Protect360 product?"
            state.session["last_question"] = q
            state.reply = q
            return "__done__"

        if stage == "await_learn_more":
            ans = _yn(msg_low)
            if ans == "yes":
                info = (
                    "Every day, Singaporeans lose thousands to online scams\n"
                    "Fraud Protect360 helps you recover financial losses due to:\n"
                    "• Online payment scams\n"
                    "• Phishing / malware attacks\n"
                    "• Identity theft\n"
                    "• Fake e-commerce transactions\n\n"
                    "Want to see how it protects you in real life situations?"
                )
                state.session["fraud_stage"] = "await_example"
                state.session["last_question"] = "Want to see how it protects you in real life situations?"
                state.reply = info
                return "__done__"
            if ans == "no":
                q = "Would you like me to recommend a personalized coverage for you?"
                state.session["fraud_stage"] = "await_recommendation"
                state.session["last_question"] = q
                state.reply = q
                return "__done__"
            # Re-ask
            q = "Would you like to learn more about our Fraud Protect360 product?"
            state.session["last_question"] = q
            state.reply = q
            return "__done__"

        if stage == "await_example":
            ans = _yn(msg_low)
            if ans == "yes":
                example = (
                    "Imagine this, you made a purchase on an online platform and did not receive your item and the seller became unresponsive – under our Fraud Protect360 you are covered up to $10,000 for your undelivered online purchase!\n\n"
                    "Would you like me to recommend a personalized coverage for you?"
                )
                state.session["fraud_stage"] = "await_recommendation"
                state.session["last_question"] = "Would you like me to recommend a personalized coverage for you?"
                state.reply = example
                return "__done__"
            if ans == "no":
                q = "Would you like me to recommend a personalized coverage for you?"
                state.session["fraud_stage"] = "await_recommendation"
                state.session["last_question"] = q
                state.reply = q
                return "__done__"
            # Re-ask
            q = "Want to see how it protects you in real life situations?"
            state.session["last_question"] = q
            state.reply = q
            return "__done__"

        if stage == "await_recommendation":
            ans = _yn(msg_low)
            if ans == "yes":
                # Proceed into slot collection next; clear stage and ensure in_progress
                state.session.pop("fraud_stage", None)
                state.session["recommendation_status"] = "in_progress"
                # This was meta-consent, not a slot answer. Clear non-slot question and
                # skip extraction for this turn to avoid consuming "Yes" as a slot value.
                state.session.pop("last_question", None)
                state.session["_skip_extraction_once"] = True
                return None
            if ans == "no":
                promo = (
                    "No problem. Fraud Protect360 covers up to $10,000 for key fraud-related losses, with higher Accidental Death/PTD, medical reimbursement, hospital cash, and emergency transport benefits on the Platinum plan. If you'd like a personalized recommendation later, just come back anytime and I'll help you choose."
                )
                # Clean up intro state and recommendation flag when user declines
                state.session.pop("fraud_stage", None)
                state.session.pop("last_question", None)
                state.session.pop("recommendation_status", None)
                state.reply = promo
                return "__done__"
            # Re-ask
            q = "Would you like me to recommend a personalized coverage for you?"
            state.session["last_question"] = q
            state.reply = q
            return "__done__"

        return None

    @classmethod
    def handle(cls, state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        """Main entry point for simplified recommendation flow."""
        logger.info("RecFlow.handle: Starting recommendation flow - message_len=%d", len(state.message or ""))
        
        # Always check for product identification/switches from current message
        current_product = state.product or state.session.get("product")
        recommendation_status = state.session.get("recommendation_status")
        rec_prev_prod_q = bool(state.session.get("_last_rec_prod_q"))

        # Short-circuit during Fraud guided intro: skip product identification and slot analysis
        if (state.session.get("fraud_stage") or "").strip():
            logger.info(
                "RecFlow.handle: Fraud guided intro active (%s) - skipping product identification and slot analysis",
                state.session.get("fraud_stage"),
            )
            if not (state.product or state.session.get("product")):
                state.product = "Fraud"
                state.session["product"] = "Fraud"
            res = cls._handle_fraud_intro(state, logger)
            if res is not None:
                return res
            # If res is None, user accepted to proceed; continue to slot collection

        # Allow lexical product switch detection during recommendation (in_progress or done)
        if recommendation_status in ("in_progress", "done"):
            try:
                # Bypass early lexical switching if a product clarification is outstanding
                clarifying_product = bool(state.session.get("_last_rec_prod_q") or state.session.get("_last_info_prod_q"))
                cand_list = [] if clarifying_product else lexical_product_candidates(state.message or "")
            except Exception:
                cand_list = []
            session_or_state = state.product or state.session.get("product")
            # Fully LLM-driven: probe whenever we have any candidates
            should_probe = bool(cand_list)
            if should_probe:
                try:
                    from ..tasks import identify_product_task as _ipt
                    from ..prompt_runner import run_direct_task as _rdt
                    lines = [
                        f"Message: {state.message}",
                        f"Session product: {session_or_state}",
                        "Candidates:",
                    ]
                    for c in cand_list[:3]:
                        reasons = ",".join(c.get("reasons", [])) if isinstance(c.get("reasons"), list) else ""
                        lines.append(
                            f"- product: {c.get('product')}, score: {c.get('score')}, polarity: {c.get('polarity')}, reasons: {reasons}"
                        )
                    lines.append("")
                    lines.append(
                        "Instruction: Choose the product explicitly requested by the CURRENT message. "
                        "Prefer positive candidates; treat negated candidates as not requested. "
                        "If ambiguous, return an empty product or a short clarifying question."
                    )
                    probe_ctx = "\n".join(lines)
                    probe = _rdt(
                        agent_obj=_ipt.agent,
                        agent_key="product_identifier",
                        task_key="identify_product",
                        context_text=probe_ctx,
                        logger=logger,
                        label="product_identifier.double_confirm.on_rec",
                    ) or {}
                    confirmed = (probe.get("product") or "").strip()
                    conf = float(probe.get("confidence") or 0.0)
                    # If identifier provides a clarification question, ask it now and short-circuit
                    ask_q = (probe.get("question") or "").strip()
                    if ask_q and not confirmed:
                        state.session["_last_rec_prod_q"] = True
                        state.session["last_question"] = ask_q
                        state.reply = ask_q
                        logger.info("RecFlow: Product clarification requested by identifier; asking and short-circuiting")
                        return "__done__"
                    if confirmed and conf >= 0.8:
                        # Commit switch and clear rec-specific state to avoid leakage
                        state.product = confirmed
                        state.session["product"] = confirmed
                        state.session.pop("slots", None)
                        state.session.pop("recommendation_status", None)
                        state.session.pop("last_question", None)
                        state.session.pop("_last_rec_prod_q", None)
                        # Refresh local tracker so downstream logic uses updated product
                        current_product = confirmed
                        logger.info("RecFlow: Product switch confirmed via lexical+LLM - %s (state cleared)", confirmed)
                except Exception:
                    pass
        
        from ..tasks import identify_product_task
        from ..prompt_runner import run_direct_task

        logger.info("RecFlow.handle: Product identification - current_product=%s, rec_status=%s, prev_prod_q=%s", current_product, recommendation_status, rec_prev_prod_q)

        product: Optional[str] = None

        # Special case: previous turn asked for product clarification in RecFlow
        if rec_prev_prod_q:
            prod_result = run_direct_task(
                agent_obj=identify_product_task.agent,
                agent_key="product_identifier",
                task_key="identify_product",
                context_text=f"Message: {state.message}",
                logger=logger,
                label="product_identifier.identify_product.rec_flow_prod_clarify",
            ) or {}
            product = prod_result.get("product") or current_product
            if product:
                state.product = product
                state.session["product"] = product
            # Clear the clarification flag regardless
            state.session.pop("_last_rec_prod_q", None)
            # Fraud: start guided intro instead of slots
            if (product or "").strip().lower() == "fraud":
                state.session.pop("recommendation_status", None)
                state.session["fraud_stage"] = "await_learn_more"
                q = "A great choice! Would you like to learn more about our Fraud Protect360 product?"
                state.session["last_question"] = q
                state.reply = q
                return "__done__"
            # Ensure recommendation is marked in progress for slot collection
            state.session["recommendation_status"] = "in_progress"

            # Skip slot extraction/validation this turn; ask next missing slot directly (or proceed if none)
            required_slots = cls._required_slots_for_product(product)
            current_slots = state.session.get("slots", {}) or {}
            user_wants_details = state.session.get("user_wants_details", True)
            missing_slots = cls._get_missing_slots(current_slots, required_slots)
            if missing_slots:
                next_slot = missing_slots[0]
                question = cls._ask_next_question(product, next_slot, current_slots, user_wants_details, state, logger)
                # Save and reply
                state.session["last_question"] = question
                state.reply = question
                logger.info("RecFlow.handle: Product clarified; asking next slot=%s", next_slot)
                return "__done__"
            # No missing slots (e.g., Car/Early) → generate immediately
            logger.info("RecFlow.handle: Product clarified; no missing slots. Proceeding to generation.")
        else:
            # Normal path with re-identification gating: only run identifier on first pass
            if current_product and (recommendation_status in ("in_progress", "done")):
                product = current_product
                logger.info("RecFlow.handle: Skipping product identification (cached product in session)")
            else:
                # Identify product synchronously
                prod_result = run_direct_task(
                    agent_obj=identify_product_task.agent,
                    agent_key="product_identifier",
                    task_key="identify_product",
                    context_text=f"Message: {state.message}\nSession product: {current_product}",
                    logger=logger,
                    label="product_identifier.identify_product.rec_flow",
                ) or {}
                identified_product = prod_result.get("product")
                logger.info(
                    "RecFlow.handle: Product identification API output - product=%s, confidence=%s, has_question=%s, keys=%s",
                    identified_product,
                    prod_result.get("confidence"),
                    bool(prod_result.get("question")),
                    list(prod_result.keys()),
                )

                # IMPORTANT: If there's a clarification question, ask it FIRST, even if a product was tentatively identified
                # This handles cases where confidence is low or the product identifier wants to confirm
                if prod_result.get("question"):
                    question = prod_result["question"]
                    state.reply = question
                    state.session["recommendation_status"] = "in_progress"
                    # Mark that next user message is a product clarification for RecFlow
                    state.session["_last_rec_prod_q"] = True
                    # Optionally save the tentative product for context
                    if identified_product:
                        state.session["_tentative_product"] = identified_product
                    logger.info(
                        "RecFlow.handle: Product clarification question provided (has_question=True), requesting clarification"
                    )
                    return "__done__"
                elif identified_product and identified_product != current_product:
                    logger.info(
                        "RecFlow.handle: Product switch detected - %s -> %s, clearing previous state",
                        current_product,
                        identified_product,
                    )
                    state.session.pop("slots", None)
                    state.session.pop("recommendation_status", None)
                    product = identified_product
                    state.product = product
                    state.session["product"] = product
                elif identified_product:
                    product = identified_product
                    state.product = product
                    state.session["product"] = product
                elif current_product:
                    product = current_product
                else:
                    # No product and no question - ask default clarification
                    question = "What type of insurance are you interested in for the recommendation: Travel, Maid, Car, Personal Accident, Home, or Early?"
                    state.reply = question
                    state.session["recommendation_status"] = "in_progress"
                    # Mark that next user message is a product clarification for RecFlow
                    state.session["_last_rec_prod_q"] = True
                    logger.info(
                        "RecFlow.handle: No product identified, requesting default clarification and setting status to in_progress"
                    )
                    return "__done__"
        
        # --- Pre-check: Explicit Purchase Intent ---
        # If the user explicitly wants to buy/purchase the detected product and we have sufficient context
        # (either slots filled or flow completed), serve the link immediately to avoid re-running the flow.
        msg_lower_purch = (state.message or "").lower()
        purchase_keywords = {
            "purchase", "buy", "get this plan", "get this policy", "sign up", 
            "apply now", "proceed", "buy now", "purchase now", "get this", "want this plan",
            "would purchase"
        }
        has_purchase_kw = any(pk in msg_lower_purch for pk in purchase_keywords)
        is_refusal = any(n in msg_lower_purch for n in ["don't", "do not", "no ", "not "])
        
        if has_purchase_kw and not is_refusal:
            # Check context: are We ready to sell? (Slots valid)
            _curr = state.session.get("slots", {}) or {}
            _req = cls._required_slots_for_product(product)
            _missing = cls._get_missing_slots(_curr, _req)
            
            if not _missing:
                product_label = (product or "Insurance").title()
                state.reply = (
                    f"Excellent choice! You can complete your purchase for {product_label} here:\n"
                    "https://www.hlas.com.sg/buy-online\n\n"
                    "Let me know if you need help with anything else!"
                )
                logger.info("RecFlow.handle: Purchase intent detected with valid context; returning purchase link.")
                return "__done__"

        # Check if recommendation is already complete for this product
        recommendation_status = state.session.get("recommendation_status")
        logger.info("RecFlow.handle: Current recommendation status=%s", recommendation_status)
        
        if recommendation_status == "done":
            message_text = (state.message or "").strip()
            message_lower = message_text.lower()
            normalized_message = message_lower.rstrip("!.?")

            acknowledgement_phrases = {
                "thanks",
                "thank you",
                "thank u",
                "thx",
                "thanks!",
                "ok",
                "okay",
                "ok thanks",
                "ok thank you",
                "great",
                "awesome",
                "cool",
                "got it",
                "all good",
                "no",
                "no thanks",
                "no thank you",
                "that's all",
                "bye",
                "bye bye",
                "goodbye",
            }

            if not message_text or normalized_message in acknowledgement_phrases:
                state.reply = "You already have a recommendation. How else can I help you?"
                logger.info("RecFlow.handle: Recommendation acknowledged; awaiting further instructions.")
                return "__done__"

            restart_keywords = [
                "new recommendation",
                "fresh recommendation",
                "start over",
                "restart",
                "again",
                "different recommendation",
            ]
            wants_new_rec = any(keyword in message_lower for keyword in restart_keywords)
            indicates_change = any(token in message_lower for token in [
                "change",
                "adjust",
                "modify",
                "update",
                "rather",
                "instead",
                "switch",
                "different",
                "another",
            ])
            contains_question = "?" in message_text

            if wants_new_rec or "recommend" in message_lower or contains_question or indicates_change:
                logger.info("RecFlow.handle: User is initiating a new recommendation; resetting previous state.")
            else:
                logger.info("RecFlow.handle: Defaulting to new recommendation flow; clearing previous state for fresh inputs.")

            state.session.pop("recommendation_status", None)
            state.session.pop("slots", None)
            state.session.pop("last_question", None)
            state.session.pop("_early_existing_cover_notice", None)
        
        # Get current slots and required slots
        current_slots = state.session.get("slots", {}) or {}
        required_slots = cls._required_slots_for_product(product)
        
        # Check user preference for detailed explanations
        user_wants_details = state.session.get("user_wants_details", True)  # Default to True
        
        logger.info("RecFlow.handle: Slot analysis - product=%s, current_slots=%s, required_slots=%s, user_wants_details=%s", 
                   product, list(current_slots.keys()), required_slots, user_wants_details)
        
        # Set recommendation status to in_progress if we have slots to collect
        if (product or "").lower() != "fraud" and required_slots and recommendation_status != "in_progress":
            state.session["recommendation_status"] = "in_progress"
            logger.info("RecFlow.handle: Set recommendation status to 'in_progress'")
        
        # Handle Car insurance (no slots required) - direct recommendation
        if (product or "").lower() == "car":
            logger.info("RecFlow.handle: Processing car insurance recommendation (no slots required)")
            
            benefits_text = ""
            try:
                benefits_text = benefits_tool.run(product=product)
                logger.info("RecFlow.handle: Retrieved car benefits - length=%d", len(benefits_text))
            except Exception as e:
                logger.error("RecFlow.handle: Car benefits retrieval failed - %s", str(e))
                
            rec_templates = {}
            try:
                base_dir = Path(__file__).resolve().parent.parent
                with open(base_dir / "config" / "recommendation_response.yaml", "r", encoding="utf-8") as rf:
                    rec_templates = yaml.safe_load(rf) or {}
            except Exception as e:
                logger.warning("RecFlow.handle: Car template loading failed - %s", str(e))

            tpl = rec_templates.get("car") or {}
            sys_t = (tpl.get("system") or "")
            usr_t = (tpl.get("user") or "").format(benefits=benefits_text or "")
            
            car_response = ""
            if sys_t and usr_t:
                logger.info("RecFlow.handle: Generating car recommendation with templates")
                try:
                    txt = azure_response_llm.call(messages=[
                        {"role": "system", "content": sys_t},
                        {"role": "user", "content": usr_t},
                    ])
                    car_response = str(txt).strip()
                    logger.info("RecFlow.handle: Car LLM response generated - length=%d", len(car_response))
                except Exception as e:
                    logger.error("RecFlow.handle: Car LLM call failed - %s", str(e))
            
            state.reply = car_response
                
            # Mark as complete
            state.session["recommendation_status"] = "done"
            
            # Clear comparison/summary states to avoid unintended bypass
            state.session.pop("compare_pending", None)
            state.session.pop("summary_pending", None)

            logger.info("RecFlow.handle: Car recommendation completed - response_len=%d", len(str(state.reply or "")))
            return "__done__"
        
        # Fraud pre-recommendation guided intro (lightweight state machine)
        if (product or "").lower() == "fraud":
            stage = state.session.get("fraud_stage")
            rec_stat = (state.session.get("recommendation_status") or "").strip().lower()
            if stage or rec_stat != "in_progress":
                res = cls._handle_fraud_intro(state, logger)
                if res is not None:
                    return res

        # Handle Early FAQs that can appear anytime
        if (product or "").lower() == "early":
            msg_low = (state.message or "").lower()
            if ("young" in msg_low and "healthy" in msg_low) or ("do i really need" in msg_low and "this" in msg_low):
                state.reply = (
                    "Serious illnesses can occur at any age. Buying CI protection earlier often means lower premiums and getting covered before any health issues arise"
                )
                logger.info("RecFlow.handle: Early FAQ answered - young/healthy question")
                return "__done__"
            if "never claim" in msg_low:
                state.reply = (
                    "The main value is peace of mind — that you and your family are protected from unexpected financial stress. Some plans also offer partial refunds or conversion options at maturity if you’d like to consider them."
                )
                logger.info("RecFlow.handle: Early FAQ answered - never claim question")
                return "__done__"
        
        # If we just received meta-consent (e.g., from Fraud intro), skip extraction this turn
        if state.session.pop("_skip_extraction_once", False):
            # Ask the first missing slot directly
            missing_slots = cls._get_missing_slots(current_slots, required_slots)
            if missing_slots:
                next_slot = missing_slots[0]
                question = cls._ask_next_question(product, next_slot, current_slots, user_wants_details, state, logger)
                state.session["last_question"] = question
                state.reply = question
                logger.info("RecFlow.handle: Skip extraction once; asking next slot=%s", next_slot)
                return "__done__"
            # No missing slots; proceed to generation below

        # Extract/update slots from current message
        extracted_slots = cls._extract_slots(state, product, logger)
        
        # Check if user needs explanation
        if "explanation_needed" in extracted_slots:
            explanation = extracted_slots["explanation_needed"]
            state.reply = explanation
            logger.info("RecFlow.handle: User explanation provided - length=%d", len(explanation))
            return "__done__"
        
        # First: Assign extracted slot values (slot extractor's job)
        updated_slots = dict(current_slots)
        slots_to_validate = []
        
        for slot_name, slot_value in extracted_slots.items():
            if slot_name in required_slots:
                if slot_value and str(slot_value).strip():
                    # Check if this is a new/different value or if slot doesn't exist
                    existing_value = cls._get_slot_value(updated_slots, slot_name)
                    is_already_valid = cls._is_slot_valid(updated_slots, slot_name)
                    
                    if existing_value != slot_value or not is_already_valid:
                        # New/different value or invalid slot - assign and mark for validation
                        cls._set_slot_value(updated_slots, slot_name, slot_value, False)
                        slots_to_validate.append(slot_name)
                        logger.info("RecFlow.handle: Slot extracted and assigned - %s=%s (needs validation)", slot_name, slot_value)
                    else:
                        # Same value and already valid - skip validation
                        logger.info("RecFlow.handle: Slot unchanged and valid - %s=%s (skipping validation)", slot_name, slot_value)
                else:
                    # Empty value: remove the slot if it exists
                    if slot_name in updated_slots:
                        updated_slots.pop(slot_name)
                        logger.info("RecFlow.handle: Slot removed - %s (empty value)", slot_name)
        
        # Second: Validate only slots that need validation (slot validator's job)
        validation_failed_slot = None
        validation_failed_question = None
        
        # Build validation targets (skip already validated)
        validate_targets = [s for s in slots_to_validate if not cls._is_slot_valid(updated_slots, s)]
        if validate_targets:
            # Run validations sequentially and stop at first invalid to preserve behavior
            for slot_name in validate_targets:
                slot_val = cls._get_slot_value(updated_slots, slot_name)
                logger.info("RecFlow.handle: Starting slot validation (sequential) - %s=%s", slot_name, slot_val)
                validation_result = cls._validate_slot(slot_name, slot_val, product, state, logger) or {}
                if validation_result.get("valid") and validation_result.get("normalized_value"):
                    # Valid: update with normalized value and mark as validated
                    cls._set_slot_value(updated_slots, slot_name, validation_result["normalized_value"], True)
                    logger.info("RecFlow.handle: Slot validated successfully - %s=%s", slot_name, validation_result["normalized_value"])
                    # Early-specific educational message when existing_cover is yes
                    if (product or "").lower() == "early" and slot_name == "existing_cover" and str(validation_result.get("normalized_value")).lower() == "yes":
                        state.session["_early_existing_cover_notice"] = (
                            "That’s excellent — medical insurance helps pay your hospital and treatment bills.\n"
                            "Critical Illness insurance complements it by giving you a cash payout, which you can use for income replacement, rehabilitation, or other expenses that aren’t covered by hospital plans"
                        )
                else:
                    # Invalid: remove from slots and ask for clarification
                    updated_slots.pop(slot_name, None)
                    logger.info("RecFlow.handle: Slot validation failed - %s (removed from slots)", slot_name)
                    validation_failed_slot = slot_name
                    
                    # Build a user-facing message that always includes the reason when available
                    reason_text = (validation_result.get("reason") or "").strip()
                    question_text = (validation_result.get("question") or "").strip()
                    # Use original user input for this slot
                    slot_value = cls._get_slot_value(updated_slots, slot_name) or ""
                    user_input = cls._get_slot_value({slot_name: slot_value}, slot_name)

                    if not question_text:
                        # Fallback: construct a helpful question with constraints
                        if slot_name == "destination":
                            question_text = "Please provide a country name (not a city). Which country will you be travelling to?"
                        elif slot_name == "coverage_scope":
                            question_text = "Please choose: myself, family, group of adults (up to 20), or group of families (up to 10)."
                        else:
                            question_text = f"Please provide a valid {slot_name.replace('_', ' ')}."

                    # If we have a reason but it's not in the question, prepend it
                    if reason_text and reason_text.lower() not in question_text.lower():
                        validation_failed_question = f"{reason_text}. {question_text}"
                    elif question_text:
                        validation_failed_question = question_text
                    else:
                        # Final fallback with user input context
                        validation_failed_question = f"'{user_input}' is not valid. Please provide a valid {slot_name.replace('_', ' ')}."
                    
                    logger.info("RecFlow.handle: Generated validation failure question: '%s'", validation_failed_question)
                    break  # Stop after first invalid to match existing behavior
        
        # Update session with processed slots
        state.session["slots"] = updated_slots
        logger.info("RecFlow.handle: Updated session slots - count=%d, valid_slots=%s", 
                   len(updated_slots), [k for k, v in updated_slots.items() if isinstance(v, dict) and v.get("valid")])
        
        # If validation failed, ask for clarification immediately
        if validation_failed_slot:
            state.reply = validation_failed_question
            logger.info("RecFlow.handle: Validation failed, asking for clarification - slot=%s", validation_failed_slot)
            return "__done__"
        
        # Check if all slots are filled and valid
        missing_slots = cls._get_missing_slots(updated_slots, required_slots)
        
        if missing_slots:
            # Ask for the next missing slot
            next_slot = missing_slots[0]
            logger.info("RecFlow.handle: Missing slots detected - missing=%s, asking_for=%s", missing_slots, next_slot)
            
            question = cls._ask_next_question(product, next_slot, updated_slots, user_wants_details, state, logger)
            
            # Prefix Early notice if present
            prefix = state.session.pop("_early_existing_cover_notice", None) if (product or "").lower() == "early" else None
            if prefix:
                combined = f"{prefix}\n\n{question}"
            else:
                combined = question
            
            # Save the question for context in next turn
            state.session["last_question"] = combined
            state.reply = combined
            logger.info("RecFlow.handle: Asked question for missing slot - slot=%s, question_len=%d", next_slot, len(question))
            return "__done__"
        else:
            # All slots filled - generate recommendation
            logger.info("RecFlow.handle: All slots filled, generating recommendation - slot_count=%d", len(updated_slots))
            recommendation = cls._generate_recommendation(product, updated_slots, state, logger)
            
            # Mark recommendation as complete
            state.session["recommendation_status"] = "done"
            # Mark last completed flow to help downstream logic (e.g., follow-up context suppression)
            try:
                state.session["last_completed"] = "recommendation"
            except Exception:
                pass
            # Clear any lingering slot question now that recommendation is complete
            try:
                state.session.pop("last_question", None)
            except Exception:
                pass

            # Clear any fraud intro remnants and rec clarification flags
            state.session.pop("fraud_stage", None)
            state.session.pop("_last_rec_prod_q", None)

            # Clear comparison/summary states to avoid unintended bypass
            state.session.pop("compare_pending", None)
            state.session.pop("summary_pending", None)

            state.reply = recommendation
            logger.info("RecFlow.handle: Recommendation flow completed - status=done, response_len=%d", len(recommendation))
            return "__done__"
