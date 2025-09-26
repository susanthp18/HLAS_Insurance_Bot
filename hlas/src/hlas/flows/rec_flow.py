from typing import Dict, Any, Optional
import logging
from pathlib import Path
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo


from ..prompt_runner import run_direct_task
from ..tools.benefits_tool import benefits_tool
from ..agents import recommendation_responder


class RecFlowHelper:
    """Simplified recommendation flow with clear separation of concerns.
    
    Architecture:
    1. slot_extractor: Extracts all possible slots from user message
    2. questionnaire_agent: Only asks questions for missing slots
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
        if p == "car":
            return []  # Car insurance has no slots to collect
        return []

    @staticmethod
    def _get_slot_descriptions(product: Optional[str]) -> Dict[str, str]:
        """Get descriptions for each slot to help extraction."""
        descriptions = {
            "travel": {
                "destination": "Country or region the user is traveling to",
                "travel_duration": "Number of days for the trip",
                "pre_existing_medical_condition": "Whether user has any pre-existing medical conditions (yes/no)",
                "plan_preference": "User's coverage preference (budget/comprehensive)"
            },
            "maid": {
                "duration_of_insurance": "How long the insurance coverage is needed (months/years)",
                "maid_country": "Country where the domestic helper is from",
                "coverage_above_mom_minimum": "Whether user wants coverage beyond MOM minimum (yes/no)",
                "add_ons": "Whether user wants additional add-on coverages (required/not_required)"
            },
            "car": {}  # Car insurance has no slots
        }
        return descriptions.get((product or "").lower(), {})

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

    @classmethod
    def _extract_slots(cls, state: Any, product: str, logger: logging.Logger) -> Dict[str, Any]:
        """Extract product-specific slots from the current user message with context awareness."""
        required_slots = cls._required_slots_for_product(product)
        slot_descriptions = cls._get_slot_descriptions(product)
        current_slots = state.session.get("slots", {}) or {}
        
        # Build product-specific slot context
        slot_info = []
        for slot in required_slots:
            current_value = cls._get_slot_value(current_slots, slot) or "not filled"
            description = slot_descriptions.get(slot, f"Information about {slot}")
            slot_info.append(f"- {slot}: {description} (current: {current_value})")
        
        # Include last bot question for context (crucial for yes/no disambiguation)
        last_bot_question = state.session.get('last_question', 'None')
        
        context = (
            f"Product: {product}\n"
            f"User message: {state.message}\n"
            f"Last bot question: {last_bot_question}\n\n"
            f"Slots to extract/update:\n" + "\n".join(slot_info)
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
            slot_key = (slot_name or "").lower()
            lines = rules_yaml.get(product_key, {}).get(slot_key, [])
            if lines:
                rules_block = "Validation rules:\n" + "\n".join(lines)
                logger.debug("RecFlow.validate_slot: Loaded %d validation rules for %s.%s", len(lines), product_key, slot_key)
        except Exception as e:
            logger.warning("RecFlow.validate_slot: Failed to load validation rules - %s", str(e))

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
        
        # Preferred questions for specific slots
        preferred_questions = {
            # Travel slots
            "destination": "Could you please tell me where you will be travelling to?",
            "travel_duration": "How long will your trip be?",
            "pre_existing_medical_condition": "Do you have any pre-existing medical conditions that we should note for this trip?",
            "plan_preference": "Do you prefer a budget-friendly plan or a comprehensive plan?",
            # Maid slots
            "maid_country": "May I know which country is your maid from?",
            "coverage_above_mom_minimum": "Would you like coverage for medical expenses beyond the MOM minimum requirement?",
            "add_ons": "Would you be also interested in add-on coverages such as increased Hospital expenses, waiver of excess and medical examination package cover?"
        }
        
        # Use preferred question if available
        if missing_slot in preferred_questions:
            question = preferred_questions[missing_slot]
            logger.info("RecFlow.ask_question: Using preferred question for slot=%s", missing_slot)
            return question
        
        # Generate custom question using question_asker agent
        context = (
            f"Product: {product}\n"
            f"Missing slot: {missing_slot}\n" 
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
            pref = (cls._get_slot_value(slots, "plan_preference") or "").strip().lower()
            if pref == "budget":
                tier = "Silver"
            elif pref == "comprehensive":
                tier = "Gold"
        elif (product or "").lower() == "maid":
            coverage_above_mom = (cls._get_slot_value(slots, "coverage_above_mom_minimum") or "").strip().lower()
            if coverage_above_mom == "yes":
                tier = "Premier"
            elif coverage_above_mom == "no":
                tier = "Enhanced"
        
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
        sys_t = (tpl.get("system") or "").format(tier=tier or "")
        
        if product_key == "maid":
            add_ons_pref = cls._get_slot_value(slots, "add_ons") or "not_required"
            usr_t = (tpl.get("user") or "").format(tier=tier or "", add_ons=add_ons_pref, benefits=benefits_text or "")
        else:
            usr_t = (tpl.get("user") or "").format(tier=tier or "", benefits=benefits_text or "")

        response = ""
        if sys_t and usr_t:
            logger.info("RecFlow.generate_recommendation: Calling LLM with templates - system_len=%d, user_len=%d", 
                       len(sys_t), len(usr_t))
            try:
                final = run_direct_task(
                    agent_obj=recommendation_responder,
                    agent_key="recommendation_responder", 
                    task_key="synthesize_response",
                    context_text=f"[System]\n{sys_t}\n\n[User]\n{usr_t}",
                    logger=logger,
                    label="recommendation.response_synthesis",
                ) or {}
                response = final.get("response") if isinstance(final, dict) else str(final)
                logger.info("RecFlow.generate_recommendation: LLM response generated - length=%d", len(response))
            except Exception as e:
                logger.error("RecFlow.generate_recommendation: LLM call failed - %s", str(e))
        
        if not response:
            response = f"We recommend {tier}.\n\nHere are key benefits:\n{benefits_text[:1500]}"
            logger.info("RecFlow.generate_recommendation: Used fallback response - tier=%s", tier)
        
        logger.info("RecFlow.generate_recommendation: Completed - tier=%s, response_len=%d", tier, len(response))
        return response

    @classmethod
    def handle(cls, state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        """Main entry point for simplified recommendation flow."""
        logger.info("RecFlow.handle: Starting recommendation flow - message_len=%d", len(state.message or ""))
        
        # Always check for product identification/switches from current message
        current_product = state.product or state.session.get("product")
        
        from ..tasks import identify_product_task
        from ..prompt_runner import run_direct_task
        
        logger.info("RecFlow.handle: Product identification - current_product=%s", current_product)
        
        # Always run product identification to handle switches and corrections
        prod_result = run_direct_task(
            agent_obj=identify_product_task.agent,
            agent_key="product_identifier",
            task_key="identify_product",
            context_text=f"Message: {state.message}\nSession product: {current_product}",
            logger=logger,
            label="product_identifier.identify_product.rec_flow",
        ) or {}
        
        identified_product = prod_result.get("product")
        logger.info("RecFlow.handle: Product identification API output - product=%s, confidence=%s, has_question=%s, keys=%s", 
                   identified_product, prod_result.get("confidence"), bool(prod_result.get("question")), list(prod_result.keys()))
        logger.info("RecFlow.handle: Product identification completed - identified=%s, confidence=%s", 
                   identified_product, prod_result.get("confidence"))
        
        # Handle product switch or correction
        if identified_product and identified_product != current_product:
            logger.info("RecFlow.handle: Product switch detected - %s -> %s, clearing previous state", 
                       current_product, identified_product)
            # Clear previous product's data
            state.session.pop("slots", None)
            state.session.pop("recommendation_status", None)
            # Set new product
            product = identified_product
            state.product = product
            state.session["product"] = product
        elif identified_product:
            # Same product confirmed
            product = identified_product
            state.product = product
            state.session["product"] = product
        elif current_product:
            # No new product identified, use current
            product = current_product
        else:
            # No product identified at all - set status to in_progress so we continue in RecFlow
            question = "What type of insurance are you interested in for the recommendation: Travel or Maid?"
            if prod_result and "question" in prod_result:
                question = prod_result["question"]
            state.reply = question
            state.session["recommendation_status"] = "in_progress"
            logger.info("RecFlow.handle: No product identified, requesting clarification and setting status to in_progress")
            return "__done__"
        
        # Check if recommendation is already complete for this product
        recommendation_status = state.session.get("recommendation_status")
        logger.info("RecFlow.handle: Current recommendation status=%s", recommendation_status)
        
        if recommendation_status == "done":
            # Check if user wants a new recommendation or is asking for something else
            message_lower = state.message.lower()
            restart_keywords = ["new recommendation", "fresh recommendation", "start over", "restart", "again", "different recommendation"]
            wants_new_rec = any(keyword in message_lower for keyword in restart_keywords)

            if wants_new_rec or "recommendation" in message_lower:
                logger.info("RecFlow.handle: User wants new recommendation, clearing previous state")
                # Clear recommendation status to allow new recommendation
                state.session.pop("recommendation_status", None)
                state.session.pop("slots", None)
            else:
                state.reply = "You already have a recommendation. How else can I help you?"
                logger.info("RecFlow.handle: Recommendation already complete, no new request detected")
                return "__done__"
        
        # Get current slots and required slots
        current_slots = state.session.get("slots", {}) or {}
        required_slots = cls._required_slots_for_product(product)
        
        # Check user preference for detailed explanations
        user_wants_details = state.session.get("user_wants_details", True)  # Default to True
        
        logger.info("RecFlow.handle: Slot analysis - product=%s, current_slots=%s, required_slots=%s, user_wants_details=%s", 
                   product, list(current_slots.keys()), required_slots, user_wants_details)
        
        # Set recommendation status to in_progress if we have slots to collect
        if required_slots and recommendation_status != "in_progress":
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
                    final = run_direct_task(
                        agent_obj=recommendation_responder,
                        agent_key="recommendation_responder",
                        task_key="synthesize_response",
                        context_text=f"[System]\n{sys_t}\n\n[User]\n{usr_t}",
                        logger=logger,
                        label="recommendation.response_synthesis.car",
                    ) or {}
                    
                    logger.info("RecFlow.handle: Car recommendation API output - type=%s, has_response=%s, keys=%s", 
                               type(final).__name__, bool(final.get("response") if isinstance(final, dict) else bool(final)), 
                               list(final.keys()) if isinstance(final, dict) else "N/A")
                    
                    if isinstance(final, dict) and final.get("response"):
                        car_response = final["response"]
                    elif final and str(final).strip():
                        car_response = str(final)
                    logger.info("RecFlow.handle: Car LLM response generated - length=%d", len(car_response))
                except Exception as e:
                    logger.error("RecFlow.handle: Car LLM call failed - %s", str(e))
            
            if not car_response:
                car_response = f"Here are the key benefits for Car insurance:\n\n{benefits_text[:4096]}"
                logger.info("RecFlow.handle: Using car fallback response")
                
            state.reply = car_response
                
            # Mark as complete
            state.session["recommendation_status"] = "done"
            
            # Clear comparison/summary states to avoid unintended bypass
            state.session.pop("compare_pending", None)
            state.session.pop("summary_pending", None)

            logger.info("RecFlow.handle: Car recommendation completed - response_len=%d", len(str(state.reply or "")))
            return "__done__"
        
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
        
        for slot_name in slots_to_validate:
            # Check if this slot is already validated
            if cls._is_slot_valid(updated_slots, slot_name):
                logger.info("RecFlow.handle: Slot validation skipped - %s (already valid)", slot_name)
                continue
            
            # Validate the slot
            slot_value = cls._get_slot_value(updated_slots, slot_name)
            logger.info("RecFlow.handle: Starting slot validation - %s=%s", slot_name, slot_value)
            validation_result = cls._validate_slot(slot_name, slot_value, product, state, logger)
            
            if validation_result.get("valid") and validation_result.get("normalized_value"):
                # Valid: update with normalized value and mark as validated
                cls._set_slot_value(updated_slots, slot_name, validation_result["normalized_value"], True)
                logger.info("RecFlow.handle: Slot validated successfully - %s=%s", slot_name, validation_result["normalized_value"])
            else:
                # Invalid: remove from slots and ask for clarification
                updated_slots.pop(slot_name)
                logger.info("RecFlow.handle: Slot validation failed - %s (removed from slots)", slot_name)
                validation_failed_slot = slot_name
                
                # Strictly use the bot's question for travel_duration, as requested
                if slot_name == "travel_duration":
                    validation_failed_question = validation_result.get("question")
                else:
                    validation_failed_question = validation_result.get("question") or f"I'm sorry, I didn't understand that. Could you please provide a valid value for {slot_name.replace('_', ' ')}?"
                
                logger.info("RecFlow.handle: Generated validation failure question: '%s'", validation_failed_question)
                break  # Stop processing other slots to focus on the invalid one
        
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

            # Save the question for context in next turn
            state.session["last_question"] = question
            state.reply = question
            logger.info("RecFlow.handle: Asked question for missing slot - slot=%s, question_len=%d", next_slot, len(question))
            return "__done__"
        else:
            # All slots filled - generate recommendation
            logger.info("RecFlow.handle: All slots filled, generating recommendation - slot_count=%d", len(updated_slots))
            recommendation = cls._generate_recommendation(product, updated_slots, state, logger)
            
            # Mark recommendation as complete
            state.session["recommendation_status"] = "done"

            # Clear comparison/summary states to avoid unintended bypass
            state.session.pop("compare_pending", None)
            state.session.pop("summary_pending", None)

            state.reply = recommendation
            logger.info("RecFlow.handle: Recommendation flow completed - status=done, response_len=%d", len(recommendation))
            return "__done__"