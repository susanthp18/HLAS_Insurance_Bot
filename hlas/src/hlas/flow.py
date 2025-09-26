from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from crewai.flow.flow import Flow, start, listen, router
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
import yaml
from .prompt_runner import run_direct_task
from zoneinfo import ZoneInfo  # Python 3.9+
from .tasks import (
    identify_product_task,
    retrieve_information_task,
    summarize_product_task,
    compare_products_task,
    provide_recommendation_task,
    route_decision_task,
    questionnaire_ask_next_slot_task,
    questionnaire_capture_pending_slot_task,
    manage_travel_recommendation_flow_task,
    manage_maid_recommendation_flow_task,
    construct_follow_up_query_task,
)
from .tools.benefits_tool import benefits_tool
from .agents import recommendation_responder
from json import loads as json_loads
import re
from .flows.info_flow import InfoFlowHelper
from .flows.compare_flow import CompareFlowHelper
from .flows.summary_flow import SummaryFlowHelper
from .flows.recommendation_flow import RecommendationFlowHelper
from .utils.greeting import get_time_based_greeting

# Try to import RecFlow with error handling
try:
    from .flows.rec_flow import RecFlowHelper
    RECFLOW_AVAILABLE = True
    logger.info("Flow.__init__: RecFlow imported successfully")
except ImportError as e:
    RECFLOW_AVAILABLE = False
    RecFlowHelper = None
    logger.warning("Flow.__init__: RecFlow import failed: %s", e)


class HlasState(BaseModel):
    session: Dict[str, Any] = {}
    message: str = ""
    product: Optional[str] = None
    doc_type: Optional[str] = None
    slot_to_ask: Optional[str] = None
    question: Optional[str] = None
    last_question: Optional[str] = None
    slot_name: Optional[str] = None
    slot_value: Optional[str] = None
    reply: str = ""
    sources: str = ""


class HlasFlow(Flow[HlasState]):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        logger.info("HlasFlow.__init__: Initializing with RECFLOW_AVAILABLE=%s", RECFLOW_AVAILABLE)
        
        # Load agent/task specs for building direct LLM prompts
        self._agents_spec = {}
        self._tasks_spec = {}
        try:
            base_dir = Path(__file__).parent
            with open(base_dir / "config/agents.yaml", "r", encoding="utf-8") as _af:
                self._agents_spec = yaml.safe_load(_af) or {}
            with open(base_dir / "config/tasks.yaml", "r", encoding="utf-8") as _tf:
                self._tasks_spec = yaml.safe_load(_tf) or {}
            logger.info("HlasFlow.__init__: Config loaded - agents=%d, tasks=%d", 
                       len(self._agents_spec), len(self._tasks_spec))
        except Exception as e:
            logger.warning("HlasFlow.__init__: Config loading failed - %s", str(e))

    def _llm_json_from_agent(self, agent_obj: Any, system_prompt: str, user_prompt: str, label: str) -> Dict[str, Any]:
        logger.debug("HlasFlow._llm_json_from_agent: Starting %s - sys_len=%d, user_len=%d", 
                    label, len(system_prompt), len(user_prompt))
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            raw = agent_obj.llm.call(messages=messages)
            txt = str(raw).strip()
            logger.debug("HlasFlow._llm_json_from_agent: LLM response for %s - length=%d", label, len(txt))
            
            # Try strict JSON first
            try:
                result = json_loads(txt)
                logger.debug("HlasFlow._llm_json_from_agent: JSON parsing successful for %s", label)
                return result
            except Exception as e:
                logger.debug("HlasFlow._llm_json_from_agent: Strict JSON failed for %s - %s, trying extraction", label, str(e))
                # Best-effort JSON extraction (no external fallback)
                m = re.search(r"\{[\s\S]*\}", txt)
                if m:
                    try:
                        result = json_loads(m.group(0))
                        logger.debug("HlasFlow._llm_json_from_agent: JSON extraction successful for %s", label)
                        return result
                    except Exception as e2:
                        logger.warning("HlasFlow._llm_json_from_agent: JSON extraction failed for %s - %s", label, str(e2))
                        return {}
                logger.warning("HlasFlow._llm_json_from_agent: No JSON pattern found for %s", label)
                return {}
        except Exception as e:
            logger.error("HlasFlow._llm_json_from_agent: LLM call failed for %s - %s", label, str(e))
            return {}

    # Slot policy helpers: ordered required slots per product
    def _required_slots_for_product(self, product: Optional[str]) -> List[str]:
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

    def _first_missing_slot(self) -> Optional[str]:
        slots = self.state.session.get("slots", {}) or {}
        for s in self._required_slots_for_product(self.state.product):
            if s not in slots or slots.get(s) in (None, ""):
                return s
        return None

    @start()
    def ingest(self) -> Dict[str, Any]:
        # Inputs provided to kickoff() populate state automatically in CrewAI Flows
        # Ensure required fields exist
        self.state.message = self.state.message or ""
        self.state.session = self.state.session or {}
        
        logger.info("HlasFlow.ingest: Starting flow - message_len=%d, session_keys=%s", 
                   len(self.state.message), list(self.state.session.keys()))
        
        return {"message": self.state.message, "session": self.state.session}

    @router(ingest)
    def decide(self, payload: Dict[str, Any]) -> str:
        # Debug session state at entry
        recommendation_status = self.state.session.get("recommendation_status")
        comparison_status = self.state.session.get("comparison_status")
        summary_status = self.state.session.get("summary_status")
        session_product = self.state.session.get("product")
        
        logger.info("HlasFlow.decide: Entry state - message='%s', rec_status='%s', cmp_status='%s', sum_status='%s', product='%s'", 
                   self.state.message[:100], recommendation_status, comparison_status, summary_status, session_product)
        
        # Check recommendation status for simplified flow control
        if recommendation_status == "in_progress":
            logger.info("HlasFlow.decide: Recommendation in progress, bypassing orchestrator to RecFlow")
            if RECFLOW_AVAILABLE and RecFlowHelper:
                return RecFlowHelper.handle(self.state, {"directive": "continue_recommendation"}, self._logger)
            else:
                logger.error("HlasFlow.decide: RecFlow not available but recommendation_status='in_progress'")
                self.state.session.pop("recommendation_status", None)

        elif comparison_status == "in_progress":
            logger.info("HlasFlow.decide: Comparison in progress, bypassing orchestrator to CompareFlow")
            return CompareFlowHelper.handle(self.state, {}, self._logger)

        elif summary_status == "in_progress":
            logger.info("HlasFlow.decide: Summary in progress, bypassing orchestrator to SummaryFlow")
            return SummaryFlowHelper.handle(self.state, {}, self._logger)

        # Cleanup for completed flows
        if recommendation_status == "done":
            logger.info("HlasFlow.decide: Recommendation done, clearing status.")
            self.state.session.pop("recommendation_status", None)

        if comparison_status == "done":
            logger.info("HlasFlow.decide: Comparison done, clearing status.")
            self.state.session.pop("comparison_status", None)

        if summary_status == "done":
            logger.info("HlasFlow.decide: Summary done, clearing status.")
            self.state.session.pop("summary_status", None)

        # Fall through to orchestrator if no multi-turn flow is active
        logger.debug("HlasFlow.decide: No active multi-turn flow, proceeding to orchestrator")

        # Create structured context for the orchestrator
        last_user_message = self.state.message
        product_in_session = self.state.session.get("product") or "None"
        
        # Get recent conversation (last 3 turns, most recent first)
        history = self.state.session.get("history", [])
        recent_conversation = []
        if history:
            # Take last 1 turn and reverse to show most recent first
            recent_turns = history[-1:]
            recent_turns.reverse()
            for turn in recent_turns:
                user_msg = turn.get("user", "")
                assistant_msg = turn.get("assistant", "")
                if user_msg and assistant_msg:
                    recent_conversation.append(f"User: {user_msg}")
                    recent_conversation.append(f"Assistant: {assistant_msg}")
        
        recent_conversation_text = "\n".join(recent_conversation) if recent_conversation else "No recent conversation"
        
        context_rd = (
            f"Last_user_message: {last_user_message}\n"
            f"Product_in_session: {product_in_session}\n"
            f"Recent_conversation:\n{recent_conversation_text}"
        )

        
        logger.info("HlasFlow.decide: Calling orchestrator - context_len=%d", len(context_rd))
        
        d = run_direct_task(
            agent_obj=route_decision_task.agent,
            agent_key="orchestrator",
            task_key="route_decision",
            context_text=context_rd,
            logger=self._logger,
            label="orchestrator.route_decision",
        ) or {"directive": "handle_capabilities"}

        # Log the orchestrator's raw output for debugging/traceability
        directive = d.get("directive", "handle_capabilities")
        logger.info("HlasFlow.decide: Orchestrator output - directive=%s, keys=%s", directive, list(d.keys()))

        # --- Smart State Cleanup ---
        # If the orchestrator decides this is NOT a follow-up, it's a new topic.
        # Safely clear any temporary flags from a previous, completed info_flow clarification.
        if directive != "handle_follow_up":
            if "_last_info_prod_q" in self.state.session:
                self.state.session.pop("_last_info_prod_q", None)
                self.state.session.pop("_last_info_user_msg", None)
                self.state.session.pop("last_question", None)
                self._logger.info("HlasFlow.decide: Directive is not a follow-up. Clearing stale info-flow flags.")
        # --- End of Smart Cleanup ---

        if directive == "greet":
            # Time-aware greeting (Singapore)
            salutation = "Hello"
            try:
                now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
                hour = now_sg.hour
                if hour < 12:
                    salutation = "Good morning"
                elif hour < 18:
                    salutation = "Good afternoon"
                else:
                    salutation = "Good evening"
                logger.debug("HlasFlow.decide: Generated time-aware greeting - hour=%d, salutation=%s", hour, salutation)
            except Exception as e:
                logger.warning("HlasFlow.decide: Time-aware greeting failed - %s, using default", str(e))
                
            self.state.reply = f"{salutation}! I'm HLAS Assistant. I can help you with insurance plans, questions, comparisons, and summaries."
            logger.info("HlasFlow.decide: Greeting generated")
            return "__done__"

        if directive == "handle_capabilities":
            self.state.reply = "I can help you with insurance plans, providing information, summaries, and comparisons."
            logger.info("HlasFlow.decide: Capabilities response generated")
            return "__done__"

        if directive == "handle_information":
            logger.info("HlasFlow.decide: Routing to InfoFlow")
            return InfoFlowHelper.handle(self.state, {}, self._logger)

        if directive == "handle_follow_up":
            # Check if this is a follow-up to a product clarification question
            if self.state.session.get("_last_info_prod_q"):
                self._logger.info("HlasFlow.decide: Handling product clarification follow-up.")
                
                # The user's message is the product
                product_clarification = self.state.message
                
                # The original question is in the session
                original_question = self.state.session.get("_last_info_user_msg")

                if original_question:
                    # Re-route to InfoFlow, but with the original question and the now-clarified product
                    self.state.message = original_question
                    
                    # Run the clarification through the identifier to normalize it (e.g., handle typos)
                    prod_probe = run_direct_task(
                        agent_obj=identify_product_task.agent,
                        agent_key="product_identifier",
                        task_key="identify_product",
                        context_text=f"Message: {product_clarification}",
                        logger=self._logger,
                        label="product_identifier.identify_product.on_clarification_follow_up",
                    ) or {}
                    
                    clarified_product = prod_probe.get("product") or product_clarification
                    self.state.product = clarified_product
                    self.state.session["product"] = clarified_product
                    
                    # Clear the flags now that they've been used
                    self.state.session.pop("_last_info_prod_q", None)
                    self.state.session.pop("_last_info_user_msg", None)
                    self.state.session.pop("last_question", None)
                    
                    self._logger.info("HlasFlow.decide: Re-routing to InfoFlow with original question and clarified product ('%s').", clarified_product)
                    return InfoFlowHelper.handle(self.state, {}, self._logger)

            self._logger.info("HlasFlow.decide: Handling generic follow-up query (pronoun resolution, etc.).")
            
            # Detect product (and switch) before constructing follow-up query to avoid leakage
            current_product = self.state.session.get("product") or self.state.product
            identified = None
            try:
                prod = run_direct_task(
                    agent_obj=identify_product_task.agent,
                    agent_key="product_identifier",
                    task_key="identify_product",
                    context_text=f"Message: {self.state.message}\nSession product: {current_product}",
                    logger=self._logger,
                    label="product_identifier.identify_product.on_follow_up",
                ) or {}
                identified = prod.get("product") or None
                
                logger.info("HlasFlow.decide: Follow-up product identification - current=%s, identified=%s, confidence=%s",
                           current_product, identified, prod.get("confidence"))
            except Exception as e:
                logger.error("HlasFlow.decide: Follow-up product identification failed - %s", str(e))

            # Handle product switch: update product and avoid reusing prior context
            history: list = self.state.session.get("history", []) or []
            use_history_pairs = []
            
            if identified and identified != current_product:
                logger.info("HlasFlow.decide: Follow-up product switch detected (%s -> %s), clearing prior context", 
                           current_product, identified)
                
                self.state.product = identified
                self.state.session["product"] = identified
                # Clear any pending recommendation state to prevent cross-product leakage
                self.state.session.pop("pending_slot", None)
                self.state.last_question = None

                # Keep only the immediately previous turn (most recent first)
                use_history_pairs = list(reversed(history[-1:]))
                logger.debug("HlasFlow.decide: Follow-up using %d history pairs (no product switch)", len(use_history_pairs))
            else:
                # Prepare recent history window (most recent first)
                use_history_pairs = list(reversed(history[-1:]))
                logger.debug("HlasFlow.decide: Follow-up using %d history pairs (no product switch)", len(use_history_pairs))

            context_lines = []
            for pair in use_history_pairs:
                try:
                    u = pair.get("user", "")
                    a = pair.get("assistant", "")
                    context_lines.append(f"User: {u}")
                    context_lines.append(f"Assistant: {a}")
                except Exception:
                    continue
            convo_context = "\n".join(context_lines)

            fu_context = (
                f"Product: {self.state.session.get('product') or ''}\n"
                f"Latest: {self.state.message}\n"
                f"Recent conversation (most recent first):\n{convo_context}"
            )

            logger.info("HlasFlow.decide: Constructing follow-up query - context_len=%d", len(fu_context))

            follow_up = run_direct_task(
                agent_obj=construct_follow_up_query_task.agent,
                agent_key="follow_up_agent",
                task_key="construct_follow_up_query",
                context_text=fu_context,
                logger=self._logger,
                label="follow_up.construct_query",
            ) or {}

            logger.info("HlasFlow.decide: Follow-up query construction - has_query=%s, keys=%s", 
                       bool(follow_up.get("query")), list(follow_up.keys()))

            query = (follow_up.get("query") or self.state.message).strip()
            # Save constructed query into state for InfoFlow to use
            self.state.session["_fu_query"] = query
            
            logger.info("HlasFlow.decide: Follow-up constructed query - length=%d, product=%s, history_pairs=%d",
                       len(query), self.state.session.get("product"), len(use_history_pairs))
            
            # Delegate to InfoFlow for retrieval/synthesis
            return InfoFlowHelper.handle(self.state, {"use_follow_up_query": True}, self._logger)

        if directive == "handle_summary":
            logger.info("HlasFlow.decide: Routing to SummaryFlow")
            return SummaryFlowHelper.handle(self.state, {}, self._logger)

        if directive == "plan_only_comparison":
            logger.info("HlasFlow.decide: Routing to CompareFlow")
            return CompareFlowHelper.handle(self.state, {}, self._logger)

        if directive == "handle_recommendation":
            logger.info("HlasFlow.decide: Routing to recommendation flow")
            if RECFLOW_AVAILABLE and RecFlowHelper:
                logger.info("HlasFlow.decide: Using RecFlow for recommendation")
                return RecFlowHelper.handle(self.state, {"directive": "handle_recommendation"}, self._logger)
            else:
                logger.error("HlasFlow.decide: RecFlow not available for recommendation")
                self.state.reply = "I'm sorry, the recommendation service is temporarily unavailable. Please try again later."
                return "__done__"

        if directive == "handle_other":
            logger.info("HlasFlow.decide: Handling unrecognized request")
            self.state.reply = (
                "I can't understand this. Can you clearly tell what you want to do?\n"
                "I can help you with insurance plans, questions, comparisons, and summaries."
            )
            return "__done__"

        # Default fallback
        logger.info("HlasFlow.decide: Using default fallback response - directive=%s", directive)
        self.state.reply = self.state.reply or "How can I help you further?"
        return "__done__"

    # No listener for "__done__" on purpose: returning this label from the router terminates the flow.