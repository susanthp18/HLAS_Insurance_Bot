from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from crewai.flow.flow import Flow, start, listen, router
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
import yaml
from .prompt_runner import run_direct_task
from .config_loader import get_agents_spec, get_tasks_spec
from zoneinfo import ZoneInfo  # Python 3.9+
from .tasks import (
    identify_product_task,
    route_decision_task,
    construct_follow_up_query_task,
    answer_capabilities_task,
)
from .tools.benefits_tool import benefits_tool
from .agents import recommendation_responder
from json import loads as json_loads
from json import dumps as json_dumps
import re
from .flows.info_flow import InfoFlowHelper
from .flows.compare_flow import CompareFlowHelper
from .flows.summary_flow import SummaryFlowHelper
from .flows.purchase_flow import PurchaseFlowHelper
from .utils.greeting import get_time_based_greeting
from .lc.history import get_last_n_pairs
from .utils.product_lex import lexical_product_hint, lexical_product_candidates

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
        
        # Use cached agent/task specs from config_loader
        self._agents_spec = get_agents_spec()
        self._tasks_spec = get_tasks_spec()
        logger.info("HlasFlow.__init__: Using cached config - agents=%d, tasks=%d", 
                   len(self._agents_spec), len(self._tasks_spec))

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
        # Live-agent short-circuit: if a human handover is active, skip orchestration entirely
        try:
            las = self.state.session.get("live_agent_status")
            is_live = False
            if isinstance(las, str):
                is_live = las.strip().lower() in ("on", "true", "yes", "1")
            else:
                is_live = bool(las)
            if is_live:
                self.state.reply = "You're now connected to a live agent. I'll pause automated replies until you're done."
                logger.info("HlasFlow.decide: live_agent_status active - short-circuiting flow.")
                return "__done__"
        except Exception:
            # If anything goes wrong in detection, continue with normal routing (fail-open)
            pass

        # Debug session state at entry
        recommendation_status = self.state.session.get("recommendation_status")
        comparison_status = self.state.session.get("comparison_status")
        summary_status = self.state.session.get("summary_status")
        session_product = self.state.session.get("product")
        
        logger.info("HlasFlow.decide: Entry state - message='%s', rec_status='%s', cmp_status='%s', sum_status='%s', product='%s'", 
                   self.state.message[:100], recommendation_status, comparison_status, summary_status, session_product)
        
        # If purchase flow is mid-stage (product clarification or confirmation),
        # give it precedence before any other multi-turn flow.
        purchase_stage = self.state.session.get("purchase_flow_stage")
        if purchase_stage:
            logger.info("HlasFlow.decide: Purchase flow stage active (%s); handling before other flows", purchase_stage)
            purchase_result = PurchaseFlowHelper.handle_stage(self.state, self._logger)
            if purchase_result:
                return purchase_result
        
        # Check recommendation status for simplified flow control
        if recommendation_status == "in_progress":
            logger.info("HlasFlow.decide: Recommendation in progress, bypassing orchestrator to RecFlow")
            if RECFLOW_AVAILABLE and RecFlowHelper:
                return RecFlowHelper.handle(self.state, {"directive": "continue_recommendation"}, self._logger)
            else:
                logger.error("HlasFlow.decide: RecFlow not available but recommendation_status='in_progress'")
                self.state.session.pop("recommendation_status", None)
        
        # Fraud guided intro in progress should bypass orchestrator/constructor as well
        try:
            if (self.state.session.get("fraud_stage") or "").strip():
                logger.info("HlasFlow.decide: Fraud guided intro stage active (%s), bypassing orchestrator to RecFlow", self.state.session.get("fraud_stage"))
                if RECFLOW_AVAILABLE and RecFlowHelper:
                    return RecFlowHelper.handle(self.state, {"directive": "continue_recommendation"}, self._logger)
        except Exception:
            pass

        if comparison_status == "in_progress":
            logger.info("HlasFlow.decide: Comparison in progress, bypassing orchestrator to CompareFlow")
            return CompareFlowHelper.handle(self.state, {}, self._logger)

        if summary_status == "in_progress":
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

        # Create compact, explicit context for orchestrator
        current_user_message = self.state.message
        product_in_session = self.state.session.get("product") or None

        # Early lexical product switch detection with double confirmation
        # Bypass if a product clarification is outstanding in any flow
        clarifying_product = bool(
            (self.state.session.get("_last_info_prod_q") or False)
            or (self.state.session.get("_last_rec_prod_q") or False)
        )
        try:
            cand_list = [] if clarifying_product else lexical_product_candidates(current_user_message)
        except Exception:
            cand_list = []
        # Fully LLM-driven: probe identifier whenever we have any candidates
        should_probe = bool(cand_list)
        session_or_state_prod = self.state.product or product_in_session
        if should_probe:
            try:
                # Double-confirm via product identifier; instruct conservatively in context
                lines = [
                    f"Message: {current_user_message}",
                    f"Session product: {session_or_state_prod}",
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
                prod_probe = run_direct_task(
                    agent_obj=identify_product_task.agent,
                    agent_key="product_identifier",
                    task_key="identify_product",
                    context_text=probe_ctx,
                    logger=self._logger,
                    label="product_identifier.double_confirm.on_decide",
                ) or {}
                confirmed = (prod_probe.get("product") or "").strip()
                conf = float(prod_probe.get("confidence") or 0.0)
                if confirmed and conf >= 0.8:
                    self.state.product = confirmed
                    self.state.session["product"] = confirmed
                    # Clear per-product recommendation state to prevent cross-product leakage
                    self.state.session.pop("slots", None)
                    self.state.session.pop("recommendation_status", None)
                    self.state.session.pop("_last_slot_name", None)
                    self.state.session.pop("_last_slot_question", None)
                    self.state.session.pop("recommended_tier", None)
                    self.state.session.pop("last_question", None)
                    # Mark switch for downstream handlers
                    self.state.session["__product_switch_confirmed__"] = True
                    logger.info("HlasFlow.decide: Product switch confirmed via lexical+LLM - %s (state cleared)", confirmed)
            except Exception as e:
                logger.warning("HlasFlow.decide: Double-confirmation failed - %s", str(e))

        # Refresh session product after any potential switch before routing
        product_in_session = self.state.session.get("product") or self.state.product or None

        history: list = self.state.session.get("history", []) or []
        history_len = len(history)

        # Pending when a bot question or info-flow clarification is outstanding
        pending_flag = bool(self.state.session.get("last_question") or self.state.session.get("_last_info_prod_q"))

        # Turn classification: treat pending_flag or last_completed as follow-up candidate
        last_completed = self.state.session.get("last_completed")
        
        if pending_flag:
            turn = "follow_up_candidate"
        elif last_completed in ("recommendation", "comparison", "summary"):
            # After a long bot response, short follow-ups are common; let the orchestrator decide
            turn = "follow_up_candidate"
        elif history_len == 0:
            turn = "first"
        else:
            turn = "normal"

        # Compact recent context (most recent first), at most 2 pairs, User → Assistant with trimming
        recent_pairs = get_last_n_pairs(self.state.session, n=2)

        def _shorten_text(val: str, max_len: int = 200) -> str:
            s = (val or "").strip()
            if len(s) <= max_len:
                return s
            return s[:max_len].rstrip() + "…"

        recent_context_lines: list[str] = []
        for idx, p in enumerate(recent_pairs, start=1):
            try:
                u = p.get("user") or ""
                a = p.get("assistant") or ""
                turn_lines: list[str] = []
                if u:
                    turn_lines.append(f"User: {_shorten_text(u)}")
                if a:
                    turn_lines.append(f"Assistant: {_shorten_text(a)}")
                if turn_lines:
                    recent_context_lines.append(f"Turn -{idx}:")
                    recent_context_lines.extend(turn_lines)
            except Exception:
                continue

        # Build explicit, non-JSON context to avoid duplicating fields
        try:
            last_completed_val = self.state.session.get("last_completed") or "none"
            lines: list[str] = []
            lines.append(f"CURRENT_MESSAGE: {current_user_message}")
            lines.append(f"SESSION_PRODUCT: {product_in_session or self.state.product or ''}")
            lines.append(f"TURN: {turn}")
            lines.append(f"LAST_COMPLETED: {last_completed_val}")
            lines.append(f"PENDING_FLAG: {str(pending_flag)}")
            if recent_context_lines:
                lines.append("RECENT_CONTEXT (most recent first):")
                lines.extend(recent_context_lines)
            orchestrator_ctx = "\n".join(lines)
        except Exception:
            orchestrator_ctx = (
                f"CURRENT_MESSAGE: {current_user_message}\n"
                f"SESSION_PRODUCT: {product_in_session or self.state.product or ''}"
            )

        logger.info("HlasFlow.decide: Calling orchestrator - context_len=%d", len(orchestrator_ctx))
        
        d = run_direct_task(
            agent_obj=route_decision_task.agent,
            agent_key="orchestrator",
            task_key="route_decision",
            context_text=orchestrator_ctx,
            logger=self._logger,
            label="orchestrator.route_decision",
        ) or {"directive": "handle_capabilities"}

        # Log the orchestrator's raw output for debugging/traceability
        directive = d.get("directive", "handle_capabilities")
        logger.info("HlasFlow.decide: Orchestrator output - directive=%s, keys=%s", directive, list(d.keys()))

        # Handle live agent intent immediately
        if directive == "live_agent":
            try:
                # Set session flag for downstream handlers
                self.state.session["live_agent_status"] = True
            except Exception:
                pass
            self.state.reply = "Please wait a minute i will connect you to live agent."
            logger.info("HlasFlow.decide: live_agent intent detected - session flag set and reply prepared")
            return "__done__"

        if directive == "handle_purchase":
            logger.info("HlasFlow.decide: Routing to PurchaseFlow helper via orchestrator directive")
            return PurchaseFlowHelper.handle(self.state, d, self._logger)

        # --- Safety guard (relaxed): only override when truly first turn with no pending flag ---
        try:
            hist_len = len(self.state.session.get("history", []) or [])
        except Exception:
            hist_len = 0
        if directive == "handle_follow_up":
            if hist_len == 0 and not (pending_flag or turn == "follow_up_candidate"):
                logger.warning("HlasFlow.decide: 'handle_follow_up' but no prior context and no pending flag; overriding to 'handle_information'.")
                directive = "handle_information"
        # --- End guard ---

        # --- Observability for follow-up ---
        if directive == "handle_follow_up":
            logger.info("HlasFlow.decide: follow_up selected by orchestrator (non-first turn). Proceeding as follow-up.")
        # --- End observability ---

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
            # Standardized greeting
            self.state.reply = (
                "Hello! 👋 I’m the HLAS Smart Bot. I’m here to guide you through our insurance products and services, "
                "answer your questions instantly, and make things easier for you. How can I help you today?"
            )
            logger.info("HlasFlow.decide: Greeting generated (standardized)")
            return "__done__"

        if directive == "handle_capabilities":
            # Answer capability/meta questions using the static knowledge base
            try:
                kb_path = Path(__file__).parent / "config" / "knowledge_base.txt"
                kb_text = kb_path.read_text(encoding="utf-8")
            except Exception:
                kb_text = ""

            ctx_lines: list[str] = []
            ctx_lines.append(f"Question: {self.state.message}")
            if kb_text:
                ctx_lines.append("")
                ctx_lines.append("Knowledge Base:")
                ctx_lines.append(kb_text)
            context_text = "\n".join(ctx_lines)

            try:
                out = run_direct_task(
                    agent_obj=answer_capabilities_task.agent,
                    agent_key="capabilities_responder",
                    task_key="answer_capabilities",
                    context_text=context_text,
                    logger=self._logger,
                    label="capabilities.answer",
                ) or {}
                reply = out.get("response") or "I can help with product information, summaries, comparisons, and recommendations. Ask me what you’d like to know."
            except Exception:
                reply = "I can help with product information, summaries, comparisons, and recommendations. Ask me what you’d like to know."

            self.state.reply = reply
            logger.info("HlasFlow.decide: Capabilities response generated via KB - len=%d", len(self.state.reply))
            return "__done__"

        if directive == "handle_information":
            logger.info("HlasFlow.decide: Routing to InfoFlow")
            # Clear any leftover question flag when a fresh info request is detected
            try:
                if "last_question" in self.state.session and self.state.message:
                    self.state.session.pop("last_question", None)
            except Exception:
                pass
            return InfoFlowHelper.handle(self.state, {}, self._logger)

        if directive == "handle_follow_up":
            # Check if this is a follow-up to a product clarification question
            clarification_follow_up = self.state.session.get("_last_info_prod_q")
            if clarification_follow_up:
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
                    self._logger.info(
                        "HlasFlow.decide: Clarified product='%s' for stored question; delegating to follow-up classifier.",
                        clarified_product,
                    )
                else:
                    self._logger.warning(
                        "HlasFlow.decide: Product clarification follow-up lacked original question; continuing with current message.")
                    self.state.session.pop("_last_info_prod_q", None)
                    self.state.session.pop("_last_info_user_msg", None)
                    self.state.session.pop("last_question", None)

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
                # Clear per-product recommendation state to prevent cross-product leakage
                self.state.session.pop("slots", None)
                self.state.session.pop("recommendation_status", None)
                self.state.session.pop("_last_slot_name", None)
                self.state.session.pop("_last_slot_question", None)
                self.state.session.pop("recommended_tier", None)
                self.state.session.pop("pending_slot", None)
                self.state.last_question = None

                # Keep last 2 pairs (most recent first)
                use_history_pairs = get_last_n_pairs(self.state.session, n=2)
                logger.debug("HlasFlow.decide: Follow-up using %d history pairs (product switch)", len(use_history_pairs))
            else:
                # Prepare recent history window (most recent first)
                use_history_pairs = get_last_n_pairs(self.state.session, n=2)
                logger.debug("HlasFlow.decide: Follow-up using %d history pairs (no product switch)", len(use_history_pairs))

            # Build recent context with turn markers and trimming
            def _shorten_text_local(val: str, max_len: int = 200) -> str:
                s = (val or "").strip()
                if len(s) <= max_len:
                    return s
                return s[:max_len].rstrip() + "…"

            recent_lines: list[str] = []
            for idx, pair in enumerate(use_history_pairs, start=1):
                try:
                    u = pair.get("user", "")
                    a = pair.get("assistant", "")
                    turn_parts: list[str] = []
                    if u:
                        turn_parts.append(f"User: {_shorten_text_local(u)}")
                    if a:
                        turn_parts.append(f"Assistant: {_shorten_text_local(a)}")
                    if turn_parts:
                        recent_lines.append(f"Turn -{idx}:")
                        recent_lines.extend(turn_parts)
                except Exception:
                    continue

            # Include available tiers dynamically (avoid hardcoding in prompts)
            available_tiers_line = ""
            try:
                prod_for_tiers = self.state.session.get('product') or self.state.product or ""
                canon_tiers = CompareFlowHelper._canonical_tiers(prod_for_tiers) if prod_for_tiers else []
                if canon_tiers:
                    available_tiers_line = f"Available tiers: {', '.join(canon_tiers)}\n"
            except Exception:
                available_tiers_line = ""

            # Build focused context for follow-up constructor
            prod_line = f"Product: {self.state.session.get('product') or ''}"
            tiers_line = (available_tiers_line or "").rstrip("\n")
            ctx_lines: list[str] = [prod_line]
            if tiers_line:
                ctx_lines.append(tiers_line)
            ctx_lines.append(f"CURRENT_MESSAGE: {self.state.message}")
            if recent_lines:
                ctx_lines.append("RECENT_CONTEXT (most recent first):")
                ctx_lines.extend(recent_lines)
            # Lightweight guidance to anchor evidence and reduce drift
            ctx_lines.append("Guidance:")
            ctx_lines.append("- Prefer CURRENT_MESSAGE. Use RECENT_CONTEXT only to resolve pronouns/vague references in CURRENT_MESSAGE.")
            ctx_lines.append("- Do not introduce topics from older turns unless CURRENT_MESSAGE refers to them.")
            ctx_lines.append("- Evidence must cite tokens in CURRENT_MESSAGE, or a pronoun in CURRENT_MESSAGE plus its antecedent in Turn -1.")
            fu_context = "\n".join(ctx_lines)

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
            constructor_query_type = (follow_up.get("query_type") or "").strip().lower() or "handle_information"
            constructor_confidence = follow_up.get("routing_confidence", 0.0)
            constructor_evidence = (follow_up.get("evidence") or "").strip()

            logger.info(
                "HlasFlow.decide: Constructor output - query_len=%d, type=%s, confidence=%.2f, evidence='%s'",
                len(query),
                constructor_query_type,
                constructor_confidence,
                constructor_evidence[:100] if constructor_evidence else "(none)",
            )

            # --- ROUTING COORDINATION: Compare constructor vs orchestrator ---
            # The orchestrator already decided handle_follow_up, but the constructor provides
            # a more informed routing suggestion based on query analysis.
            
            # Map constructor query_type to orchestrator directives
            constructor_directive = constructor_query_type
            
            # Prefer constructor if:
            # 1. Confidence >= 0.7 AND
            # 2. Evidence is present (indicating grounded decision)
            prefer_constructor = (constructor_confidence >= 0.7 and len(constructor_evidence) > 0)
            
            # Check for disagreement: constructor suggests info/summary but orchestrator chose follow_up
            # (which can lead to comparison misrouting)
            orchestrator_would_route_comparison = (directive == "handle_follow_up")  # We're in this block
            
            final_query_type = constructor_query_type
            routing_source = "constructor"
            
            if prefer_constructor:
                logger.info(
                    "HlasFlow.decide: Preferring constructor routing - confidence=%.2f >= 0.7, evidence present",
                    constructor_confidence
                )
                final_query_type = constructor_query_type
                routing_source = "constructor"
            else:
                # Low confidence or missing evidence - evaluate if query is self-contained
                # If constructor suggests info/summary but confidence is low, still prefer info as safe default
                if constructor_query_type in ("handle_information", "handle_summary"):
                    logger.info(
                        "HlasFlow.decide: Constructor suggests info/summary with low confidence (%.2f), using as safe default",
                        constructor_confidence
                    )
                    final_query_type = constructor_query_type
                    routing_source = "constructor_safe_default"
                else:
                    logger.info(
                        "HlasFlow.decide: Constructor confidence low (%.2f) and not info/summary, using constructor suggestion anyway",
                        constructor_confidence
                    )
                    final_query_type = constructor_query_type
                    routing_source = "constructor_fallback"
            
            logger.info(
                "HlasFlow.decide: Final routing decision - type=%s, source=%s, product=%s",
                final_query_type,
                routing_source,
                self.state.session.get("product"),
            )

            # Clear any lingering clarification flag
            try:
                self.state.session.pop("last_question", None)
            except Exception:
                pass

            if final_query_type == "plan_only_comparison":
                self.state.session.pop("_fu_query", None)
                self.state.message = query
                logger.info("HlasFlow.decide: Routing follow-up to CompareFlow (source=%s)", routing_source)
                return CompareFlowHelper.handle(self.state, {"from_follow_up": True}, self._logger)

            if final_query_type == "handle_summary":
                self.state.session.pop("_fu_query", None)
                self.state.message = query
                logger.info("HlasFlow.decide: Routing follow-up to SummaryFlow (source=%s)", routing_source)
                return SummaryFlowHelper.handle(self.state, {"from_follow_up": True}, self._logger)

            if final_query_type == "handle_recommendation":
                self.state.session.pop("_fu_query", None)
                self.state.message = query
                logger.info("HlasFlow.decide: Routing follow-up to RecFlow (source=%s)", routing_source)
                if RECFLOW_AVAILABLE and RecFlowHelper:
                    return RecFlowHelper.handle(self.state, {"directive": "handle_recommendation", "from_follow_up": True}, self._logger)
                logger.error("HlasFlow.decide: RecFlow not available for follow-up recommendation request.")
                self.state.reply = "I'm sorry, the recommendation service is temporarily unavailable. Please try again later."
                return "__done__"

            if final_query_type == "handle_purchase":
                self.state.session.pop("_fu_query", None)
                self.state.message = query
                logger.info("HlasFlow.decide: Routing follow-up to PurchaseFlow (source=%s)", routing_source)
                return PurchaseFlowHelper.handle(self.state, {"directive": "handle_purchase", "from_follow_up": True}, self._logger)

            # Default to information handling
            self.state.session["_fu_query"] = query
            logger.info("HlasFlow.decide: Routing follow-up to InfoFlow (type=%s, source=%s)", final_query_type, routing_source)
            return InfoFlowHelper.handle(self.state, {"use_follow_up_query": True}, self._logger)

        if directive == "handle_summary":
            logger.info("HlasFlow.decide: Routing to SummaryFlow")
            # Clear any stale slot question; summary is a new flow context
            try:
                self.state.session.pop("last_question", None)
            except Exception:
                pass
            return SummaryFlowHelper.handle(self.state, {}, self._logger)

        if directive == "plan_only_comparison":
            logger.info("HlasFlow.decide: Routing to CompareFlow")
            # Clear any stale slot question; comparison is a new flow context
            try:
                self.state.session.pop("last_question", None)
            except Exception:
                pass
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