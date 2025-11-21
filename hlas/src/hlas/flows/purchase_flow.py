from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

from ..prompt_runner import run_direct_task
from ..tasks import (
    identify_product_task,
    classify_confirmation_task,
)
from ..config_loader import get_purchase_links
from .rec_flow import RecFlowHelper
from .info_flow import InfoFlowHelper
from .compare_flow import CompareFlowHelper
from .summary_flow import SummaryFlowHelper

PURCHASE_PROMPT = "Would you like to purchase this plan?"
PRODUCT_FRIENDLY_NAMES = {
    "travel": "Travel",
    "maid": "Maid",
    "car": "Car",
    "personalaccident": "Personal Accident",
    "home": "Home",
    "early": "Early Critical Illness",
    "fraud": "Fraud Protect360",
    "hospital": "Hospital Protect360",
}


class PurchaseFlowHelper:
    """Handles purchase intent routing and confirmation."""

    @classmethod
    def handle(cls, state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        product = cls._ensure_product_context(state, logger)
        if product is None:
            # Question asked; wait for user reply
            return "__done__"

        # If we have already provided a recommendation for this specific product in the past,
        # skip the "do you need a recommendation" check and go straight to purchase.
        last_rec_prod = str((state.session or {}).get("last_recommendation_product") or "").strip().lower()
        current_prod = str(product or "").strip().lower()
        if last_rec_prod and last_rec_prod == current_prod:
            logger.info(
                "PurchaseFlow: Existing recommendation found for product=%s; skipping rec-check and sending purchase link",
                product,
            )
            state.session.pop("purchase_flow_stage", None)
            state.session.pop("last_question", None)
            cls._send_purchase_link(state, logger)
            return "__done__"

        cls._prompt_recommendation_check(state, product, logger)
        return "__done__"

    @classmethod
    def handle_stage(cls, state: Any, logger: logging.Logger) -> Optional[str]:
        stage = (state.session or {}).get("purchase_flow_stage")
        if not stage:
            return None
        if stage == "await_product_for_purchase":
            handled = cls._handle_product_clarification(state, logger)
            return "__done__" if handled else None
        if stage == "await_rec_check":
            return cls._handle_rec_check_confirmation(state, logger)
        if stage == "await_purchase_decision":
            return cls._handle_purchase_confirmation(state, logger)
        # Unknown stage; clear and fall through
        state.session.pop("purchase_flow_stage", None)
        return None

    # ------------------------------------------------------------------
    # Stage handlers
    # ------------------------------------------------------------------

    @classmethod
    def _handle_product_clarification(cls, state: Any, logger: logging.Logger) -> bool:
        product, clarification_question = cls._identify_product_from_message(state, logger)
        if product:
            state.product = product
            state.session["product"] = product
            state.session.pop("purchase_flow_stage", None)
            cls._prompt_recommendation_check(state, product, logger)
            return True
        question = clarification_question or cls._default_product_question()
        state.session["purchase_flow_stage"] = "await_product_for_purchase"
        state.session["last_question"] = question
        state.reply = question
        return True

    @classmethod
    def _handle_rec_check_confirmation(cls, state: Any, logger: logging.Logger) -> Optional[str]:
        classification = cls._classify_confirmation(state, logger)
        if classification is None:
            cls._set_fallback(state)
            return None
        label, confidence = classification
        if label == "yes":
            logger.info("PurchaseFlow: User confirmed they need a recommendation (confidence=%.2f)", confidence)
            state.session.pop("purchase_flow_stage", None)
            state.session.pop("last_question", None)
            state.session["recommendation_status"] = "in_progress"
            state.session["_skip_extraction_once"] = True
            # Hand off to RecFlow immediately
            return RecFlowHelper.handle(state, {"directive": "handle_recommendation"}, logger)
        if label == "no":
            logger.info("PurchaseFlow: User declined recommendation (confidence=%.2f)", confidence)
            state.session.pop("purchase_flow_stage", None)
            state.session.pop("last_question", None)
            cls._send_purchase_link(state, logger)
            return "__done__"
        logger.info("PurchaseFlow: Confirmation classified as alternate intent '%s' (confidence=%.2f)", label, confidence)
        return cls._route_alternate_intent(state, label, logger)

    @classmethod
    def _handle_purchase_confirmation(cls, state: Any, logger: logging.Logger) -> Optional[str]:
        classification = cls._classify_confirmation(state, logger)
        if classification is None:
            cls._set_fallback(state)
            return None
        label, confidence = classification
        if label == "yes":
            logger.info("PurchaseFlow: User confirmed purchase (confidence=%.2f)", confidence)
            state.session.pop("purchase_flow_stage", None)
            state.session.pop("last_question", None)
            cls._send_purchase_link(state, logger)
            return "__done__"
        if label == "no":
            logger.info("PurchaseFlow: User declined purchase (confidence=%.2f)", confidence)
            state.session.pop("purchase_flow_stage", None)
            state.session.pop("last_question", None)
            state.reply = "Okay, let me know if you need anything else."
            return "__done__"
        logger.info("PurchaseFlow: Purchase confirmation classified as alternate intent '%s' (confidence=%.2f)", label, confidence)
        return cls._route_alternate_intent(state, label, logger)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_product_context(cls, state: Any, logger: logging.Logger) -> Optional[str]:
        product = state.product or (state.session or {}).get("product")
        if product:
            return product
        product, clarification_question = cls._identify_product_from_message(state, logger)
        if product:
            state.product = product
            state.session["product"] = product
            return product
        question = clarification_question or cls._default_product_question()
        state.session["purchase_flow_stage"] = "await_product_for_purchase"
        state.session["last_question"] = question
        state.reply = question
        return None

    @classmethod
    def _prompt_recommendation_check(cls, state: Any, product: str, logger: logging.Logger) -> None:
        friendly = cls._friendly_product_name(product)
        question = f"Do you need a recommendation for {friendly} insurance?"
        state.session["purchase_flow_stage"] = "await_rec_check"
        state.session["last_question"] = question
        state.reply = question
        logger.info("PurchaseFlow: Prompting recommendation check for product=%s", product)

    @classmethod
    def _identify_product_from_message(
        cls, state: Any, logger: logging.Logger
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            context = f"Message: {state.message}\nSession product: {state.session.get('product')}"
            result = run_direct_task(
                agent_obj=identify_product_task.agent,
                agent_key="product_identifier",
                task_key="identify_product",
                context_text=context,
                logger=logger,
                label="purchase_flow.identify_product",
            ) or {}
        except Exception as exc:
            logger.warning("PurchaseFlow: identify_product failed - %s", exc)
            return None, None
        product = (result.get("product") or "").strip()
        question = (result.get("question") or "").strip() or None
        if product:
            logger.info("PurchaseFlow: Product identified as %s (confidence=%s)", product, result.get("confidence"))
            return product, None
        if question:
            logger.info("PurchaseFlow: Product clarification requested: %s", question)
        return None, question

    @classmethod
    def _classify_confirmation(cls, state: Any, logger: logging.Logger) -> Optional[Tuple[str, float]]:
        question = (state.session or {}).get("last_question") or ""
        if not question:
            logger.warning("PurchaseFlow: Missing last_question for confirmation classification")
            return None
        context = (
            "Question: {question}\n"
            "User reply: {reply}\n"
            "Classify whether the user agreed (yes), declined (no), or instead requested information, comparison, summary, or a new recommendation.\n"
            "Use 'plan_only_comparison' ONLY when the reply uses explicit comparison verbs such as 'compare', 'difference between', or 'which plan is better' together with plan/tier wording.\n"
            "For generic questions like 'what are the other plans available?' or 'what other options are there?' WITHOUT comparison verbs, prefer 'handle_information' (or 'handle_recommendation' if the user explicitly asks you to recommend another plan)."
        ).format(question=question.strip(), reply=(state.message or "").strip())
        try:
            result = run_direct_task(
                agent_obj=classify_confirmation_task.agent,
                agent_key="confirmation_classifier",
                task_key="classify_confirmation",
                context_text=context,
                logger=logger,
                label="purchase_flow.classify_confirmation",
            ) or {}
        except Exception as exc:
            logger.error("PurchaseFlow: classify_confirmation failed - %s", exc)
            return None
        label = (result.get("classification") or "").strip().lower()
        confidence = float(result.get("confidence") or 0.0)
        allowed = {"yes", "no", "handle_information", "plan_only_comparison", "handle_summary", "handle_recommendation"}
        if label not in allowed:
            return None
        return label, confidence

    @classmethod
    def _route_alternate_intent(cls, state: Any, intent: str, logger: logging.Logger) -> Optional[str]:
        """Route to other flows when user pivots instead of confirming."""
        state.session.pop("purchase_flow_stage", None)
        state.session.pop("last_question", None)
        intent = intent or ""
        if intent == "handle_information":
            logger.info("PurchaseFlow: Redirecting user to InfoFlow after purchase pivot")
            return InfoFlowHelper.handle(state, {}, logger)
        if intent == "plan_only_comparison":
            logger.info("PurchaseFlow: Redirecting user to CompareFlow after purchase pivot")
            return CompareFlowHelper.handle(state, {"from_purchase": True}, logger)
        if intent == "handle_summary":
            logger.info("PurchaseFlow: Redirecting user to SummaryFlow after purchase pivot")
            return SummaryFlowHelper.handle(state, {"from_purchase": True}, logger)
        if intent == "handle_recommendation":
            logger.info(
                "PurchaseFlow: User requested a new recommendation during purchase stage; restarting recommendation flow"
            )
            # Treat this as a fresh recommendation request instead of reusing the previous product/slots.
            # This avoids generating another recommendation for the old product when the user mentions
            # a different or unknown product name (e.g., 'blueberry insurance').
            try:
                state.product = None
            except Exception:
                pass
            state.session.pop("product", None)
            state.session.pop("slots", None)
            state.session.pop("recommendation_status", None)
            # Hand off to RecFlow; it will re-identify the product (or ask for clarification) from the new message.
            return RecFlowHelper.handle(state, {"directive": "handle_recommendation", "from_purchase": True}, logger)
        # Unknown intent -> fallback to orchestrator
        cls._set_fallback(state)
        return None

    @classmethod
    def _send_purchase_link(cls, state: Any, logger: logging.Logger) -> None:
        product = (state.product or state.session.get("product") or "").strip()
        normalized = cls._normalize_product_key(product)
        friendly = cls._friendly_product_name(product)
        links = get_purchase_links()
        link = links.get(normalized) if normalized else None
        if link:
            state.reply = f"You can purchase the {friendly} plan securely here: {link}"
            logger.info("PurchaseFlow: Provided purchase link for product=%s", normalized)
        else:
            state.reply = (
                f"I don't have a direct purchase link for the {friendly} plan right now. "
                "Please let me know if you'd like me to connect you with a specialist."
            )
            logger.warning("PurchaseFlow: Missing purchase link for product=%s", normalized)

    @staticmethod
    def _default_product_question() -> str:
        return (
            "Which product would you like to buy? "
            "Available options: Travel, Maid, Car, Personal Accident, Home, Fraud, Early Critical Illness, Hospital."
        )

    @staticmethod
    def _friendly_product_name(product: Optional[str]) -> str:
        key = PurchaseFlowHelper._normalize_product_key(product)
        if key and key in PRODUCT_FRIENDLY_NAMES:
            return PRODUCT_FRIENDLY_NAMES[key]
        return product or "this"

    @staticmethod
    def _normalize_product_key(product: Optional[str]) -> Optional[str]:
        if not product:
            return None
        return str(product).strip().lower()

    @classmethod
    def _set_fallback(cls, state: Any) -> None:
        state.session.pop("purchase_flow_stage", None)
        # Preserve last_question until next flow overrides or remove?
        state.session.pop("last_question", None)


