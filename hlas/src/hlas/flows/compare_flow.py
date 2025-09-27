from typing import Dict, Any
import logging
from pathlib import Path
import yaml

from ..tasks import identify_product_task, identify_tiers_task
from ..prompt_runner import run_direct_task
from ..llm import azure_llm, azure_response_llm
from ..tools.benefits_tool import benefits_tool


class CompareFlowHelper:
    """Intelligent, stateful comparison handler.

    - Initializes and maintains `comparison_slot` in session: {product, tiers[]}
    - Uses identifier agents to fill product/tiers once; avoids re-calling after set
    - Bypasses orchestrator while `compare_pending` exists
    - Generates guided clarification when needed (LLM-based, with safe fallback)
    - Retrieves only benefit chunks and synthesizes concise comparison
    - Cleans up to allow smooth transition to normal flow
    Returns "__done__" always.
    """

    @staticmethod
    def handle(state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        # ---- Entry & state bootstrap ----
        session = state.session if isinstance(getattr(state, "session", None), dict) else {}
        message = state.message or ""
        comparison_slot = session.get("comparison_slot")
        first_msg = comparison_slot is None

        # Initialize comparison_slot on first message
        if first_msg:
            comparison_slot = {"product": None, "tiers": []}
            session["comparison_slot"] = comparison_slot
        try:
            logger.info(
                "CompareFlow.start: first_msg=%s message=%s session_keys=%s",
                first_msg,
                (message[:200] if message else ""),
                list(session.keys()),
            )
        except Exception:
            pass

        # Utility: ask guided clarification (LLM with safe fallback)
        def ask_clarify(await_key: str, product_hint: str | None, tiers_hint: list[str] | None) -> str:
            try:
                # Use dedicated followup clarification agent to generate one short question
                ctx_lines = [
                    f"await={await_key}",
                    f"product={product_hint or ''}",
                    f"known_tiers={', '.join(tiers_hint or [])}",
                    f"flow_type=comparison",
                ]
                # Add available tiers for the identified product
                if product_hint:
                    prod_lower = product_hint.lower()
                    if prod_lower == "travel":
                        ctx_lines.append("available_tiers=Basic, Silver, Gold, Platinum")
                    elif prod_lower == "maid":
                        ctx_lines.append("available_tiers=Basic, Enhanced, Premier, Exclusive")
                    elif prod_lower == "personalaccident":
                        ctx_lines.append("available_tiers=Bronze, Silver, Premier, Platinum")
                    elif prod_lower == "car":
                        ctx_lines.append("available_tiers=None (Car has no tiers)")
                
                hist = session.get("history", []) or []
                for pair in (hist[-3:] if hist else []):
                    try:
                        ctx_lines.append(f"User: {pair.get('user','')}")
                        ctx_lines.append(f"Assistant: {pair.get('assistant','')}")
                    except Exception:
                        continue
                context_text = "\n".join(ctx_lines)
                from ..tasks import build_task
                # Build on-the-fly or reuse mapped task if present
                try:
                    # Prefer mapped task
                    from ..tasks import task_by_name  # type: ignore
                    task_obj = task_by_name.get("followup_clarification")  # type: ignore
                except Exception:
                    task_obj = None
                if task_obj is None:
                    # Fallback: run through prompt_runner with configured task
                    from ..prompt_runner import run_direct_task
                    from ..agents import followup_clarification_agent as clar_agent
                    res = run_direct_task(
                        agent_obj=clar_agent,
                        agent_key="followup_clarification_agent",
                        task_key="followup_clarification",
                        context_text=context_text,
                        logger=logger,
                        label="followup_clarification.generate",
                    ) or {}
                else:
                    from ..prompt_runner import run_direct_task
                    res = run_direct_task(
                        agent_obj=task_obj.agent,
                        agent_key="followup_clarification_agent",
                        task_key="followup_clarification",
                        context_text=context_text,
                        logger=logger,
                        label="followup_clarification.generate",
                    ) or {}
                # Extract question from JSON response or text fallback
                q = ""
                if isinstance(res, dict):
                    q = (res.get("question") or res.get("response") or "").strip()
                else:
                    q = str(res).strip() if res else ""
                try:
                    logger.info("CompareFlow.clarification_agent_result: res_type=%s res_keys=%s question_len=%d", 
                               type(res).__name__, list(res.keys()) if isinstance(res, dict) else "N/A", len(q))
                except Exception:
                    pass
                if q:
                    return q
            except Exception:
                pass
            # Fallbacks
            if await_key == "product":
                return "Which product would you like to compare: Travel, Maid, or Car?"
            if await_key == "tiers":
                prod = (product_hint or "").lower()
                if prod == "travel":
                    return "Which Travel tiers would you like to compare? Available: Basic, Silver, Gold, Platinum"
                if prod == "maid":
                    return "Which Maid tiers would you like to compare? Available: Basic, Enhanced, Premier, Exclusive"
                if prod == "personalaccident":
                    return "Which Personal Accident tiers would you like to compare? Available: Bronze, Silver, Premier, Platinum"
                if prod == "car":
                    return "Car has no tiers to compare. Which aspects would you like me to compare?"
                return "Which two tiers should I compare?"
            return "Could you clarify what you want me to compare?"

        # Helper: identify product once (no user-facing output from identifier agent)
        def ensure_product() -> None:
            if comparison_slot.get("product"):
                return
            # Try from existing session first
            existing = session.get("product") or state.product
            if existing:
                comparison_slot["product"] = existing
                return
            # Call product identifier once
            prod = run_direct_task(
                agent_obj=identify_product_task.agent,
                agent_key="product_identifier",
                task_key="identify_product",
                context_text=f"User Message: {message}\nSession product: {session.get('product')}",
                logger=logger,
                label="product_identifier.identify_product.for_comparison",
            ) or {}
            try:
                logger.info(
                    "CompareFlow.product_identified: product=%s confidence=%s",
                    prod.get("product"), prod.get("confidence"),
                )
            except Exception:
                pass
            if prod.get("product"):
                # If product changed, clear existing tiers to avoid cross-product leakage
                old_product = comparison_slot.get("product")
                new_product = prod.get("product")
                if old_product and old_product != new_product:
                    comparison_slot["tiers"] = []
                    logger.info("CompareFlow.product_switch: cleared tiers (%s -> %s)", old_product, new_product)
                comparison_slot["product"] = new_product
                state.product = new_product
                session["product"] = new_product

        # Helper: identify tiers until at least two (do not emit identifier question)
        def ensure_tiers() -> None:
            if (comparison_slot.get("product") or "").lower() == "car":
                return
            tiers = comparison_slot.get("tiers") or []
            if len(tiers) >= 2:
                return
            hist = session.get("history", []) or []
            history_pairs = hist[-3:]
            ctx_lines = []
            for pair in history_pairs:
                try:
                    ctx_lines.append(f"User: {pair.get('user','')}")
                    ctx_lines.append(f"Assistant: {pair.get('assistant','')}")
                except Exception:
                    continue
            tiers_ctx = (
                f"Product: {comparison_slot.get('product')}\n"
                f"User Message: {message}\n"
                f"Recent conversation (most recent first):\n" + "\n".join(ctx_lines)
            )
            tiers_res = run_direct_task(
                agent_obj=identify_tiers_task.agent,
                agent_key="tier_identifier",
                task_key="identify_tiers",
                context_text=tiers_ctx,
                logger=logger,
                label="tier_identifier.identify_tiers",
            ) or {}
            new_tiers = tiers_res.get("tiers") or []
            # If identifier infers product, handle product switches
            inferred_product = tiers_res.get("product")
            if inferred_product:
                current_product = comparison_slot.get("product")
                if not current_product:
                    # No product yet, accept inferred
                    comparison_slot["product"] = inferred_product
                    state.product = inferred_product
                    session["product"] = inferred_product
                    try:
                        logger.info("CompareFlow.product_inferred_from_tiers: product=%s", inferred_product)
                    except Exception:
                        pass
                elif current_product != inferred_product:
                    # Product switch detected, clear existing tiers
                    comparison_slot["tiers"] = []
                    comparison_slot["product"] = inferred_product
                    state.product = inferred_product
                    session["product"] = inferred_product
                    try:
                        logger.info("CompareFlow.product_switch_from_tiers: (%s -> %s), cleared tiers", current_product, inferred_product)
                    except Exception:
                        pass
            # Deduplicate, preserve order
            merged = []
            for t in (tiers + new_tiers):
                if t and t not in merged:
                    merged.append(t)
            comparison_slot["tiers"] = merged
            try:
                logger.info("CompareFlow.tiers_identified: tiers=%s", comparison_slot["tiers"])
            except Exception:
                pass

        # ---- First pass: fill what we can without asking ----
        ensure_product()
        ensure_tiers()

        product = (comparison_slot.get("product") or "").strip()
        tiers_list = comparison_slot.get("tiers") or []

        # Handle missing product
        if not product:
            # Set status and ask a single guided question
            session["comparison_status"] = "in_progress"
            q = ask_clarify("product", None, None)
            state.reply = q
            logger.info("CompareFlow.pending: await=product")
            logger.info("CompareFlow.clarify_question: %s", q)
            return "__done__"

        # Car product: ignore tiers and proceed
        if product.lower() == "car":
            if tiers_list:
                logger.info("CompareFlow: Car has no tiers; ignoring tiers=%s", tiers_list)
            # Proceed to synthesis directly for Car
        else:
            # Need at least two tiers
            if len(tiers_list) < 2:
                session["comparison_status"] = "in_progress"
                q = ask_clarify("tiers", product, tiers_list)
                state.reply = q
                logger.info("CompareFlow.pending: await=tiers")
                logger.info("CompareFlow.clarify_question: %s", q)
                return "__done__"

        # ---- Ready: both product and tiers (or Car) identified ----
        try:
            logger.info("CompareFlow.ready: product=%s tiers=%s", product, tiers_list)
        except Exception:
            pass

        # Retrieve all benefits for product
        try:
            benefits_text = benefits_tool.run(product=product)
        except Exception:
            benefits_text = ""
        try:
            logger.info("CompareFlow.benefits: text_len=%d", len(benefits_text or ""))
        except Exception:
            pass
        context_str = benefits_text or ""

        # Load comparison templates
        try:
            base_dir = Path(__file__).resolve().parent.parent
            with open(base_dir / "config" / "cmp_response.yaml", "r", encoding="utf-8") as rf:
                cmp_templates = yaml.safe_load(rf) or {}
        except Exception:
            cmp_templates = {}

        tpl = cmp_templates.get(product.lower(), {})
        sys_t = tpl.get("system") or "You are an insurance comparison responder. Compare tiers succinctly using only the provided context."
        tiers_txt = ", ".join(tiers_list) if tiers_list else ("N/A" if product.lower()=="car" else "")
        usr_t = (tpl.get("user") or "Product: {product}\nTiers: {tiers}\nQuestion: {question}\n\n[Context]\n{context}").format(
            product=product,
            tiers=tiers_txt,
            question=message,
            context=context_str,
        )

        logger.info("LLM Direct [comparison.synthesis]:\n[SYSTEM]\n%s\n\n[USER]\n%s", sys_t, usr_t)
        try:
            # Use response LLM for user-facing comparison synthesis
            txt = azure_response_llm.call(messages=[
                {"role": "system", "content": sys_t},
                {"role": "user", "content": usr_t},
            ])
            answer = str(txt).strip()
        except Exception:
            answer = ""

        state.reply = answer or ("Here is a concise comparison." if product.lower()=="car" else "Which two tiers should I compare?")
        try:
            logger.info("CompareFlow.reply_len=%d", len(state.reply or ""))
        except Exception:
            pass

        # Mark flow as done and clean up the working slot
        session["comparison_status"] = "done"
        session.pop("comparison_slot", None)

        logger.info("CompareFlow.completed: status set to 'done' and slot cleared.")

        try:
            session.setdefault("comparison_history", [])
            session["comparison_history"].append({
                "product": product,
                "tiers": list(tiers_list),
                "completed": True,
            })
            # Cap history length
            if len(session["comparison_history"]) > 10:
                session["comparison_history"] = session["comparison_history"][-10:]
        except Exception:
            pass
        try:
            session["last_completed"] = "comparison"
        except Exception:
            pass
        return "__done__"


