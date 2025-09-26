from typing import Dict, Any
import logging
from pathlib import Path
import yaml

from ..tasks import identify_product_task, identify_tiers_task
from ..prompt_runner import run_direct_task
from ..tools.benefits_tool import benefits_tool
from ..llm import azure_llm


class SummaryFlowHelper:
    """Intelligent, stateful summary handler.

    - Initializes and maintains `summary_slot` in session: {product, tiers[]}
    - Uses identifier agents to fill product/tiers (1+ tiers acceptable; Car ignores tiers)
    - Bypasses orchestrator while `summary_pending` exists
    - Generates guided clarification when needed (LLM-based, with safe fallback)
    - Retrieves only benefit chunks and synthesizes concise summary (or differences if 2+ tiers)
    - Cleans up to allow smooth transition to normal flow
    Returns "__done__" always.
    """

    @staticmethod
    def handle(state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        session = state.session if isinstance(getattr(state, "session", None), dict) else {}
        message = state.message or ""

        # ---- Entry & state bootstrap ----
        summary_slot = session.get("summary_slot")
        first_msg = summary_slot is None
        if first_msg:
            summary_slot = {"product": None, "tiers": []}
            session["summary_slot"] = summary_slot
        try:
            logger.info(
                "SummaryFlow.start: first_msg=%s message=%s session_keys=%s",
                first_msg,
                (message[:200] if message else ""),
                list(session.keys()),
            )
        except Exception:
            pass

        # Utility: ask guided clarification (reuse followup clarification agent)
        def ask_clarify(await_key: str, product_hint: str | None, tiers_hint: list[str] | None) -> str:
            try:
                ctx_lines = [
                    f"await={await_key}",
                    f"product={product_hint or ''}",
                    f"known_tiers={', '.join(tiers_hint or [])}",
                    f"flow_type=summary",
                ]
                # Add available tiers for the identified product
                if product_hint:
                    pl = product_hint.lower()
                    if pl == "travel":
                        ctx_lines.append("available_tiers=Basic, Silver, Gold, Platinum")
                    elif pl == "maid":
                        ctx_lines.append("available_tiers=Basic, Enhanced, Premier, Exclusive")
                    elif pl == "car":
                        ctx_lines.append("available_tiers=None (Car has no tiers)")
                # Include brief history window to resolve "above plan(s)"
                hist = session.get("history", []) or []
                for pair in (hist[-3:] if hist else []):
                    try:
                        ctx_lines.append(f"User: {pair.get('user','')}")
                        ctx_lines.append(f"Assistant: {pair.get('assistant','')}")
                    except Exception:
                        continue
                context_text = "\n".join(ctx_lines)
                from ..prompt_runner import run_direct_task
                from ..agents import followup_clarification_agent as clar_agent
                res = run_direct_task(
                    agent_obj=clar_agent,
                    agent_key="followup_clarification_agent",
                    task_key="followup_clarification",
                    context_text=context_text,
                    logger=logger,
                    label="summary.followup_clarification.generate",
                ) or {}
                # Extract question from JSON response or text fallback
                q = ""
                if isinstance(res, dict):
                    q = (res.get("question") or res.get("response") or "").strip()
                else:
                    q = str(res).strip() if res else ""
                try:
                    logger.info(
                        "SummaryFlow.clarification_agent_result: res_type=%s res_keys=%s question_len=%d",
                        type(res).__name__, list(res.keys()) if isinstance(res, dict) else "N/A", len(q)
                    )
                except Exception:
                    pass
                if q:
                    return q
            except Exception:
                pass
            # Fallbacks
            if await_key == "product":
                return "Which product would you like summarized: Travel, Maid, or Car?"
            if await_key == "tiers":
                prod = (product_hint or "").lower()
                if prod == "travel":
                    return "Which Travel tier(s) should I summarize? Available: Basic, Silver, Gold, Platinum"
                if prod == "maid":
                    return "Which Maid tier(s) should I summarize? Available: Basic, Enhanced, Premier, Exclusive"
                if prod == "car":
                    return "Car has no tiers. Which aspects should I summarize?"
                return "Which tier(s) should I summarize?"
            return "Could you clarify what you want me to summarize?"

        # Helper: identify product once
        def ensure_product() -> None:
            if summary_slot.get("product"):
                return
            existing = session.get("product") or state.product
            if existing:
                summary_slot["product"] = existing
                return
            prod = run_direct_task(
                agent_obj=identify_product_task.agent,
                agent_key="product_identifier",
                task_key="identify_product",
                context_text=f"User Message: {message}\nSession product: {session.get('product')}",
                logger=logger,
                label="product_identifier.identify_product.for_summary",
            ) or {}
            try:
                logger.info(
                    "SummaryFlow.product_identified: product=%s confidence=%s",
                    prod.get("product"), prod.get("confidence"),
                )
            except Exception:
                pass
            if prod.get("product"):
                # If product changed, clear existing tiers to avoid cross-product leakage
                old_product = summary_slot.get("product")
                new_product = prod.get("product")
                if old_product and old_product != new_product:
                    summary_slot["tiers"] = []
                    logger.info("SummaryFlow.product_switch: cleared tiers (%s -> %s)", old_product, new_product)
                summary_slot["product"] = new_product
                state.product = new_product
                session["product"] = new_product

        # Helper: identify tiers (allow 1+ for summary; Car ignores tiers)
        def ensure_tiers() -> None:
            if (summary_slot.get("product") or "").lower() == "car":
                return
            tiers = summary_slot.get("tiers") or []
            if len(tiers) >= 1:
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
                f"Product: {summary_slot.get('product')}\n"
                f"User Message: {message}\n"
                f"Recent conversation (most recent first):\n" + "\n".join(ctx_lines)
            )
            tiers_res = run_direct_task(
                agent_obj=identify_tiers_task.agent,
                agent_key="tier_identifier",
                task_key="identify_tiers",
                context_text=tiers_ctx,
                logger=logger,
                label="tier_identifier.identify_tiers.for_summary",
            ) or {}
            new_tiers = tiers_res.get("tiers") or []
            inferred_product = tiers_res.get("product")
            if inferred_product:
                current_product = summary_slot.get("product")
                if not current_product:
                    # No product yet, accept inferred
                    summary_slot["product"] = inferred_product
                    state.product = inferred_product
                    session["product"] = inferred_product
                    try:
                        logger.info("SummaryFlow.product_inferred_from_tiers: product=%s", inferred_product)
                    except Exception:
                        pass
                elif current_product != inferred_product:
                    # Product switch detected, clear existing tiers
                    summary_slot["tiers"] = []
                    summary_slot["product"] = inferred_product
                    state.product = inferred_product
                    session["product"] = inferred_product
                    try:
                        logger.info("SummaryFlow.product_switch_from_tiers: (%s -> %s), cleared tiers", current_product, inferred_product)
                    except Exception:
                        pass
            merged = []
            for t in (tiers + new_tiers):
                if t and t not in merged:
                    merged.append(t)
            summary_slot["tiers"] = merged
            try:
                logger.info("SummaryFlow.tiers_identified: tiers=%s", summary_slot["tiers"])
            except Exception:
                pass

        # ---- First pass: fill from message/history ----
        ensure_product()
        ensure_tiers()

        product = (summary_slot.get("product") or "").strip()
        tiers_list = summary_slot.get("tiers") or []

        # Missing product → ask
        if not product:
            session["summary_status"] = "in_progress"
            q = ask_clarify("product", None, None)
            state.reply = q
            logger.info("SummaryFlow.pending: await=product")
            logger.info("SummaryFlow.clarify_question: %s", q)
            return "__done__"

        # Car: ignore tiers and proceed
        if product.lower() == "car":
            pass
        else:
            # Need at least 1 tier for summary
            if len(tiers_list) < 1:
                session["summary_status"] = "in_progress"
                q = ask_clarify("tiers", product, tiers_list)
                state.reply = q
                logger.info("SummaryFlow.pending: await=tiers")
                logger.info("SummaryFlow.clarify_question: %s", q)
                return "__done__"

        # ---- Ready: have product and (optionally multiple) tiers ----
        try:
            logger.info("SummaryFlow.ready: product=%s tiers=%s", product, tiers_list)
        except Exception:
            pass

        # Retrieve benefits (product-only; template will focus by tiers)
        try:
            benefits_text = benefits_tool.run(product=product)
        except Exception:
            benefits_text = ""
        try:
            logger.info("SummaryFlow.benefits: text_len=%d", len(benefits_text or ""))
        except Exception:
            pass

        # Load summary templates
        try:
            base_dir = Path(__file__).resolve().parent.parent
            with open(base_dir / "config" / "summary_response.yaml", "r", encoding="utf-8") as rf:
                sum_templates = yaml.safe_load(rf) or {}
        except Exception:
            sum_templates = {}

        tpl = sum_templates.get(product.lower(), {})
        sys_t = tpl.get("system") or "You are an insurance summary responder. Summarize succinctly using only the provided context."
        tiers_txt = ", ".join(tiers_list) if tiers_list else ("N/A" if product.lower()=="car" else "")
        usr_t = (tpl.get("user") or "Product: {product}\nTiers: {tiers}\nQuestion: {question}\n\n[Context]\n{context}").format(
            product=product,
            tiers=tiers_txt,
            question=message,
            context=benefits_text or "",
        )

        logger.info("LLM Direct [summary.synthesis]:\n[SYSTEM]\n%s\n\n[USER]\n%s", sys_t, usr_t)
        try:
            txt = azure_llm.call(messages=[
                {"role": "system", "content": sys_t},
                {"role": "user", "content": usr_t},
            ])
            answer = str(txt).strip()
        except Exception:
            answer = ""

        # Set reply with safe fallback
        state.reply = answer or ("Here is a concise summary." if product.lower()=="car" else "Which tier should I summarize?")
        try:
            logger.info("SummaryFlow.reply_len=%d", len(state.reply or ""))
        except Exception:
            pass

        # Mark flow as done and clean up the working slot
        session["summary_status"] = "done"
        session.pop("summary_slot", None)

        logger.info("SummaryFlow.completed: status set to 'done' and slot cleared.")

        try:
            session.setdefault("summary_history", [])
            session["summary_history"].append({
                "product": product,
                "tiers": list(tiers_list),
                "completed": True,
            })
            if len(session["summary_history"]) > 10:
                session["summary_history"] = session["summary_history"][-10:]
        except Exception:
            pass
        try:
            session["last_completed"] = "summary"
        except Exception:
            pass
        return "__done__"


