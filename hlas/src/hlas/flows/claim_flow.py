import re
from datetime import datetime
from typing import Dict, Any, Optional
from ..prompt_runner import run_direct_task
from ..tasks import answer_status_task


def _normalize_date(val: str) -> Optional[str]:
    s = (val or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        try:
            dt = datetime.strptime(digits, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


class ClaimStatusFlowHelper:
    @staticmethod
    def _required_slots() -> list[str]:
        return ["claim_number", "policy_number", "date_of_incident"]

    @staticmethod
    def _validate_slot(slot: str, value: str) -> Optional[str]:
        s = (value or "").strip()
        if not s:
            return None
        if slot in ("claim_number", "policy_number"):
            return s if re.fullmatch(r"[A-Za-z0-9\-_]{6,16}", s) else None
        if slot == "date_of_incident":
            return _normalize_date(s)
        return None

    @staticmethod
    def _ask_for(slot: str) -> str:
        if slot == "claim_number":
            return "Please provide your claim number (6–16 characters, letters/numbers)."
        if slot == "policy_number":
            return "Please provide the associated policy number (6–16 characters)."
        if slot == "date_of_incident":
            return "What is the date of incident? (YYYY-MM-DD)"
        return "Please provide the requested detail."

    @staticmethod
    def _collect_or_ask(state, logger) -> Optional[str]:
        session: Dict[str, Any] = state.session
        slots: Dict[str, Any] = session.get("claim_status_slots") or {}
        last_slot: Optional[str] = session.get("claim_status_last_slot")

        if last_slot:
            normalized = ClaimStatusFlowHelper._validate_slot(last_slot, state.message or "")
            if normalized:
                slots[last_slot] = normalized
                session["claim_status_slots"] = slots
                session["claim_status_last_slot"] = None
            else:
                q = ClaimStatusFlowHelper._ask_for(last_slot) + " (The format seems off.)"
                state.reply = q
                session["last_question"] = q
                logger.info("ClaimStatusFlow: validation failed for %s", last_slot)
                return "__done__"

        for slot in ClaimStatusFlowHelper._required_slots():
            if not (slots.get(slot) or "").strip():
                q = ClaimStatusFlowHelper._ask_for(slot)
                state.reply = q
                session["claim_status_last_slot"] = slot
                session["last_question"] = q
                logger.info("ClaimStatusFlow: asking for %s", slot)
                return "__done__"

        return None

    @staticmethod
    def handle(state, args: Dict[str, Any], logger):
        session: Dict[str, Any] = state.session or {}
        if session.get("claim_status_status") not in ("in_progress", "done"):
            session["claim_status_status"] = "in_progress"
            session.setdefault("claim_status_slots", {})
            session["claim_status_last_slot"] = None

        pending = ClaimStatusFlowHelper._collect_or_ask(state, logger)
        if pending == "__done__":
            return "__done__"

        slots = session.get("claim_status_slots") or {}
        mock_data = {
            "status": "Processing",
            "last_update": "2025-11-01",
            "estimated_completion": "2025-11-20",
        }

        try:
            import json
            ctx_lines = []
            ctx_lines.append(f"type: claim")
            ctx_lines.append(f"user_question: {state.message}")
            ctx_lines.append("captured_fields:")
            ctx_lines.append(json.dumps(slots, ensure_ascii=False))
            ctx_lines.append("mock_data:")
            ctx_lines.append(json.dumps(mock_data, ensure_ascii=False))
            context_text = "\n".join(ctx_lines)
        except Exception:
            context_text = f"type: claim\nuser_question: {state.message}\n"

        out = run_direct_task(
            agent_obj=answer_status_task.agent,
            agent_key="status_responder",
            task_key="answer_status",
            context_text=context_text,
            logger=logger,
            label="claim_status.final_response",
        ) or {}

        reply = out.get("response") or (
            f"Here is your claim status:\n"
            f"• Status: {mock_data['status']}\n"
            f"• Last update: {mock_data['last_update']}\n"
            f"• Estimated completion: {mock_data['estimated_completion']}"
        )

        state.reply = reply
        session["claim_status_status"] = "done"
        session.pop("claim_status_last_slot", None)
        session.pop("last_question", None)
        return "__done__"


