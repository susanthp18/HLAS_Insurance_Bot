import re
from datetime import datetime
from typing import Dict, Any, Optional
from ..prompt_runner import run_direct_task
from ..tasks import answer_status_task


def _normalize_date(val: str) -> Optional[str]:
    s = (val or "").strip()
    if not s:
        return None
    # Try YYYY-MM-DD first
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # Fallback: digits only, e.g., 20251110 -> 2025-11-10 (len 8)
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        try:
            dt = datetime.strptime(digits, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


class PolicyStatusFlowHelper:
    @staticmethod
    def _required_slots() -> list[str]:
        return ["policy_number", "full_name", "date_of_birth"]

    @staticmethod
    def _validate_slot(slot: str, value: str) -> Optional[str]:
        s = (value or "").strip()
        if not s:
            return None
        if slot == "policy_number":
            # Alphanumeric with optional hyphen/underscore, 6-16 chars
            return s if re.fullmatch(r"[A-Za-z0-9\-_]{6,16}", s) else None
        if slot == "full_name":
            # Letters/spaces/hyphens, at least 2 characters
            return s if re.fullmatch(r"[A-Za-z][A-Za-z\-\s]{1,}", s) else None
        if slot == "date_of_birth":
            return _normalize_date(s)
        return None

    @staticmethod
    def _ask_for(slot: str) -> str:
        if slot == "policy_number":
            return "Please provide your policy number (6–16 characters, letters/numbers)."
        if slot == "full_name":
            return "What is the full name on the policy?"
        if slot == "date_of_birth":
            return "What is your date of birth? (YYYY-MM-DD)"
        return "Please provide the requested detail."

    @staticmethod
    def _collect_or_ask(state, logger) -> Optional[str]:
        session: Dict[str, Any] = state.session
        slots: Dict[str, Any] = session.get("policy_status_slots") or {}
        last_slot: Optional[str] = session.get("policy_status_last_slot")

        # If we previously asked for a specific slot, attempt to validate and store current message
        if last_slot:
            normalized = PolicyStatusFlowHelper._validate_slot(last_slot, state.message or "")
            if normalized:
                slots[last_slot] = normalized
                session["policy_status_slots"] = slots
                session["policy_status_last_slot"] = None
            else:
                q = PolicyStatusFlowHelper._ask_for(last_slot) + " (The format seems off.)"
                state.reply = q
                session["last_question"] = q
                logger.info("PolicyStatusFlow: validation failed for %s", last_slot)
                return "__done__"

        # Find next missing
        for slot in PolicyStatusFlowHelper._required_slots():
            if not (slots.get(slot) or "").strip():
                q = PolicyStatusFlowHelper._ask_for(slot)
                state.reply = q
                session["policy_status_last_slot"] = slot
                session["last_question"] = q
                logger.info("PolicyStatusFlow: asking for %s", slot)
                return "__done__"

        # All collected
        return None

    @staticmethod
    def handle(state, args: Dict[str, Any], logger):
        session: Dict[str, Any] = state.session or {}
        # Initialize flow status
        if session.get("policy_status_status") not in ("in_progress", "done"):
            session["policy_status_status"] = "in_progress"
            session.setdefault("policy_status_slots", {})
            session["policy_status_last_slot"] = None

        # Collect/ask until slots complete
        pending = PolicyStatusFlowHelper._collect_or_ask(state, logger)
        if pending == "__done__":
            return "__done__"

        # Produce mock status
        slots = session.get("policy_status_slots") or {}
        mock_data = {
            "status": "Active",
            "expiry_date": "2026-12-31",
            "next_payment_due": "2026-01-15",
        }

        # Build context for LLM responder
        try:
            import json
            ctx_lines = []
            ctx_lines.append(f"type: policy")
            ctx_lines.append(f"user_question: {state.message}")
            ctx_lines.append("captured_fields:")
            ctx_lines.append(json.dumps(slots, ensure_ascii=False))
            ctx_lines.append("mock_data:")
            ctx_lines.append(json.dumps(mock_data, ensure_ascii=False))
            context_text = "\n".join(ctx_lines)
        except Exception:
            context_text = f"type: policy\nuser_question: {state.message}\n"

        out = run_direct_task(
            agent_obj=answer_status_task.agent,
            agent_key="status_responder",
            task_key="answer_status",
            context_text=context_text,
            logger=logger,
            label="policy_status.final_response",
        ) or {}

        reply = out.get("response") or (
            f"Here is your policy status:\n"
            f"• Status: {mock_data['status']}\n"
            f"• Expiry date: {mock_data['expiry_date']}\n"
            f"• Next payment due: {mock_data['next_payment_due']}"
        )

        state.reply = reply
        # Mark flow done and clean up ephemeral keys
        session["policy_status_status"] = "done"
        session.pop("policy_status_last_slot", None)
        session.pop("last_question", None)
        # Keep slots for audit; caller will clear status on next router pass
        return "__done__"


