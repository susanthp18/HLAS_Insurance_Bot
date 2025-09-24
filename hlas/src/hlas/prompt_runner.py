from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import re
import yaml
import logging
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover - fallback if zoneinfo unavailable
    ZoneInfo = None


# Load specs once at import
_BASE_DIR = Path(__file__).parent
try:
    with open(_BASE_DIR / "config/agents.yaml", "r", encoding="utf-8") as _af:
        AGENTS_SPEC: Dict[str, Any] = yaml.safe_load(_af) or {}
except Exception:
    AGENTS_SPEC = {}

try:
    with open(_BASE_DIR / "config/tasks.yaml", "r", encoding="utf-8") as _tf:
        TASKS_SPEC: Dict[str, Any] = yaml.safe_load(_tf) or {}
except Exception:
    TASKS_SPEC = {}


def build_prompts(agent_key: str, task_key: str, context_text: str, logger: logging.Logger) -> tuple[str, str]:
    agent_spec = AGENTS_SPEC.get(agent_key, {})
    task_spec = TASKS_SPEC.get(task_key, {})
    role = agent_spec.get("role", agent_key)
    backstory = (agent_spec.get("backstory") or "").strip()
    goal = (agent_spec.get("goal") or "").strip()
    description = (task_spec.get("description") or "").strip()
    expected = (task_spec.get("expected_output") or "").strip()

    # Force include 'reason' in contract for validate_slot (unified validator)
    if task_key == "validate_slot" and "\"reason\"" not in expected:
        expected = '{ "valid": true|false, "slot_name": string, "normalized_value"?: string, "question"?: string, "reason"?: string }'

    # Build system prompt first
    system_prompt = (
        f"You are {role}. {backstory}\n{goal}\n\n"
        f"Output contract (JSON):\n{expected}"
    ).strip()

    # Add an explicit focus line to the SYSTEM prompt for validators
    if task_key in ("validate_slot",):
        try:
            slot_name = None
            for _line in context_text.splitlines():
                if _line.lower().startswith("slot:"):
                    slot_name = _line.split(":", 1)[1].strip()
                    break
            system_prompt = f"{system_prompt}\n\nFocus only on validating {slot_name or 'the provided slot'}."
        except Exception:
            system_prompt = f"{system_prompt}\n\nFocus only on validating the provided slot."

    # USER prompt will now contain the rules within its context, as passed from the flow helper
    # Check if context_text already has structured [System]/[User] sections to avoid duplication
    if context_text.strip().startswith(("[System]", "[User]")):
        user_prompt = context_text.strip()
    else:
        user_prompt = (
            f"Task:\n{description}\n\n"
            f"[Context]\n{context_text}"
        ).strip()
    return system_prompt, user_prompt


def call_direct_json(agent_obj: Any, system_prompt: str, user_prompt: str, logger: Any, label: str, allow_text_fallback: bool = False) -> Dict[str, Any]:
    try:
        # Log actual prompts
        logger.info("LLM Direct [%s]:\n[SYSTEM]\n%s\n\n[USER]\n%s", label, system_prompt, user_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = agent_obj.llm.call(messages=messages)
        txt = str(raw).strip()
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            # Optional fallback: wrap raw text as JSON for text-only tasks
            if allow_text_fallback and txt:
                try:
                    logger.info("LLM Direct [%s]: no JSON detected; using text fallback (len=%d)", label, len(txt))
                except Exception:
                    pass
                return {"response": txt}
            return {}
    except Exception:
        return {}


def run_direct_task(agent_obj: Any, agent_key: str, task_key: str, context_text: str, logger: Any, label: str) -> Dict[str, Any]:
    system_prompt, user_prompt = build_prompts(agent_key, task_key, context_text, logger)
    allow_text = task_key in ("synthesize_response", "followup_clarification")
    return call_direct_json(agent_obj, system_prompt, user_prompt, logger, label, allow_text_fallback=allow_text)


