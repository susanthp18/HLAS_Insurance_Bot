from __future__ import annotations

"""
Lightweight LangChain-compatible structured output wrappers.

Design:
- Optional: only used when LC_STRUCTURED_ENABLED=true and LangChain is importable.
- Falls back to None so callers can use existing JSON/regex parsing.

This module intentionally does NOT import LangChain at module import time.
It imports lazily inside functions to avoid hard dependency when disabled.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import os
import logging


# -----------------
# Pydantic schemas
# -----------------

class RouteDecisionOutput(BaseModel):
    directive: Literal[
        "greet",
        "handle_recommendation",
        "handle_information",
        "handle_follow_up",
        "plan_only_comparison",
        "handle_summary",
        "handle_capabilities",
        "live_agent",
        "handle_other",
    ]


class IdentifyProductOutput(BaseModel):
    product: str = Field(default="")
    confidence: float = Field(default=0.0)
    question: Optional[str] = None


class FollowUpQueryOutput(BaseModel):
    query: str
    query_type: Literal[
        "plan_only_comparison",
        "handle_summary",
        "handle_recommendation",
        "handle_information",
    ]
    routing_confidence: float = 0.0
    evidence: Optional[str] = None


class ValidateSlotOutput(BaseModel):
    valid: bool
    slot_name: str
    normalized_value: Optional[str] = None
    question: Optional[str] = None
    reason: Optional[str] = None


def _get_schema_for_task(task_key: str):
    mapping = {
        "route_decision": RouteDecisionOutput,
        "identify_product": IdentifyProductOutput,
        "construct_follow_up_query": FollowUpQueryOutput,
        "validate_slot": ValidateSlotOutput,
    }
    return mapping.get(task_key)


def structured_invoke(
    *,
    system_prompt: str,
    user_prompt: str,
    task_key: str,
    logger: Optional[logging.Logger] = None,
    label: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Try to call a LangChain chat model with with_structured_output for the given task.
    Returns dict on success, or None to indicate the caller should fall back.
    """
    if os.getenv("LC_STRUCTURED_ENABLED", "false").lower() not in ("1", "true", "yes"):  # feature gate
        return None

    schema = _get_schema_for_task(task_key)
    if schema is None:
        return None

    try:
        # Import lazily to keep LC optional
        from langchain_core.prompts import ChatPromptTemplate
        # Try Azure first if configured
        use_azure = (
            os.getenv("AZURE_OPENAI_API_KEY")
            and os.getenv("AZURE_OPENAI_ENDPOINT")
            and os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )

        if use_azure:
            try:
                from langchain_openai import AzureChatOpenAI  # type: ignore
                model = AzureChatOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
                    temperature=float(os.getenv("LC_TEMPERATURE", "0")),
                    max_tokens=int(os.getenv("LC_MAX_TOKENS", "512")),
                )
            except Exception:
                # Fallback to non-Azure if Azure client is not available
                use_azure = False
        if not use_azure:
            from langchain_openai import ChatOpenAI  # type: ignore
            model = ChatOpenAI(
                model=os.getenv("LC_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
                temperature=float(os.getenv("LC_TEMPERATURE", "0")),
                max_tokens=int(os.getenv("LC_MAX_TOKENS", "512")),
            )

        structured = model.with_structured_output(schema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            ("user", "{user}"),
        ])

        if logger:
            try:
                logger.info("LC Structured [%s]:\n[SYSTEM]\n%s\n\n[USER]\n%s", label, system_prompt, user_prompt)
            except Exception:
                pass

        chain = prompt | structured
        out = chain.invoke({"system": system_prompt, "user": user_prompt})
        # out is a Pydantic model; convert to dict
        if isinstance(out, BaseModel):
            return out.model_dump()
        if isinstance(out, dict):
            return out
        try:
            # As a last resort
            return dict(out)
        except Exception:
            return None

    except ImportError:
        # LangChain not installed
        return None
    except Exception as e:
        if logger:
            try:
                logger.warning("LC Structured [%s] failed: %s", label, str(e))
            except Exception:
                pass
        return None



