from __future__ import annotations

import logging
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import _load_knowledge_base, _router_model

logger = logging.getLogger(__name__)

def _capabilities_tool(question: str) -> str:
    """Answer capability/meta questions using the static knowledge base."""

    kb_text = _load_knowledge_base()
    sys_msg = (
        "You are the HLAS Smart Bot. Answer questions about what you can do, "
        "which products you support, and how you help customers. Use only the "
        "knowledge base below; do not invent new capabilities."
    )
    user_parts: List[str] = [f"Question: {question}"]
    if kb_text:
        user_parts.append("")
        user_parts.append("Knowledge Base:")
        user_parts.append(kb_text)
    user_content = "\n".join(user_parts)

    try:
        msg = _router_model.invoke(
            [
                SystemMessage(content=sys_msg),
                HumanMessage(content=user_content),
            ]
        )
        return str(getattr(msg, "content", "") or "").strip() or (
            "I can help with product information, summaries, comparisons, "
            "recommendations and purchase links for HLAS insurance plans."
        )
    except Exception as e:
        logger.warning("Capabilities responder failed: %s", e)
        return (
            "I can help with product information, summaries, comparisons, and "
            "recommendations for HLAS insurance plans."
        )
