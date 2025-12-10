from __future__ import annotations

from typing import List, Optional

# Local infrastructure imports (lazy singleton - no explicit init needed)
from ..infrastructure import get_response_llm
from .benefits import get_product_benefits
from ..config import _load_summary_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm


def _summary_tool(
    product: Optional[str], tiers: List[str], question: str
) -> tuple[str, List[str]]:
    """Summary tool: product/tier summaries using benefits and templates."""

    prod = _normalize_product_key(product)
    if not prod:
        prod = _normalize_product_key(_detect_product_llm(question))
    if not prod:
        ask = (
            "Which product would you like a summary for: Travel, Maid, Car, Personal Accident, "
            "Home, Early, Fraud or Hospital?"
        )
        return ask, []

    try:
        benefits_text = get_product_benefits(prod)
    except Exception:
        benefits_text = ""

    sum_templates = _load_summary_templates()
    tpl = sum_templates.get(prod, {}) if sum_templates else {}
    sys_t = tpl.get("system") or (
        "You are an insurance summary responder. Summarize succinctly using only the provided context."
    )
    tiers_txt = ", ".join(tiers) if tiers else ("N/A" if prod in ("car", "early") else "")
    usr_t = (tpl.get("user") or "Product: {product}\nTiers: {tiers}\nQuestion: {question}\n\n[Context]\n{context}").format(
        product=prod,
        tiers=tiers_txt,
        question=question,
        context=benefits_text or "",
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_response_llm()  # Thread-safe singleton
        messages = [SystemMessage(content=sys_t), HumanMessage(content=usr_t)]
        response = llm.invoke(messages)
        answer = str(response.content).strip()
    except Exception:
        answer = ""

    if not answer:
        answer = (
            "Here is a concise overview of our {prod} plans. You can also ask about specific benefits or tiers."
        ).format(prod=prod.title())
    return answer, []
