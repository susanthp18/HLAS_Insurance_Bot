from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
from langchain_core.tools import tool

from .info import _info_tool
from .compare import _compare_tool
from .recommendation import _generate_recommendation_text
from .purchase import _purchase_tool
from ..utils.slots import _normalize_product_key

logger = logging.getLogger(__name__)

@tool
def search_product_knowledge(query: str, product: Optional[str] = None) -> str:
    """Search the HLAS knowledge base (RAG) for specific coverage details, benefits, exclusions, or claims info.

    LLM NOTE: When the user asks what is covered, not covered, excluded, or how a policy applies in a
    specific scenario (e.g., whether a destination or situation is covered), you MUST call this tool
    instead of answering from your own general knowledge.

    Args:
        query: The user's question or keywords (e.g., "Does travel insurance cover covid?").
        product: The product to filter by (e.g., "travel", "maid"). If unknown, leave null.
    """
    query_preview = (query or "").replace("\n", " ")[:160]
    logger.info(
        "AgenticTool.search_product_knowledge: product=%s query='%s'",
        product,
        query_preview,
    )

    ans, sources = _info_tool(product, query)
    logger.debug(
        "AgenticTool.search_product_knowledge.completed: product=%s sources=%d",
        product,
        len(sources or []),
    )
    if sources:
        # Append sources so the LLM can see them.
        return f"{ans}\n\n(Sources: {', '.join(sources)})"
    return ans

@tool
def compare_plans(product: str, question: str) -> str:
    """
    Compare different tiers or plans for a specific product.
    
    IMPORTANT: The output contains coverage amounts. You MUST include these dollar amounts in your response.
    
    Args:
        product: The product name (e.g., "travel", "maid").
        question: The user's comparison question (e.g., "What is the difference between Basic and Premier?").
    """
    question_preview = (question or "").replace("\n", " ")[:160]
    logger.info(
        "AgenticTool.compare_plans: product=%s question='%s'",
        product,
        question_preview,
    )

    # We pass empty tiers list as the underlying tool handles it or extracts from question
    ans, _ = _compare_tool(product, [], question)
    return ans

@tool
def get_product_recommendation(product: str, slots: Dict[str, Any]) -> str:
    """
    Generate a specific plan recommendation based on collected user information.
    ONLY use this tool when you have collected the REQUIRED slots for the product.
    
    IMPORTANT: The output contains coverage amounts (e.g., "$500,000", "$300,000"). 
    You MUST include these dollar amounts in your response to the user.
    
    Args:
        product: The product name (e.g., "travel", "maid").
        slots: A dictionary of collected information. 
               Keys must match the required slots (e.g., {"destination": "Japan", "coverage_scope": "Individual"}).
    """
    prod = _normalize_product_key(product)
    if not prod:
        return "Error: Could not identify a valid product for recommendation."

    logger.info(
        "AgenticTool.get_product_recommendation: product=%s slots=%s",
        prod,
        sorted(list(slots.keys())),
    )

    _, rec_text = _generate_recommendation_text(prod, slots)
    return rec_text

@tool
def generate_purchase_link(product: str) -> str:
    """
    Generate a direct purchase link for the user to buy the product.
    Use this when the user expresses intent to buy or asks for a quote/link.
    
    Args:
        product: The product name.
    """
    logger.info("AgenticTool.generate_purchase_link: product=%s", product)
    return _purchase_tool(product)
