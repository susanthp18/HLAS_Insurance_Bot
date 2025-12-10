from __future__ import annotations

import logging
import asyncio
from typing import Tuple, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage
from ..config import _router_model
from ..infrastructure import get_weaviate_client, get_embeddings
from weaviate.classes.query import TargetVectors, Filter

logger = logging.getLogger(__name__)

# 1. Query Refinement Prompt
QUERY_GEN_PROMPT = """You are a policy compliance assistant.
Formulate a search query for the insurance knowledge base to check if the following user message violates any policies (sanctions, illegal acts, exclusions).

Context:
{history}

Message: "{message}"

Output:
A specific search query to retrieve policy documents relevant to this message.
"""

# 2. Validation Prompt
VALIDATION_PROMPT = """You are a Policy Compliance Officer.

Retrieved Policy Documents:
{context}

User Message: "{message}"

Task:
Based ONLY on the retrieved policy documents above, determine if the user's request is EXPLICITLY PROHIBITED.

Rules:
1. The retrieved documents are your ONLY source of truth.
2. Return VIOLATION only if the documents EXPLICITLY and CLEARLY prohibit what the user is asking.
3. General policy requirements or conditions (e.g., "trip must start from Singapore") are NOT violations - they are just how the policy works.
4. If the documents don't clearly prohibit the request, return CLEAN.
5. When in doubt, return CLEAN.

Output (one line only):
- "VIOLATION: <specific prohibition from the documents>" - ONLY if 100% confident
- "CLEAN" - for all other cases
"""

def _get_embeddings():
    """Get embeddings from local infrastructure (thread-safe singleton)."""
    return get_embeddings()

async def _rag_policy_check(query: str, message: str, product: Optional[str] = None) -> Tuple[bool, str]:
    """Perform the actual RAG check against Weaviate."""
    try:
        logger.debug(
            "PolicyRAG.start: product=%s query='%s'",
            product,
            (query or "").replace("\n", " ")[:160],
        )

        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")
        embeddings = _get_embeddings()
        
        if not embeddings:
            logger.error("Policy Check: Embeddings not initialized")
            return False, ""
            
        emb = embeddings.embed_query(query)
        
        # Search options
        filters = None
        if product:
             filters = Filter.by_property("product_name").equal(product.lower())
             
        # Search for policy/exclusion documents
        result = collection.query.hybrid(
            query=query,
            vector={
                "content_vector": emb,
                "questions_vector": emb,
            },
            target_vector=TargetVectors.average(["content_vector", "questions_vector"]),
            filters=filters,
            limit=3,
            alpha=0.7,
            return_properties=["content", "product_name"]
        )
        
        objects = getattr(result, "objects", []) or []
        logger.debug(
            "PolicyRAG.query_done: product=%s hits=%d",
            product,
            len(objects),
        )
        if not objects:
            return False, ""
            
        context_str = "\n---\n".join(
            [str(obj.properties.get("content", "") or "") for obj in objects]
        )
        
        # Final LLM decision based on retrieved docs
        validation_res = await _router_model.ainvoke([
            SystemMessage(content=VALIDATION_PROMPT.format(context=context_str, message=message)),
        ])
        
        content = str(validation_res.content).strip()
        
        if content.upper().startswith("VIOLATION"):
            reason = content.split(":", 1)[1].strip() if ":" in content else "Policy Restriction"
            logger.info(
                "PolicyRAG.violation: product=%s reason='%s'",
                product,
                reason[:200],
            )
            rejection = (
                f"I apologize, but I cannot assist with this request because our policy restricts coverage regarding {reason}. "
                "However, I'd be happy to help you with other destinations or insurance needs! Is there anything else I can do for you?"
            )
            return True, rejection
            
    except Exception as e:
        logger.error("Policy RAG check failed: %s", e)
        
    return False, ""

async def check_policy(message: str, history_str: str = "", product: Optional[str] = None) -> Tuple[bool, str]:
    """
    Deep Policy Checker for ALL messages:
    1. Generate Search Query based on context + message.
    2. If Query != SKIP, perform RAG check against Weaviate.
    3. If Violation found, return True + Message.
    """
    if not message:
        return False, ""

    msg_preview = (message or "").replace("\n", " ")[:160]
    logger.info(
        "PolicyCheck.start: product=%s msg='%s' history_len=%d",
        product,
        msg_preview,
        len(history_str or ""),
    )

    try:
        # Stage 1: Generate Query
        query_res = await _router_model.ainvoke([
            SystemMessage(content=QUERY_GEN_PROMPT.format(history=history_str, message=message))
        ])
        search_query = str(query_res.content).strip()
        logger.debug(
            "PolicyCheck.query_generated: product=%s query='%s'",
            product,
            search_query.replace("\n", " ")[:160],
        )
        
        # Stage 2: RAG Check using the generated query
        is_violation, reply = await _rag_policy_check(search_query, message, product)
        logger.info(
            "PolicyCheck.completed: product=%s violation=%s",
            product,
            is_violation,
        )
        return is_violation, reply
            
    except Exception as e:
        logger.warning("Policy validator failed: %s", e)
        return False, ""
        
    return False, ""


