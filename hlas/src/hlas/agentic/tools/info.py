from __future__ import annotations

import logging
from typing import List, Optional

from weaviate.classes.query import TargetVectors, Filter

# Local infrastructure imports (lazy singleton - no explicit init needed)
from ..infrastructure import get_weaviate_client, get_embeddings, get_response_llm
from ..config import _load_ir_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm

logger = logging.getLogger(__name__)

def _get_models():
    """Get LLM models from local infrastructure (thread-safe singletons)."""
    return get_embeddings(), get_response_llm()

def _info_tool(product: Optional[str], question: str) -> tuple[str, List[str]]:
    """Information tool: RAG over Weaviate using ir_response.yaml templates."""

    raw_product = product
    prod = _normalize_product_key(product)
    if not prod:
        detected = _detect_product_llm(question)
        prod = _normalize_product_key(detected)
    logger.debug(
        "Agentic.info.product_resolve: raw=%s resolved=%s",
        raw_product,
        prod,
    )
    if not prod:
        ask = (
            "Which product would you like to ask about: Travel, Maid, Car, Personal Accident, "
            "Home, Early, Fraud or Hospital?"
        )
        return ask, []

    try:
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")
    except Exception as e:
        logger.error("Agentic.info: failed to initialise Weaviate client - %s", e)
        return (
            "I'm having trouble accessing my knowledge base right now. Please try again later or "
            "ask a simpler question.",
            [],
        )

    embeddings, llm = _get_models()

    try:
        if embeddings:
            emb = embeddings.embed_query(question)
        else:
            logger.error("Agentic.info: Embeddings model not initialized")
            emb = None
    except Exception as e:
        logger.error("Agentic.info: embedding failed - %s", e)
        emb = None

    objects = []
    if emb is not None:
        try:
            result = collection.query.hybrid(
                query=question,
                vector={
                    "content_vector": emb,
                    "questions_vector": emb,
                },
                target_vector=TargetVectors.average(["content_vector", "questions_vector"]),
                filters=Filter.by_property("product_name").equal(prod),
                limit=10,
                alpha=0.7,
                return_properties=["content", "product_name", "doc_type", "source_file"],
            )
            objects = getattr(result, "objects", []) or []
            logger.debug(
                "Agentic.info.weaviate_hits: product=%s hits=%d",
                prod,
                len(objects),
            )
        except Exception as e:
            logger.error("Agentic.info: Weaviate hybrid query failed - %s", e)
            objects = []

    if not objects:
        question_preview = (question or "").replace("\n", " ")[:160]
        logger.info(
            "Agentic.info.no_results: product=%s question='%s'",
            prod,
            question_preview,
        )
        return (
            f"I couldn’t find detailed information for that in our {prod.title()} plans. Could you share a bit more so I can look up the exact coverage?",
            [],
        )

    context_str = "\n---\n".join(
        [str(obj.properties.get("content", "") or "") for obj in objects]
    )
    sources = sorted(
        {
            str(obj.properties.get("source_file", "") or "")
            for obj in objects
            if obj.properties.get("source_file")
        }
    )

    ir_templates = _load_ir_templates()
    tpl = ir_templates.get(prod, {}) if ir_templates else {}
    sys_t = tpl.get("system") or (
        "You are an insurance information responder. Answer using only the provided context."
    )
    usr_t = (tpl.get("user") or "Question: {question}\n\n[Context]\n{context}").format(
        question=question,
        context=context_str,
    )

    try:
        if llm:
            txt = llm.call(
                messages=[
                    {"role": "system", "content": sys_t},
                    {"role": "user", "content": usr_t},
                ]
            )
            answer = str(txt).strip()
        else:
             logger.error("Agentic.info: Response LLM not initialized")
             answer = ""
    except Exception as e:
        logger.error("Agentic.info: response LLM failed - %s", e)
        answer = ""

    answer = answer or "I couldn't find precise details. Could you clarify your question?"

    logger.info(
        "Agentic.info.completed: product=%s answer_len=%d sources=%d",
        prod,
        len(answer),
        len(sources),
    )
    return answer, sources
