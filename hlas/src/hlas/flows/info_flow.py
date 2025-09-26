from typing import Dict, Any
import logging

from ..tasks import identify_product_task
from ..vector_store import get_weaviate_client
from ..llm import azure_llm, azure_embeddings
from ..prompt_runner import run_direct_task
from pathlib import Path
import yaml
# Import TargetVectors and Filter for the query
from weaviate.classes.query import TargetVectors, Filter


class InfoFlowHelper:
    """One-turn information handler using RAG after ensuring product context.

    Updates `state` in-place and returns "__done__".
    Expected state fields: session, product, message, reply.
    """

    @staticmethod
    def handle(state: Any, decision: Dict[str, Any], logger: logging.Logger) -> str:
        # Log session flags at entry for debugging
        logger.info("InfoFlow.handle: Starting - fast_path_available=%s, product=%s, message_len=%d", 
                   bool(decision.get("use_follow_up_query") and state.session.get("_fu_query")), 
                   state.product, 
                   len(state.message or ""))

        question = state.message
        use_fast_path = False

        # Check if we can use the fast path for a constructed follow-up query
        if decision.get("use_follow_up_query"):
            fu_q = state.session.get("_fu_query")
            if fu_q:
                # A constructed query exists. Now, ensure we have a product.
                if not state.product:
                    # Try to get product from session if not in state
                    session_product = state.session.get("product")
                    if session_product:
                        state.product = session_product
                        logger.info("InfoFlow.fast_path: product restored from session: %s", state.product)

                # If we have a product now, we can use the fast path
                if state.product:
                    logger.info("InfoFlow.fast_path: using constructed query (len=%d), bypassing product ID.", len(fu_q))
                    question = fu_q
                    use_fast_path = True

        if not use_fast_path:
            # Ensure product
            if not state.product:
                prod = run_direct_task(
                    agent_obj=identify_product_task.agent,
                    agent_key="product_identifier",
                    task_key="identify_product",
                    context_text=f"Message: {state.message}\nSession product: {state.session.get('product')}",
                    logger=logger,
                    label="product_identifier.identify_product",
                ) or {}
                
                logger.info("InfoFlow.identify_product: product=%s, confidence=%s, has_question=%s",
                           prod.get("product"),
                           prod.get("confidence"),
                           bool(prod.get("question")))
                if prod.get("product"):
                    state.product = prod.get("product")
                    state.session["product"] = state.product
                    logger.info("InfoFlow.product_resolved: %s", state.product)
                else:
                    # Persist a hint that InfoFlow asked for product clarification,
                    # and save the original user message to reconstruct query on the next turn
                    state.session["_last_info_prod_q"] = True
                    state.session["_last_info_user_msg"] = state.message
                    logger.info("InfoFlow.product_clarification: Requesting product clarification, saved user message")
                    
                    q = prod.get("question") or "Which product would you like to ask about: Travel, Maid, or Car?"
                    state.reply = q
                    return "__done__"
    
            # Hybrid multi-vector search (BM25 + content_vector + questions_vector), filtered by product

            # Edge-case handling: If the previous turn asked for product clarification
            # and the current user replied with a product name, use the original question
            # (saved from the previous turn) as the retrieval query instead of the product word.
            last_info_prod_q = state.session.get("_last_info_prod_q", False)
            logger.info("InfoFlow.edge_case_check: Checking for product clarification flow, flag=%s", last_info_prod_q)

            if last_info_prod_q:
                # Instead of a rigid set, use the product identifier agent to check if the reply is a product.
                prod_check = run_direct_task(
                    agent_obj=identify_product_task.agent,
                    agent_key="product_identifier",
                    task_key="identify_product",
                    context_text=f"Message: {state.message}",
                    logger=logger,
                    label="product_identifier.identify_product.edge_case_check",
                ) or {}
                identified_product = prod_check.get("product")

                logger.info("InfoFlow.edge_case_check: API output - product=%s, confidence=%s, has_question=%s, keys=%s",
                           identified_product, prod_check.get("confidence"), bool(prod_check.get("question")), list(prod_check.keys()))
                logger.info("InfoFlow.edge_case_check: User reply product detection - identified=%s", identified_product)
                
                if identified_product:
                    # A product was identified. Update the state and proceed with the original question.
                    state.product = identified_product
                    state.session["product"] = identified_product
                    logger.info("InfoFlow.edge_case_check: Product confirmed, retrieving original question")

                    prior_user_q = state.session.get("_last_info_user_msg")

                    if prior_user_q and prior_user_q.strip():
                        logger.info("InfoFlow.edge_case_check: Using original question for search, clearing session flags")
                        question = prior_user_q

                        # Clear flags after use to avoid stale behavior
                        state.session.pop("_last_info_prod_q", None)
                        state.session.pop("_last_info_user_msg", None)
                    else:
                        logger.warning("InfoFlow.edge_case_check: Original question is empty or missing")
                else:
                    logger.info("InfoFlow.edge_case_check: User reply not recognized as product, continuing normal flow")
            else:
                logger.debug("InfoFlow.edge_case_check: No previous product clarification detected")
        
        product = (state.product or "").strip()
        logger.info("InfoFlow.retrieval: Starting search - query='%s', product='%s', query_len=%d", 
                   question[:100], product, len(question))

        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")
        
        # Embed query once and reuse for both named vectors
        emb = None
        try:
            emb = azure_embeddings.embed_query(question)
            logger.info("InfoFlow.embedding: Successfully generated embeddings")
        except Exception as e:
            logger.warning("InfoFlow.embedding: Failed to generate embeddings - %s, falling back to BM25", str(e))

        # Perform hybrid search
        objects = []
        search_method = "unknown"
        
        if emb:
            try:
                logger.info("InfoFlow.search: Executing hybrid search with multi-vector")
                result = collection.query.hybrid(
                    query=question,
                    vector={
                        "content_vector": emb,
                        "questions_vector": emb,
                    },
                    target_vector=TargetVectors.average(["content_vector", "questions_vector"]),
                    filters=Filter.by_property("product_name").equal(product),
                    limit=10,
                    alpha=0.7,
                    return_properties=["content", "product_name", "doc_type", "source_file"],
                )
                objects = getattr(result, "objects", []) or []
                search_method = "hybrid"
                logger.info("InfoFlow.search: Hybrid search completed - results=%d", len(objects))
            except Exception as e:
                logger.error("InfoFlow.search: Hybrid search failed - %s", str(e))
                objects = []

        if not objects:
            try:
                logger.info("InfoFlow.search: Falling back to BM25 search")
                result = collection.query.bm25(
                    query=question,
                    filters=Filter.by_property("product_name").equal(product),
                    limit=5,
                    return_properties=["content", "product_name", "doc_type", "source_file"],
                )
                objects = getattr(result, "objects", []) or []
                search_method = "bm25"
                logger.info("InfoFlow.search: BM25 search completed - results=%d", len(objects))
            except Exception as e:
                logger.error("InfoFlow.search: BM25 search failed - %s", str(e))
                objects = []
                search_method = "failed"

        # Synthesize an answer from retrieved chunks
        if objects:
            logger.info("InfoFlow.synthesis: Processing %d chunks using %s search", len(objects), search_method)
            
            # Log chunk details for debugging
            total_content_length = 0
            doc_types = {}
            sources = set()
            
            for i, obj in enumerate(objects):
                content = obj.properties.get('content', '') or ''
                doc_type = obj.properties.get('doc_type', '') or ''
                source = obj.properties.get('source_file', '') or ''
                
                total_content_length += len(content)
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                if source:
                    sources.add(source)
                
                logger.debug("InfoFlow.chunk[%d]: length=%d, type=%s, source=%s", i, len(content), doc_type, source)
            
            logger.info("InfoFlow.synthesis: Total content=%d chars, doc_types=%s, sources=%d", 
                       total_content_length, dict(doc_types), len(sources))
            
            context_str = "\n---\n".join([
                f"Source (Type: {obj.properties.get('doc_type','')}): {obj.properties.get('content','')}" for obj in objects
            ])

            # Load product-specific IR response templates
            ir_templates = {}
            try:
                base_dir = Path(__file__).resolve().parent.parent
                with open(base_dir / "config" / "ir_response.yaml", "r", encoding="utf-8") as rf:
                    ir_templates = yaml.safe_load(rf) or {}
                logger.debug("InfoFlow.templates: Loaded templates for products: %s", list(ir_templates.keys()))
            except Exception as e:
                logger.warning("InfoFlow.templates: Failed to load templates - %s", str(e))

            tpl = ir_templates.get(product.lower(), {}) if product else {}
            sys_t = tpl.get("system") or (
                "You are an insurance information responder. Answer using only the provided context."
            )
            usr_t = (tpl.get("user") or "Question: {question}\n\n[Context]\n{context}").format(
                question=question,
                context=context_str,
            )

            logger.info("InfoFlow.llm: Calling LLM with system_template_len=%d, user_template_len=%d", 
                       len(sys_t), len(usr_t))
            
            try:
                txt = azure_llm.call(messages=[
                    {"role": "system", "content": sys_t},
                    {"role": "user", "content": usr_t},
                ])
                answer_text = str(txt).strip()
                logger.info("InfoFlow.llm: Generated response length=%d", len(answer_text))
            except Exception as e:
                logger.error("InfoFlow.llm: LLM call failed - %s", str(e))
                answer_text = ""

            state.reply = answer_text or "I couldn't find precise details. Could you clarify your question?"
            
            # Attach sources
            source_files = [obj.properties.get("source_file", "") for obj in objects]
            state.sources = "\n".join([s for s in source_files if s])
            
            logger.info("InfoFlow.complete: Success with %d sources", len([s for s in source_files if s]))
            return "__done__"

        # If retrieval failed or empty, ask for clarification
        logger.info("InfoFlow.complete: No results found, requesting clarification")
        state.reply = (
            f"I couldn't find that in our {product} documents. Could you specify a bit more so I can search precisely?"
        )
        state.sources = ""
        return "__done__"