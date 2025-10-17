from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Type, Optional, Any
from ..vector_store import get_weaviate_client
from ..llm import azure_embeddings
from weaviate.classes.query import Filter, TargetVectors
import os
import hashlib
import json

class RAGToolInput(BaseModel):
    model_config = ConfigDict(extra='allow')
    query: Any = Field(..., description="User's question to search for")
    product: Any = Field(..., description="Product filter, e.g., Travel or Maid or Car")
    doc_type: Any = Field(default=None, description="Optional doc type filter, e.g., benefits or policy")
    retrieve_all: Optional[bool] = Field(default=False, description="If true, ignore top-k limit")


class RAGTool(BaseTool):
    name: str = "Insurance RAG Tool"
    description: str = "Performs a hybrid search on the insurance knowledge base to find relevant information."
    args_schema: Type[BaseModel] = RAGToolInput

    def _run(self, query: Any, product: Any = None, doc_type: Any = None, retrieve_all: Optional[bool] = False, **kwargs: Any) -> str:
        # Coerce inputs defensively
        if not isinstance(query, str):
            if isinstance(query, dict):
                query = query.get("query") or query.get("description") or ""
            else:
                query = str(query)
        if not isinstance(product, str):
            if isinstance(product, dict):
                product = product.get("product") or product.get("value") or product.get("name") or ""
            else:
                product = "" if product is None else str(product)
        if not isinstance(doc_type, str) and doc_type is not None:
            if isinstance(doc_type, dict):
                doc_type = doc_type.get("doc_type") or doc_type.get("value") or doc_type.get("name")
            else:
                doc_type = str(doc_type)
        if not product:
            return "Product is required. Please specify: Travel, Maid, or Car."
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")

        filters = None
        if product:
            filters = Filter.by_property("product_name").equal(product)
        if doc_type:
            doc_filter = Filter.by_property("doc_type").equal(doc_type)
            filters = Filter.all_of([filters, doc_filter]) if filters is not None else doc_filter

        limit = None if retrieve_all else int(os.environ.get("RAG_TOP_K", 15))

        # Small Redis cache (optional) via existing redis_utils if available
        cache_ttl = int(os.environ.get("RAG_CACHE_TTL", "0") or 0)
        cache_key = None
        cached_value = None
        if cache_ttl > 0:
            try:
                from ..redis_utils import get_redis  # type: ignore
                r = get_redis()
                key_payload = {
                    "q": query,
                    "p": product,
                    "d": doc_type,
                    "l": limit,
                }
                cache_key = "rag:" + hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
                cached_value = r.get(cache_key)
                if cached_value:
                    try:
                        return cached_value.decode("utf-8")
                    except Exception:
                        return str(cached_value)
            except Exception:
                cache_key = None
                cached_value = None

        # Embed the query once; reuse for multi-vector target
        try:
            embedding = azure_embeddings.embed_query(query)
        except Exception:
            embedding = None

        # Strict mode: require embeddings for RAG. If embedding generation failed,
        # do not fall back to BM25; return empty string to indicate no results.
        if embedding is None:
            return ""

        # Build hybrid kwargs similar to InfoFlow: use named vectors and averaged target when we have an embedding
        hybrid_kwargs = {
            "query": query,
            "alpha": float(os.environ.get("RAG_ALPHA", 0.7)),
            "limit": limit or 15,
            "filters": filters,
            "return_properties": ["content", "product_name", "doc_type", "source_file"],
        }

        hybrid_kwargs["vector"] = {
            "content_vector": embedding,
            "questions_vector": embedding,
        }
        hybrid_kwargs["target_vector"] = TargetVectors.average(["content_vector", "questions_vector"])

        response = collection.query.hybrid(**hybrid_kwargs)

        objects = getattr(response, "objects", []) or []
        if not objects and isinstance(response, dict):
            # Defensive: handle unexpected shapes
            return ""
        text = "\n".join([obj.properties.get("content", "") for obj in objects])

        # Fill cache
        if cache_key and cache_ttl > 0:
            try:
                from ..redis_utils import get_redis  # type: ignore
                r = get_redis()
                r.setex(cache_key, cache_ttl, text)
            except Exception:
                pass

        return text

retrieval_tool = RAGTool()