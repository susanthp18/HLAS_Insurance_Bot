from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Type, Optional, Any
from ..vector_store import get_weaviate_client
from ..llm import azure_embeddings
from weaviate.classes.query import Filter
import os

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

        # Embed the query once; reuse for multi-vector target
        try:
            embedding = azure_embeddings.embed_query(query)
        except Exception:
            embedding = None

        response = collection.query.hybrid(
            query=query,
            vector=embedding,
            alpha=float(os.environ.get("RAG_ALPHA", 0.7)),
            target_vector="average(['content_vector', 'questions_vector'])",
            limit=limit or 15,
            filters=filters,
            properties=["content", "product_name", "doc_type", "source_file"],
        )

        objects = getattr(response, "objects", []) or []
        if not objects and isinstance(response, dict):
            # Defensive: handle unexpected shapes
            return ""
        return "\n".join([obj.properties.get("content", "") for obj in objects])

retrieval_tool = RAGTool()