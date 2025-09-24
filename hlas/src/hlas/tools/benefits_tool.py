from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ..vector_store import get_weaviate_client
from weaviate.classes.query import Filter

class BenefitsToolInput(BaseModel):
    """Input for the Benefits Tool."""
    product: str = Field(..., description="The insurance product to retrieve benefits for.")
    tier: Optional[str] = Field(None, description="The specific tier of the product (e.g., 'Classic', 'Plus').")

class BenefitsTool(BaseTool):
    name: str = "Product Benefits Tool"
    description: str = "Retrieves all benefits for a specific insurance product and optional tier."
    args_schema: Type[BaseModel] = BenefitsToolInput

    def _run(self, product: str, tier: Optional[str] = None) -> str:
        """
        Retrieves the benefits for a given product and optional tier.
        """
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")

        # Use schema's property name 'product_name' instead of 'product'
        filters = Filter.by_property("product_name").equal(product)
        # Only benefits chunks per schema (exclude faq/policy)
        filters = Filter.all_of([filters, Filter.by_property("doc_type").equal("benefits")])

        response = collection.query.fetch_objects(
            filters=filters,
            limit=500,
            return_properties=["content", "product_name", "doc_type", "source_file"],
        )
        objects = getattr(response, "objects", []) or []
        return "\n".join([obj.properties.get("content", "") for obj in objects])

benefits_tool = BenefitsTool()