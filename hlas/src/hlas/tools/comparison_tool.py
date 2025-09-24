from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
from .benefits_tool import benefits_tool

class ComparisonToolInput(BaseModel):
    """Input for ComparisonTool."""
    products: List[str] = Field(..., description="A list of insurance products to compare.")

class ComparisonTool(BaseTool):
    name: str = "Comparison Tool"
    description: str = "Compares the benefits of up to three insurance products."
    args_schema: Type[BaseModel] = ComparisonToolInput

    def _run(self, products: List[str]) -> str:
        """Use the tool."""
        if len(products) > 3:
            return "Error: You can compare a maximum of 3 products."

        comparison_result = ""
        for product in products:
            # Assuming product is in the format "Product Name Tier"
            parts = product.split()
            product_name = parts[0]
            tier = parts[1] if len(parts) > 1 else None
            benefits = benefits_tool.run(product=product_name, tier=tier)
            comparison_result += f"## {product}\n{benefits}\n\n"

        return comparison_result

comparison_tool = ComparisonTool()