from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class SummaryToolInput(BaseModel):
    """Input for SummaryTool."""
    product: str = Field(..., description="The insurance product to summarize.")
    tier: str = Field(None, description="The specific tier of the product.")

class SummaryTool(BaseTool):
    name: str = "Summary Tool"
    description: str = "Summarizes a specified insurance product and tier."
    args_schema: Type[BaseModel] = SummaryToolInput

    def _run(self, product: str, tier: str = None) -> str:
        """Use the tool."""
        # This is a placeholder. A real implementation would call a summarization model.
        return f"This is a summary for {product} {tier if tier else ''}."

summary_tool = SummaryTool()