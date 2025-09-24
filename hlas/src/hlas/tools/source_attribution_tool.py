from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import re

class SourceAttributionInput(BaseModel):
    context: str = Field(description="The context from which to extract the source.")

class SourceAttributionTool(BaseTool):
    name: str = "source_attribution_tool"
    description: str = "Extracts the source from the context."
    args_schema: Type[BaseModel] = SourceAttributionInput

    def _run(self, context: str) -> str:
        # A simple regex to find URLs as sources.
        # This can be expanded to be more sophisticated.
        sources = re.findall(r'https?://[\S]+', context)
        if sources:
            return "Sources: " + ", ".join(sources)
        return ""

source_attribution_tool = SourceAttributionTool()