from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Type, Optional, Any
from datetime import datetime


class ExplanationToolInput(BaseModel):
    # Accept flexible inputs from the agent; ignore extras
    model_config = ConfigDict(extra='allow')
    query: Optional[str] = Field(default=None, description="Optional user query about capabilities.")
    description: Optional[str] = Field(default=None, description="Optional description passed by the agent.")
    context: Optional[str] = Field(default=None, description="Optional context.")


class ExplanationTool(BaseTool):
    name: str = "Explanation Tool"
    description: str = "Explains the chatbot's capabilities in user-friendly terms (no internal details)."
    args_schema: Type[BaseModel] = ExplanationToolInput

    def _run(self, query: Optional[str] = None, description: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> str:
        # Friendly greeting based on time
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 18:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        message = (
            f"{greeting}! I'm HLAS Assistant. I can help you:\n"
            "- Recommend plans\n"
            "- Answer questions\n"
            "- Compare options\n"
            "- Summarize products\n"
        )
        return message


explanation_tool = ExplanationTool()