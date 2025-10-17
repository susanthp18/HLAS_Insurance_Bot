from crewai.tools import BaseTool
from typing import Type, List, Tuple
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
        for item in products:
            # Support patterns:
            # 1) "Product: Home; Tier: Platinum" (preferred)
            # 2) "Home Platinum" or "Personal Accident Premier"
            product_name, tier = _parse_product_and_tier(item)
            benefits = benefits_tool.run(product=product_name, tier=tier)
            header = f"{product_name}{(' ' + tier) if tier else ''}"
            comparison_result += f"## {header}\n{benefits}\n\n"

        return comparison_result


def _parse_product_and_tier(text: str) -> Tuple[str, str | None]:
    s = (text or "").strip()
    if not s:
        return "", None
    # Pattern 1: key-value form
    if ";" in s and ":" in s:
        try:
            parts = [p.strip() for p in s.split(";")]
            kv = {}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    kv[k.strip().lower()] = v.strip()
            prod = kv.get("product", "")
            tier = kv.get("tier")
            return prod, tier or None
        except Exception:
            pass
    # Pattern 2: split by space, product may be multi-word from a known list
    known_products = [
        "Travel",
        "Maid",
        "Car",
        "Personal Accident",
        "PersonalAccident",
        "Home",
        "Early",
        "Fraud",
    ]
    # Try to match longest known product prefix
    for kp in sorted(known_products, key=len, reverse=True):
        if s.lower().startswith(kp.lower() + " ") or s.lower() == kp.lower():
            rest = s[len(kp):].strip()
            tier = rest if rest else None
            # Normalize Personal Accident
            if kp == "Personal Accident":
                kp = "PersonalAccident"
            return kp, tier or None
    # Fallback: first token as product, second as tier
    parts = s.split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], None

comparison_tool = ComparisonTool()