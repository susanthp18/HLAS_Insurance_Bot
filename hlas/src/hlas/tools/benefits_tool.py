from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type, Dict, Any
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BenefitsToolInput(BaseModel):
    """Input for the Benefits Tool."""
    product: str = Field(..., description="The insurance product to retrieve benefits for.")
    tier: Optional[str] = Field(None, description="The specific tier of the product (e.g., 'Classic', 'Plus').")

class BenefitsTool(BaseTool):
    name: str = "Product Benefits Tool"
    description: str = "Retrieves raw benefits text for a product from benefits_raw.json (no RAG)."
    args_schema: Type[BaseModel] = BenefitsToolInput

    _cache: Dict[str, Any] | None = None

    @staticmethod
    def _normalize_product_key(name: str) -> str:
        base = (name or "").lower().strip()
        # remove common words and separators
        if base.endswith("_benefits"):
            base = base.replace("_benefits", "")
        if base.endswith("-benefits"):
            base = base.replace("-benefits", "")
        
        # Normalize alphanumeric only
        clean = "".join(ch for ch in base if ch.isalnum())
        
        # Explicit alias mapping to match benefits_raw.json keys
        aliases = {
            "pa": "personalaccident",
            "personal": "personalaccident",
            "accident": "personalaccident",
            "familyprotect360": "personalaccident",
            "familyprotect": "personalaccident",
            "family": "personalaccident",
            "travelprotect360": "travel",
            "maidprotect360": "maid",
            "carprotect360": "car",
            "homeprotect360": "home",
            "earlyprotect360": "early",
            "earlyci": "early",
            "ci": "early",
            "fraudprotect360": "fraud",
            "hospitalprotect360": "hospital",
            "hospitalincome": "hospital"
        }
        
        return aliases.get(clean, clean)

    @classmethod
    def _load_cache(cls) -> Dict[str, Any]:
        if isinstance(cls._cache, dict):
            return cls._cache
        # benefits_raw.json resides in hlas/src/hlas/config/
        config_dir = Path(__file__).resolve().parent.parent / "config"
        json_path = config_dir / "benefits_raw.json"
        try:
            text = json_path.read_text(encoding="utf-8")
            cls._cache = json.loads(text) or {}
            logger.info("BenefitsTool: Loaded benefits_raw.json with %d products", len(cls._cache))
        except FileNotFoundError:
            logger.error("BenefitsTool: benefits_raw.json not found at %s", str(json_path))
            cls._cache = {}
        except Exception as e:
            logger.error("BenefitsTool: Failed to read/parse benefits_raw.json - %s", str(e))
            cls._cache = {}
        return cls._cache

    def _run(self, product: str, tier: Optional[str] = None) -> str:
        """Return raw benefits text for the product by concatenating all docs from JSON."""
        data = self._load_cache()
        key = self._normalize_product_key(product)
        entry = data.get(key)
        if not entry:
            logger.warning("BenefitsTool: No benefits found for product='%s' (key=%s)", product, key)
            return ""
        docs = entry.get("docs") or []
        # Join with clear separators to mimic chunk boundaries
        return "\n\n---\n\n".join([str(d or "").strip() for d in docs if str(d or "").strip()])

benefits_tool = BenefitsTool()