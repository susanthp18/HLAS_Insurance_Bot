from __future__ import annotations

from typing import Optional

from ..config import _load_purchase_links
from ..utils.slots import _normalize_product_key

def _purchase_tool(product: Optional[str]) -> str:
    """Purchase tool: returns purchase link or friendly fallback."""

    prod = _normalize_product_key(product)
    if not prod:
        return (
            "Which product would you like to buy? Available options: Travel, Maid, Car, Personal Accident, "
            "Home, Fraud, Early Critical Illness, Hospital."
        )

    links = _load_purchase_links()
    link = links.get(prod)
    friendly_names = {
        "travel": "Travel",
        "maid": "Maid",
        "car": "Car",
        "personalaccident": "Personal Accident",
        "home": "Home",
        "early": "Early Critical Illness",
        "fraud": "Fraud Protect360",
        "hospital": "Hospital Protect360",
    }
    friendly = friendly_names.get(prod, product or "this")
    if link:
        return f"Great! You can purchase your {friendly} insurance plan securely here: {link}"
    return (
        f"I don't have a direct purchase link for the {friendly} plan right now. "
        "Please let me know if you'd like me to connect you with a specialist."
    )
