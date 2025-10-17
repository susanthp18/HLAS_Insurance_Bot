import re
from typing import Optional

# Canonical product names used across the system
CANONICALS = [
    "Travel",
    "Maid",
    "Car",
    "PersonalAccident",
    "Home",
    "Early",
    "Fraud",
    "Hospital",
]

# Word-boundary and brand-alias patterns (case-insensitive)
# Keep patterns strict to avoid generic false-positives; confirmation is done via product_identifier.
PRODUCT_PATTERNS = {
    "Travel": [
        r"\btravel\s+insurance\b",
        r"\btravel\s*protect\s*360\b",
        r"\btravel\b",
    ],
    "Maid": [
        r"\bmaid\s+insurance\b",
        r"\bmaid\s*protect\s*360\b",
        r"\bdomestic\s+(?:helper|worker)\b",
        r"\bhelper\b",
        r"\bfdw\b",
        r"\bfww\b",
        r"\bmaid\b",
    ],
    "Car": [
        r"\b(?:car|motor)\s+insurance\b",
        r"\bcar\s*protect\s*360\b",
    ],
    "PersonalAccident": [
        r"\bpersonal\s+accident\b",
        r"\bpa\s+insurance\b",
        r"\bfamily\s*protect\s*360\b",
    ],
    "Home": [
        r"\bhome\s+insurance\b",
        r"\bhome\s*protect\s*360\b",
        r"\bhome\s*protect360\b",
    ],
    "Early": [
        r"\bearly\s*protect\s*360\b",
        r"\b(?:critical\s+illness|ci)\s+(?:insurance)?\b",
    ],
    "Fraud": [
        r"\bfraud\s*protect\s*360\s*plus\b",
        r"\bfraud\s*protect\s*360\b",
    ],
    "Hospital": [
        r"\bhospital\s+insurance\b",
        r"\bhospital\s*protect\s*360\b",
        r"\bhospital\s*protect360\b",
    ],
}

_COMPILED = {
    k: [re.compile(pat, re.IGNORECASE) for pat in pats] for k, pats in PRODUCT_PATTERNS.items()
}

def lexical_product_hint(text: str) -> Optional[str]:
    """Return a canonical product name if exactly one product alias is matched in text.
    If multiple different products match, return None to avoid ambiguous switches.
    """
    if not text:
        return None
    hits = set()
    for product, patterns in _COMPILED.items():
        for rgx in patterns:
            if rgx.search(text):
                hits.add(product)
                break
    if len(hits) == 1:
        return next(iter(hits))
    return None