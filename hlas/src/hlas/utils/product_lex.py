import re
from typing import Optional, Dict, List, Tuple

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

# Weighted, brand-aware aliases compiled at import time.
# Each entry: product -> list of (compiled_regex, weight, is_brand)
_ALIASES: Dict[str, List[Tuple[re.Pattern, int, bool]]] = {p: [] for p in CANONICALS}

_WS = r"[-\s]*"  # treat hyphen and whitespace similarly between brand tokens
_PROTECT_NUM = r"3(?:20|60)"  # allow both 320 and 360 variants seen in docs
_PROTECT_SUFFIX = rf"(?:{_WS}(?:plus|pro|\+))?"  # optional plus/pro/+ suffix
_PROTECT_BASE = rf"protect{_WS}{_PROTECT_NUM}{_PROTECT_SUFFIX}"


def _add(patterns: Dict[str, List[Tuple[re.Pattern, int, bool]]], product: str, expr: str, weight: int,
         is_brand: bool = False) -> None:
    patterns[product].append((re.compile(expr, re.IGNORECASE), weight, is_brand))


def _brand_patterns() -> None:
    brand_kw = {
        "Travel": ["travel"],
        "Car": ["car"],
        "Home": ["home"],
        "Maid": ["maid"],
        # Personal Accident brand is "Family Protect360"
        "PersonalAccident": ["family"],
        "Early": ["early"],
        "Fraud": ["fraud"],
        "Hospital": ["hospital"],
    }
    for product, kws in brand_kw.items():
        for kw in kws:
            kw_esc = re.escape(kw)
            # e.g., "travel protect360", "protect360 travel"
            _add(_ALIASES, product, rf"\b{kw_esc}{_WS}{_PROTECT_BASE}\b", 4, True)
            _add(_ALIASES, product, rf"\b{_PROTECT_BASE}{_WS}{kw_esc}\b", 4, True)
            # Also accept brand without the 320/360 number: e.g., "travel protect", "protect travel"
            _add(_ALIASES, product, rf"\b{kw_esc}{_WS}protect{_PROTECT_SUFFIX}\b", 3, True)
            _add(_ALIASES, product, rf"\bprotect{_PROTECT_SUFFIX}{_WS}{kw_esc}\b", 3, True)


def _synonym_patterns() -> None:
    # Travel
    _add(_ALIASES, "Travel", r"\b(?:travel|trip|vacation)\s+(?:insurance|plan(?:s)?|polic(?:y|ies))\b", 2)
    _add(_ALIASES, "Travel", r"\boverseas[-\s]*travel\s+insurance\b", 2)
    _add(_ALIASES, "Travel", r"\btravel\s+plan(?:s)?\b", 2)

    # Maid (Foreign Domestic Worker)
    _add(_ALIASES, "Maid", r"\bmaid\s+insurance\b", 2)
    _add(_ALIASES, "Maid", r"\bdomestic[-\s]*(?:helper|worker)\b", 2)
    _add(_ALIASES, "Maid", r"\bforeign[-\s]*domestic[-\s]*worker\b", 2)
    _add(_ALIASES, "Maid", r"\b(?:fdw|fww|mdw)\b", 2)

    # Car / Motor
    _add(_ALIASES, "Car", r"\b(?:car|motor|auto|automobile)\s+insurance\b", 2)

    # Personal Accident (Family Protect360 branding)
    _add(
        _ALIASES,
        "PersonalAccident",
        r"\bpersonal{0}accident(?:\s+(?:insurance|plan(?:s)?|polic(?:y|ies)|cover(?:age)?))?\b".format(_WS),
        2,
    )
    _add(_ALIASES, "PersonalAccident", r"\bpa\s+(?:insurance|plan(?:s)?|polic(?:y|ies)|cover(?:age)?)\b", 2)
    _add(_ALIASES, "PersonalAccident", r"\bpersonal\s+injury\s+insurance\b", 2)

    # Home / Property
    _add(_ALIASES, "Home", r"\bhome\s+insurance\b", 2)
    _add(_ALIASES, "Home", r"\bhome{0}contents\s+insurance\b".format(_WS), 2)
    _add(_ALIASES, "Home", r"\bhousehold{0}contents\s+insurance\b".format(_WS), 2)
    _add(_ALIASES, "Home", r"\bhouse\s+insurance\b", 2)
    _add(_ALIASES, "Home", r"\bproperty\s+insurance\b", 1)

    # Early Critical Illness
    _add(_ALIASES, "Early", r"\bearly\s+critical\s+illness\b", 2)
    # Accept bare "critical illness" and common variants (plan/policy/cover/insurance)
    _add(
        _ALIASES,
        "Early",
        r"\b(?:critical\s+illness|ci)(?:\s+(?:insurance|plan(?:s)?|polic(?:y|ies)|cover(?:age)?))?\b",
        2,
    )

    # Fraud / Scam
    _add(_ALIASES, "Fraud", r"\bfraud\s+insurance\b", 2)
    _add(_ALIASES, "Fraud", r"\bscam\s+insurance\b", 2)

    # Hospital / Hospital Cash
    _add(_ALIASES, "Hospital", r"\bhospital\s+insurance\b", 2)
    _add(_ALIASES, "Hospital", r"\bhospital{0}cash(?:\s+(?:plan|insurance))?\b".format(_WS), 2)
    _add(_ALIASES, "Hospital", r"\bhospitali[sz]ation\s+insurance\b", 2)


_brand_patterns()
_synonym_patterns()


def _bare_and_about_patterns() -> None:
    """Add low-weight bare product tokens and 'about/for/regarding <product>' forms.

    These broaden recall so the LLM identifier can decide, without overpowering brand/specific cues.
    """
    # Helper to add both bare and about/for/regarding forms
    def add_token(product: str, token: str, bare_w: int = 1, about_w: int = 2) -> None:
        tok = re.escape(token)
        _add(_ALIASES, product, rf"\b{tok}\b", bare_w)
        _add(_ALIASES, product, rf"\b(?:about|regarding|for)\s+{tok}\b", about_w)

    # Travel
    add_token("Travel", "travel")
    # Maid
    add_token("Maid", "maid")
    add_token("Maid", "helper")
    # Car
    add_token("Car", "car")
    # Personal Accident
    add_token("PersonalAccident", "personal accident")
    add_token("PersonalAccident", "pa")
    # Home
    add_token("Home", "home")
    # Early
    add_token("Early", "early")
    add_token("Early", "critical illness")
    add_token("Early", "ci")
    # Fraud
    add_token("Fraud", "fraud")
    add_token("Fraud", "scam")
    # Hospital
    add_token("Hospital", "hospital")
    add_token("Hospital", "hospital cash")


_bare_and_about_patterns()


def lexical_product_hint(text: str) -> Optional[str]:
    """Return a single best product hint using weighted matching.

    Backward-compatible wrapper around lexical_product_candidates: returns the top
    positive candidate when sufficiently ahead; otherwise None.
    """
    if not text:
        return None
    cands = lexical_product_candidates(text, max_candidates=3)
    if not cands:
        return None
    top = cands[0]
    # Only return positive (or mixed) polarity with sufficient lead
    if top.get("polarity") == "negated":
        return None
    if len(cands) == 1:
        return top.get("product")
    lead = (top.get("score") or 0) - (cands[1].get("score") or 0)
    if lead >= 0.5:
        return top.get("product")
    return None


def lexical_product_candidates(text: str, max_candidates: int = 3) -> List[Dict[str, object]]:
    """Return ranked lexical candidates with scores; no negation blocking.

    Each candidate: { product: str, score: float, polarity: "positive", reasons: [str] }

    Scoring:
    - Base: sum of alias weights matched
    - +1.0 if positive intent phrase near a match (optional minor boost)
    - No negation penalties; leave switch decision to LLM
    - Brand hits implicitly boost via higher weights
    """
    if not text:
        return []

    lowered = text.lower()
    # Positive intent cues: allow 'ask/asking (about/for)'
    pos_kw = re.compile(
        r"\b(?:just\s+)?(?:want|need|looking\s+for|get|buy|quote|plan|ask(?:ing)?(?:\s+(?:about|for))?)\b",
        re.IGNORECASE,
    )

    # Track per-product stats and local match windows
    scores: Dict[str, float] = {p: 0.0 for p in CANONICALS}
    pos_hits: Dict[str, int] = {p: 0 for p in CANONICALS}
    reasons: Dict[str, List[str]] = {p: [] for p in CANONICALS}

    # Scan patterns and evaluate local sentiment near each match
    for product, patterns in _ALIASES.items():
        for rgx, weight, is_brand in patterns:
            for m in rgx.finditer(lowered):
                scores[product] += float(weight)
                span_start, span_end = m.span()
                window_start = max(0, span_start - 40)
                window_end = min(len(lowered), span_end + 40)
                window_text = lowered[window_start:window_end]
                # Detect positive desire in window
                if pos_kw.search(window_text):
                    pos_hits[product] += 1
                    scores[product] += 1.0
                    reasons[product].append("positive_intent_near_match")
                # No negation handling (LLM decides later)

    # Build candidates list
    raw = [
        {
            "product": p,
            "score": round(scores[p], 2),
            "polarity": "positive",
            "reasons": reasons[p],
        }
        for p in CANONICALS
        if scores[p] > 0
    ]

    # Sort by score desc
    raw.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return raw[:max_candidates]