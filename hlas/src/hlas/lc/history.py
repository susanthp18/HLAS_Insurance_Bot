from __future__ import annotations

"""
Redis-backed message history adapter bridging SessionManager to a LangChain-like
interface. We keep this extremely small and optional.
"""

from typing import List, Dict, Any


def get_last_n_pairs(session: Dict[str, Any], n: int = 3) -> List[Dict[str, str]]:
    """
    Returns up to N recent conversation pairs from the session history (most recent first).
    Each pair is a dict with optional 'user' and 'assistant' keys.
    """
    history = session.get("history", []) or []
    if not isinstance(history, list):
        return []
    tail = list(history[-n:])
    tail.reverse()
    pairs: List[Dict[str, str]] = []
    for turn in tail:
        try:
            u = turn.get("user", "")
            a = turn.get("assistant", "")
            pair: Dict[str, str] = {}
            if u:
                pair["user"] = u
            if a:
                pair["assistant"] = a
            if pair:
                pairs.append(pair)
        except Exception:
            continue
    return pairs



