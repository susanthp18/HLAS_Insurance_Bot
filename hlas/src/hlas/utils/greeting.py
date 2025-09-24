"""Utility helpers for generating user-facing greetings."""

from datetime import datetime

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover - fallback for older Python versions
    ZoneInfo = None


def get_time_based_greeting() -> str:
    """Return the standard time-aware greeting used by the assistant."""

    salutation = "Hello"

    try:
        now_sg = datetime.now(ZoneInfo("Asia/Singapore")) if ZoneInfo else datetime.utcnow()
        hour = now_sg.hour

        if hour < 12:
            salutation = "Good morning"
        elif hour < 18:
            salutation = "Good afternoon"
        else:
            salutation = "Good evening"
    except Exception:
        # Fall back to generic salutation if timezone lookup fails
        salutation = "Hello"

    return (
        f"{salutation}! I'm HLAS Assistant. I can help you with insurance plans, questions, "
        "comparisons, and summaries."
    )


