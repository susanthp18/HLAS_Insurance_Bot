"""Utility helpers for generating user-facing greetings."""

from datetime import datetime
from zoneinfo import ZoneInfo


def get_time_based_greeting() -> str:
    """Return the standard time-aware greeting used by the assistant."""

    salutation = "Hello"

    try:
        now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
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


