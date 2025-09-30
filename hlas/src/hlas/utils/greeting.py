"""Utility helpers for generating user-facing greetings."""

import random
from datetime import datetime
from zoneinfo import ZoneInfo


def get_time_based_greeting() -> str:
    """Return the standardized greeting for the HLAS Smart Bot."""
    return (
        "Hello! 👋 I’m the HLAS Smart Bot. I’m here to guide you through our insurance products and services, "
        "answer your questions instantly, and make things easier for you. How can I help you today?"
    )


