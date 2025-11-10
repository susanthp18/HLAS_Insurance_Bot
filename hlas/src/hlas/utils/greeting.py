"""Utility helpers for generating user-facing greetings."""

import random
from datetime import datetime
from zoneinfo import ZoneInfo


def get_time_based_greeting() -> str:
    """Return the standardized greeting for the BT Smart Bot."""
    return (
        "Hello! 👋 I’m the BT Smart Bot. I can check your policy or claim status, "
        "guide you through our insurance products, answer questions instantly, and make things easier for you. "
        "How can I help you today?"
    )


