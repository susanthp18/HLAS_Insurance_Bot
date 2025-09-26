"""Utility helpers for generating user-facing greetings."""

import random
from datetime import datetime
from zoneinfo import ZoneInfo


def get_time_based_greeting() -> str:
    """Return a random, time-aware, and human-like greeting message."""

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
        salutation = "Hello"

    # Define the 3 templates
    template1 = f"{salutation}! I'm the HLAS Assistant. I can help you get a recommendation, compare insurance plans, or answer any questions you have. What can I help you with today?"
    template2 = f"{salutation}! You've reached the HLAS Assistant. Whether you need to compare plans, get a personalized recommendation, or ask about policy details, I'm here to help. What's on your mind?"
    template3 = f"{salutation}! This is the HLAS Assistant. My purpose is to provide information, comparisons, and recommendations for our insurance products. How can I assist you?"

    # Randomly choose one of the templates
    return random.choice([template1, template2, template3])


