"""
Session Idle Monitor for Agentic Chatbot
=========================================

Background task that monitors WhatsApp sessions for inactivity and sends
a farewell message before the session times out.

Environment Variables:
    ENABLE_IDLE_FAREWELL: Enable/disable the monitor (default: false)
    IDLE_FAREWELL_SECONDS: Seconds of inactivity before sending farewell (default: 0 = disabled)
    IDLE_FAREWELL_MESSAGE: Custom farewell message
    IDLE_MONITOR_POLL_SECONDS: How often to scan for idle sessions (default: 60)
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

try:
    import orjson
    def _loads(s): return orjson.loads(s)
    def _dumps(obj): return orjson.dumps(obj, default=str).decode("utf-8")
except ImportError:
    import json
    def _loads(s): return json.loads(s)
    def _dumps(obj): return json.dumps(obj, default=str)

try:
    from zoneinfo import ZoneInfo
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:
    SGT_TZ = timezone(timedelta(hours=8))

from .redis_utils import get_redis, RedisLock, session_lock_key
from .metrics import AGENTIC_MESSAGES_TOTAL

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "false") -> bool:
    val = os.getenv(name, default)
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


# Configuration
ENABLE_IDLE_FAREWELL = _env_flag("ENABLE_IDLE_FAREWELL", "false")
IDLE_FAREWELL_SECONDS = int(os.getenv("IDLE_FAREWELL_SECONDS", "0") or "0")
IDLE_FAREWELL_MESSAGE = os.getenv(
    "IDLE_FAREWELL_MESSAGE",
    "It looks like you haven't sent any new questions for a while, so I'll close this chat now. "
    "If you need anything else, just message me again. Have a great day!",
)
IDLE_MONITOR_POLL_SECONDS = int(os.getenv("IDLE_MONITOR_POLL_SECONDS", "60") or "60")

# WhatsApp handler reference (set by main.py on startup)
_whatsapp_handler = None


def set_whatsapp_handler(handler) -> None:
    """Set the WhatsApp handler reference for sending farewell messages."""
    global _whatsapp_handler
    _whatsapp_handler = handler
    logger.info("IdleMonitor: WhatsApp handler registered")


async def _send_farewell(phone: str) -> bool:
    """Send farewell message via WhatsApp."""
    if _whatsapp_handler is None:
        logger.warning("IdleMonitor: No WhatsApp handler registered; cannot send farewell")
        return False
    
    try:
        await _whatsapp_handler._send_message_async(phone, IDLE_FAREWELL_MESSAGE)
        AGENTIC_MESSAGES_TOTAL.labels(result="idle_farewell", product="none").inc()
        return True
    except Exception as e:
        logger.error("IdleMonitor: Failed to send farewell to %s: %s", phone, e)
        return False


async def _process_idle_session(session_id: str, session: Dict[str, Any], now: datetime) -> None:
    """
    Process an idle WhatsApp session under a Redis lock.
    Sends farewell message and marks session.
    """
    if not session_id.startswith("whatsapp_"):
        return

    lock_key = session_lock_key(session_id)
    redis = get_redis()
    redis_key = f"agentic:session:{session_id}"

    try:
        with RedisLock(lock_key, ttl_seconds=15.0, wait_timeout=0.2):
            raw = redis.get(redis_key)
            if not raw:
                return

            try:
                current = _loads(raw)
            except Exception:
                logger.warning("IdleMonitor: Failed to decode session JSON for %s", session_id)
                return

            # Skip if live agent is active
            las = current.get("live_agent_status")
            is_live = False
            if isinstance(las, str):
                is_live = las.strip().lower() in ("on", "true", "yes", "1")
            else:
                is_live = bool(las)
            if is_live:
                logger.debug("IdleMonitor: Session %s is in live_agent state; skipping", session_id)
                return

            # Skip if farewell already sent
            if current.get("idle_farewell_sent"):
                return

            # Re-check idle duration
            last_active = current.get("last_active")
            if isinstance(last_active, str):
                try:
                    last_active = datetime.fromisoformat(last_active)
                except Exception:
                    last_active = None
            if isinstance(last_active, datetime):
                try:
                    if last_active.tzinfo is None:
                        last_active = last_active.replace(tzinfo=SGT_TZ)
                    else:
                        last_active = last_active.astimezone(SGT_TZ)
                except Exception:
                    last_active = None

            if not last_active:
                return

            delta = now - last_active
            if delta < timedelta(seconds=IDLE_FAREWELL_SECONDS):
                return

            # Send farewell
            phone = session_id[len("whatsapp_"):]
            if not await _send_farewell(phone):
                return

            # Mark session
            current["idle_farewell_sent"] = True
            current["last_idle_farewell_ts"] = now.isoformat()

            # Clear transient flags
            for k in (
                "recommendation_status",
                "comparison_status", 
                "summary_status",
                "pending_slot",
                "last_question",
            ):
                current.pop(k, None)

            # Persist
            try:
                ttl = redis.ttl(redis_key)
            except Exception:
                ttl = None

            try:
                payload = _dumps(current)
                if ttl is not None and ttl > 0:
                    redis.set(redis_key, payload, ex=ttl)
                else:
                    redis.set(redis_key, payload)
                logger.info("IdleMonitor: Sent farewell and updated session %s", session_id)
            except Exception as e:
                logger.error("IdleMonitor: Failed to persist session %s: %s", session_id, e)

    except TimeoutError:
        logger.debug("IdleMonitor: Could not acquire lock for %s; skipping", session_id)
    except Exception as e:
        logger.error("IdleMonitor: Error processing session %s: %s", session_id, e)


async def run_idle_farewell_scan_once() -> None:
    """Single scan pass over Redis sessions to detect idle WhatsApp conversations."""
    if not ENABLE_IDLE_FAREWELL or IDLE_FAREWELL_SECONDS <= 0:
        return

    try:
        redis = get_redis()
    except Exception as e:
        logger.error("IdleMonitor: Failed to get Redis client: %s", e)
        return

    now = datetime.now(SGT_TZ)
    cursor = 0

    try:
        while True:
            cursor, keys = redis.scan(cursor=cursor, match="agentic:session:*", count=100)
            
            for key in keys:
                if not isinstance(key, str):
                    continue
                if not key.startswith("agentic:session:"):
                    continue
                    
                session_id = key.split("agentic:session:", 1)[-1]
                if not session_id.startswith("whatsapp_"):
                    continue

                raw = redis.get(key)
                if not raw:
                    continue

                try:
                    session = _loads(raw)
                except Exception:
                    continue

                # Fast pre-filters
                if session.get("idle_farewell_sent"):
                    continue

                las = session.get("live_agent_status")
                is_live = False
                if isinstance(las, str):
                    is_live = las.strip().lower() in ("on", "true", "yes", "1")
                else:
                    is_live = bool(las)
                if is_live:
                    continue

                last_active = session.get("last_active")
                if isinstance(last_active, str):
                    try:
                        last_active = datetime.fromisoformat(last_active)
                    except Exception:
                        last_active = None
                if isinstance(last_active, datetime):
                    try:
                        if last_active.tzinfo is None:
                            last_active = last_active.replace(tzinfo=SGT_TZ)
                        else:
                            last_active = last_active.astimezone(SGT_TZ)
                    except Exception:
                        last_active = None

                if not last_active:
                    continue

                delta = now - last_active
                if delta < timedelta(seconds=IDLE_FAREWELL_SECONDS):
                    continue

                # Process under lock
                await _process_idle_session(session_id, session, now)

            if cursor == 0 or cursor == "0":
                break
                
    except Exception as e:
        logger.error("IdleMonitor: Error scanning sessions: %s", e)


async def idle_monitor_loop() -> None:
    """
    Background loop to periodically scan for idle sessions.
    Start this in FastAPI lifespan.
    """
    logger.info(
        "IdleMonitor: Starting (enabled=%s, idle_seconds=%d, poll_interval=%ds)",
        ENABLE_IDLE_FAREWELL,
        IDLE_FAREWELL_SECONDS,
        IDLE_MONITOR_POLL_SECONDS,
    )
    
    if not ENABLE_IDLE_FAREWELL or IDLE_FAREWELL_SECONDS <= 0:
        logger.info("IdleMonitor: Disabled; exiting loop")
        return
    
    try:
        while True:
            try:
                await run_idle_farewell_scan_once()
            except Exception as e:
                logger.error("IdleMonitor: Error in scan loop: %s", e)
            await asyncio.sleep(IDLE_MONITOR_POLL_SECONDS)
    except asyncio.CancelledError:
        logger.info("IdleMonitor: Loop cancelled; shutting down")
        raise


__all__ = [
    "idle_monitor_loop",
    "run_idle_farewell_scan_once",
    "set_whatsapp_handler",
    "ENABLE_IDLE_FAREWELL",
    "IDLE_FAREWELL_SECONDS",
    "IDLE_FAREWELL_MESSAGE",
]
