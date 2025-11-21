import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import orjson

from ..redis_utils import get_redis, RedisLock, session_lock_key
from ..session import SGT_TZ
from ..metrics import IDLE_FAREWELLS_SENT_TOTAL
from .whatsapp_handler import whatsapp_handler

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "false") -> bool:
    val = os.getenv(name, default)
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


ENABLE_IDLE_FAREWELL = _env_flag("ENABLE_IDLE_FAREWELL", "false")
IDLE_FAREWELL_SECONDS = int(os.getenv("IDLE_FAREWELL_SECONDS", "0") or "0")
IDLE_FAREWELL_MESSAGE = os.getenv(
    "IDLE_FAREWELL_MESSAGE",
    "It looks like you haven’t sent any new questions for a while, so I’ll close this chat now. "
    "If you need anything else, just message me again. Have a great day!",
)


async def _process_idle_session(session_id: str, session: Dict[str, Any], now: datetime) -> None:
    """
    Re-validate an idle WhatsApp session under a Redis lock and, if still eligible,
    send a one-off farewell message and mark the session accordingly.
    """
    # Only handle WhatsApp sessions here; other channels have no push mechanism today.
    if not session_id.startswith("whatsapp_"):
        return

    lock_key = session_lock_key(session_id)
    redis = get_redis()
    redis_key = f"session:{session_id}"

    try:
        # Best-effort lock acquisition; wait briefly, but never block real traffic for long.
        with RedisLock(lock_key, ttl_seconds=15.0, wait_timeout=0.2):
            raw = redis.get(redis_key)
            if not raw:
                return

            try:
                current = orjson.loads(raw)
            except Exception:
                logger.warning("IdleMonitor: Failed to decode session JSON for %s", session_id)
                return

            # Re-check live agent status under the lock
            las = current.get("live_agent_status")
            is_live = False
            if isinstance(las, str):
                is_live = las.strip().lower() in ("on", "true", "yes", "1")
            else:
                is_live = bool(las)
            if is_live:
                logger.debug("IdleMonitor: Session %s is in live_agent state; skipping farewell.", session_id)
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
                # No longer idle enough; a recent interaction likely updated last_active.
                return

            # All checks passed; send farewell
            phone = session_id[len("whatsapp_") :]
            try:
                await whatsapp_handler._send_message_async(phone, IDLE_FAREWELL_MESSAGE)
                IDLE_FAREWELLS_SENT_TOTAL.labels(channel="wa").inc()
            except Exception as e:
                logger.error("IdleMonitor: Failed to send farewell to %s: %s", phone, e)
                return

            # Mark session and optionally clear transient flags so next turn feels fresh
            current["idle_farewell_sent"] = True
            current["last_idle_farewell_ts"] = now.isoformat()

            for k in (
                "recommendation_status",
                "comparison_status",
                "summary_status",
                "fraud_stage",
                "purchase_flow_stage",
                "pending_slot",
                "last_question",
                "_fu_query",
                "_last_info_prod_q",
                "_last_info_user_msg",
                "_last_rec_prod_q",
                "_tentative_product",
                "_skip_extraction_once",
                "_early_existing_cover_notice",
                "__product_switch_confirmed__",
            ):
                if k in current:
                    current.pop(k, None)

            try:
                ttl = redis.ttl(redis_key)
            except Exception:
                ttl = None

            try:
                payload = orjson.dumps(current, default=str).decode("utf-8")
                if ttl is not None and ttl > 0:
                    redis.set(redis_key, payload, ex=ttl)
                else:
                    redis.set(redis_key, payload)
            except Exception as e:
                logger.error("IdleMonitor: Failed to persist updated session %s: %s", session_id, e)
    except TimeoutError:
        # Another worker or request holds the lock; skip quietly.
        logger.debug("IdleMonitor: Could not acquire lock for session %s (busy); skipping.", session_id)
    except Exception as e:
        logger.error("IdleMonitor: Unexpected error while processing session %s: %s", session_id, e)


async def run_idle_farewell_scan_once() -> None:
    """
    Single scan pass over Redis sessions to detect and handle idle WhatsApp conversations.
    """
    if not ENABLE_IDLE_FAREWELL or IDLE_FAREWELL_SECONDS <= 0:
        return

    try:
        redis = get_redis()
    except Exception as e:
        logger.error("IdleMonitor: Failed to get Redis client: %s", e)
        return

    now = datetime.now(SGT_TZ)
    cursor: int | str = 0

    try:
        while True:
            cursor, keys = redis.scan(cursor=cursor, match="session:*", count=100)
            for key in keys:
                # Keys are of the form "session:{session_id}"
                if not isinstance(key, str):
                    continue
                if not key.startswith("session:"):
                    continue
                session_id = key.split("session:", 1)[-1]
                if not session_id.startswith("whatsapp_"):
                    continue

                raw = redis.get(key)
                if not raw:
                    continue

                try:
                    session = orjson.loads(raw)
                except Exception:
                    logger.warning("IdleMonitor: Failed to decode session JSON for %s", session_id)
                    continue

                # Fast pre-filters before acquiring lock
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

                # Passed all quick checks; process under lock
                await _process_idle_session(session_id, session, now)

            if cursor == 0 or cursor == "0":
                break
    except Exception as e:
        logger.error("IdleMonitor: Error while scanning sessions: %s", e)


async def idle_monitor_loop(poll_interval_seconds: int = 60) -> None:
    """
    Background loop to periodically run the idle farewell scanner.
    Intended to be started once per FastAPI worker via the app lifespan hook.
    """
    logger.info(
        "IdleMonitor: Starting idle monitor loop (enabled=%s, idle_seconds=%d, poll_interval=%ds)",
        ENABLE_IDLE_FAREWELL,
        IDLE_FAREWELL_SECONDS,
        poll_interval_seconds,
    )
    try:
        while True:
            try:
                await run_idle_farewell_scan_once()
            except Exception as e:
                logger.error("IdleMonitor: Unexpected error in scan loop: %s", e)
            await asyncio.sleep(poll_interval_seconds)
    except asyncio.CancelledError:
        logger.info("IdleMonitor: Monitor loop cancelled; shutting down.")
        raise


