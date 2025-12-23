"""
Enhanced WhatsApp Handler for Production-Grade BT Insurance Chatbot
===================================================================

This module provides comprehensive WhatsApp message handling with robust
error recovery, validation, and production-grade features.
"""

import os
import re
import logging
import uuid
from typing import Dict, Any, Optional, Tuple
from functools import partial
from datetime import datetime
import asyncio
from fastapi import Request, Response
import httpx
from contextlib import redirect_stdout, redirect_stderr
import io
import time
import hmac
import hashlib
from zoneinfo import ZoneInfo

from ..redis_utils import RateLimiter, Deduplicator, OrderGuard, RedisLock, session_lock_key
from ..metrics import WA_MESSAGES_PROCESSED_TOTAL, REDIS_LOCK_TIMEOUTS

# Import BT chatbot components at module level to avoid circular imports and runtime overhead
try:
    # Backwards-compatible alias: MongoSessionManager now maps to Redis-only SessionManager
    from ..session import MongoSessionManager
    from ..flow import HlasFlow
    from ..utils.greeting import get_time_based_greeting
    HLAS_IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BT chatbot imports not available: {e}")
    MongoSessionManager = None
    HlasFlow = None
    get_time_based_greeting = None
    HLAS_IMPORTS_AVAILABLE = False

from .zoom.engagement import EngagementManager

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("ZOOM_BASE_URL", "https://us01cciapi.zoom.us")

class WhatsAppMessageHandler:
    """
    Enhanced WhatsApp message handler with production features.
    """
    
    def __init__(self):
        self.verify_token = os.environ.get("META_VERIFY_TOKEN")
        self.access_token = os.environ.get("META_ACCESS_TOKEN")
        self.phone_number_id = os.environ.get("META_PHONE_NUMBER_ID")
        self.max_message_length = 4096  # WhatsApp limit

        # Reusable async HTTP client for outbound WhatsApp messages
        # Set sane timeouts and allow re-use of connections to reduce latency.
        self._http: Optional[httpx.AsyncClient] = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)
        )

        # Redis-backed controls
        self.rate_limiter = RateLimiter()
        self.deduper = Deduplicator()
        self.order_guard = OrderGuard()
        
        # Initialize shared Redis-backed session manager
        self._mongo_session_manager = None
        if HLAS_IMPORTS_AVAILABLE and MongoSessionManager:
            try:
                self._mongo_session_manager = MongoSessionManager()
                try:
                    from ..redis_utils import get_redis
                    r = get_redis()
                    if r.set("log_once:wa_handler_init", "1", nx=True, ex=3600):
                        logger.info("WhatsApp handler initialized with Redis-only session manager")
                except Exception:
                    logger.info("WhatsApp handler initialized with Redis-only session manager")
            except Exception as e:
                logger.error(f"Failed to initialize session manager: {e}")
                self._mongo_session_manager = None
        
    def _is_live_agent_on(self, val: Any) -> bool:
        """Normalize various truthy representations of 'on'."""
        try:
            if isinstance(val, str):
                return val.strip().lower() in ("on", "true", "yes", "1")
            return bool(val)
        except Exception:
            return False
        
    def verify_webhook(self, request: Request) -> Response:
        """
        Verifies the webhook subscription with Meta with enhanced validation.
        """
        try:
            # Extract query parameters
            mode = request.query_params.get("hub.mode")
            token = request.query_params.get("hub.verify_token")
            challenge = request.query_params.get("hub.challenge")
            
            logger.info(f"Webhook verification attempt - Mode: {mode}, Token present: {bool(token)}")
            
            # Validate required parameters
            if not all([mode, token, challenge]):
                logger.warning("Missing required webhook verification parameters")
                return Response(content="Missing parameters", status_code=400)
            
            # Check the mode and token
            if mode == "subscribe" and token == self.verify_token:
                logger.info("Webhook verification successful")
                return Response(content=challenge, status_code=200)
            else:
                logger.warning(f"Webhook verification failed - Invalid mode or token")
                return Response(content="Verification failed", status_code=403)
                
        except Exception as e:
            logger.error(f"Error in webhook verification: {str(e)}")
            return Response(content="Internal error", status_code=500)
    
    def extract_message_data(self, data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Extract message and user information from WhatsApp webhook data with validation.
        
        Returns:
            Tuple[message, user_phone_number, metadata]
        """
        try:
            # Check if this is a status update (e.g., 'sent', 'delivered', 'read')
            value = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
            if 'statuses' in value:
                try:
                    status_info = value['statuses'][0]
                    status = status_info.get('status', 'unknown')
                    recipient_id = status_info.get('recipient_id', 'unknown')
                    logger.info(f"Received '{status}' status update for {recipient_id}. Ignoring.")
                except (IndexError, KeyError):
                    logger.info("Received a status update with unexpected format. Ignoring.")
                return None, None, {}

            # Multiple extraction patterns for different webhook formats
            extraction_patterns = [
                # Standard format
                lambda d: (
                    d['entry'][0]['changes'][0]['value']['messages'][0]['text']['body'],
                    d['entry'][0]['changes'][0]['value']['messages'][0]['from']
                ),
                # Alternative format 1
                lambda d: (
                    d['entry']['changes']['value']['messages']['text']['body'],
                    d['entry']['changes']['value']['messages']['from']
                ),
                # Alternative format 2
                lambda d: (
                    d['body']['text'],
                    d['from']
                )
            ]
            
            message = None
            user_phone = None
            metadata = {}
            
            for pattern in extraction_patterns:
                try:
                    message, user_phone = pattern(data)
                    if message and user_phone:
                        break
                except (KeyError, IndexError, TypeError):
                    continue
            
            if not message or not user_phone:
                # This is now only an error if it's not a status update
                logger.warning(f"Could not extract message data from webhook. Not a user message or status update: {data}")
                return None, None, {}
            
            # Extract additional metadata
            try:
                if 'entry' in data and isinstance(data['entry'], list):
                    entry = data['entry'][0]
                    if 'changes' in entry and isinstance(entry['changes'], list):
                        change = entry['changes'][0]
                        if 'value' in change and 'messages' in change['value']:
                            msg_data = change['value']['messages'][0]
                            metadata = {
                                'message_id': msg_data.get('id'),
                                'timestamp': msg_data.get('timestamp'),
                                'type': msg_data.get('type', 'text'),
                                'from_name': change['value'].get('contacts', [{}])[0].get('profile', {}).get('name', 'Unknown')
                            }
            except Exception as e:
                logger.warning(f"Could not extract metadata: {str(e)}")
            
            # Validate and clean message
            message = self.validate_and_clean_message(message)
            user_phone = self.validate_phone_number(user_phone)
            
            return message, user_phone, metadata
            
        except Exception as e:
            logger.error(f"Error extracting message data: {str(e)}")
            return None, None, {}
    
    def validate_and_clean_message(self, message: str) -> Optional[str]:
        """
        Validate and clean incoming message.
        """
        if not message:
            return None
        
        # Remove excessive whitespace
        message = re.sub(r'\s+', ' ', message.strip())
        
        # Check length
        if len(message) > self.max_message_length:
            logger.warning(f"Message too long: {len(message)} characters")
            message = message[:self.max_message_length] + "..."
        
        # Basic content filtering (can be enhanced)
        if len(message) < 1:
            return None
        
        return message
    
    def validate_phone_number(self, phone: str) -> Optional[str]:
        """
        Validate and normalize phone number.
        """
        if not phone:
            return None
        
        # Remove non-numeric characters except +
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        # Basic validation
        if len(clean_phone) < 8 or len(clean_phone) > 15:
            logger.warning(f"Invalid phone number format: {phone}")
            return None
        
        return clean_phone
    
    def check_rate_limit(self, user_phone: str) -> bool:
        """
        Redis-backed rate limiting check (mandatory).
        """
        # Any Redis issue will raise and be logged by the limiter.
        return self.rate_limiter.allow(user_phone)
    
    async def handle_message(self, message: str, user_phone: str, metadata: Dict[str, Any]) -> str:
        """
        Process the message through the BT chat system with error handling.
        """
        try:
            logger.info(f"Processing message from {user_phone}: {message[:100]}...")
            
            # Check if BT components are available
            if not HLAS_IMPORTS_AVAILABLE or not self._mongo_session_manager:
                logger.error("BT components not available for message processing")
                return "I'm sorry, the service is temporarily unavailable. Please try again later."
            
            # Use phone number as session ID (could be enhanced with user mapping)
            session_id = f"whatsapp_{user_phone}"
            
            # Check for "Hi" greeting BEFORE loading session to avoid using old state
            if message.lower().strip() == "hi":
                logger.info("WhatsApp handler: Received 'hi' greeting - resetting session before processing")

                try:
                    self._mongo_session_manager.reset_session(session_id)
                except Exception as e:
                    logger.error(f"WhatsApp handler: Failed to reset session for 'hi' greeting - {e}")

                greeting = get_time_based_greeting()
                logger.info("WhatsApp handler: Responding with time-based greeting")
                return greeting
            
            # Get session
            session = self._mongo_session_manager.get_session(session_id)

            # If live agent is active, short-circuit without calling orchestrator
            if self._is_live_agent_on(session.get("live_agent_status")):
                logger.info("WhatsApp handler: live_agent_status active for %s - short-circuiting reply", session_id)
                return "Live agent integration is under development. It will be coming soon. Please say 'hi' exactly to reset the session."
            
            # Process through BT Flow
            flow = HlasFlow()
            
            # Suppress third-party console UIs during flow execution
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = await flow.kickoff_async(inputs={"message": message, "session": session})
            
            # Get the response
            response = str(flow.state.reply or "")
            
            # Decide whether to trim assistant reply for history storage (only for rec/summary/comparison)
            assistant_reply_hist = response
            try:
                s = flow.state.session or {}
                should_truncate = False
                if s.get("recommendation_status") is not None or s.get("comparison_status") is not None or s.get("summary_status") is not None:
                    should_truncate = True
                if s.get("last_completed") in ("recommendation", "comparison", "summary"):
                    should_truncate = True
                if should_truncate and isinstance(response, str) and len(response) > 100:
                    assistant_reply_hist = response[:100]
            except Exception:
                assistant_reply_hist = response

            # Add to history and save session
            self._mongo_session_manager.add_history_entry(session_id, message, assistant_reply_hist, response)

            # Re-fetch the latest session from Redis to preserve the updated history
            cached_after_hist = self._mongo_session_manager.get_session(session_id)

            # Update session state based on latest cached snapshot (preserves history)
            new_session = dict(cached_after_hist)
            new_session.update({
                "product": flow.state.product or cached_after_hist.get("product"),
            })

            # Persist session state (similar to main.py logic), but preserve updates and propagate removals
            # 1) Core fields
            if "slots" in flow.state.session:
                new_session["slots"] = flow.state.session["slots"]
            if flow.state.product or flow.state.session.get("product"):
                new_session["product"] = flow.state.product or flow.state.session.get("product") or new_session.get("product")

            # 2) Flow statuses: persist when present; remove when cleared by flow
            for key in ("recommendation_status", "comparison_status", "summary_status", "policy_status_status", "claim_status_status"):
                if key in flow.state.session:
                    new_session[key] = flow.state.session.get(key)
                else:
                    if key in new_session:
                        new_session.pop(key, None)

            # 3) Working slots for multi-turn flows: keep when present; remove when flow cleared them
            for key in ("comparison_slot", "summary_slot"):
                if key in flow.state.session:
                    new_session[key] = flow.state.session.get(key)
                else:
                    new_session.pop(key, None)

            # 4) Ephemeral guidance flags: copy if present; otherwise ensure they are removed
            for key in (
                # Existing keys (already working)
                "last_question", "_last_info_prod_q", "_last_info_user_msg",
                "pending_slot", "_fu_query", "fraud_stage",

                # Slot extraction context (fixes Early flow stuck bug on server)
                "_last_slot_name", "_last_slot_question",

                # RecFlow product clarification
                "_last_rec_prod_q", "_tentative_product",

                # Fraud flow skip extraction flag
                "_skip_extraction_once",

                # Early product educational notice
                "_early_existing_cover_notice",

                # Product switch confirmation marker
                "__product_switch_confirmed__",

                # Policy/Claim status flow working state
                "policy_status_slots", "policy_status_last_slot",
                "claim_status_slots", "claim_status_last_slot",
            ):
                if key in flow.state.session and flow.state.session.get(key) not in (None, ""):
                    new_session[key] = flow.state.session.get(key)
                else:
                    new_session.pop(key, None)

            # Step 2: Handle recommended_tier removal propagation (e.g., on product switch)
            if "recommended_tier" in flow.state.session:
                new_session["recommended_tier"] = flow.state.session["recommended_tier"]
            else:
                # Flow cleared it (e.g., on product switch) - propagate removal to prevent stale tier
                new_session.pop("recommended_tier", None)

            # 5) Histories and markers: copy if present
            for key in ("comparison_history", "summary_history", "last_completed"):
                if key in flow.state.session:
                    new_session[key] = flow.state.session.get(key)

            # 6) Live agent flag: only set when explicitly present in flow session
            if "live_agent_status" in flow.state.session:
                new_session["live_agent_status"] = flow.state.session.get("live_agent_status")

            # Save session
            self._mongo_session_manager.save_session(session_id, new_session)
            
            # Validate response
            if not response:
                response = "I'm sorry, I couldn't process your request. Please try again or ask for help."
            
            # Ensure response fits WhatsApp limits
            if len(response) > self.max_message_length:
                response = response[:self.max_message_length-50] + "...\n\nMessage was truncated. Please ask for specific details!"
            
            logger.info(f"Generated response for {user_phone}: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message from {user_phone}: {str(e)}")
            return "I'm sorry, there was an error processing your message. Please try again later."
    
    async def _send_message_async(self, recipient_number: str, message_body: str):
        """
        Sends a WhatsApp message asynchronously using the Meta API with retries/backoff.
        """
        if not self.phone_number_id or not self.access_token:
            logger.error("Environment variables META_PHONE_NUMBER_ID and/or META_ACCESS_TOKEN are not set.")
            return

        if not self._http:
            raise RuntimeError("HTTP client not initialized")

        url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_number,
            "type": "text",
            "text": {
                "body": message_body
            }
        }

        # Retry with exponential backoff (non-blocking)
        attempts = 0
        max_attempts = 3
        backoff = 0.5
        while True:
            try:
                response = await self._http.post(url, headers=headers, json=payload)
                response.raise_for_status()
                try:
                    logger.info(f"Message sent successfully to {recipient_number}. Response: {response.json()}")
                except Exception:
                    logger.info(f"Message sent successfully to {recipient_number}.")
                break
            except httpx.HTTPError as e:
                attempts += 1
                logger.warning(f"Send attempt {attempts}/{max_attempts} failed for {recipient_number}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        logger.warning(f"Response status code: {e.response.status_code}")
                        logger.warning(f"Response content: {e.response.text}")
                    except Exception:
                        pass
                if attempts >= max_attempts:
                    logger.error(f"Exhausted retries sending message to {recipient_number}")
                    break
                await asyncio.sleep(backoff)
                backoff *= 2
    
    async def _process_and_respond(self, message: str, user_phone: str, metadata: Dict[str, Any]):
        """
        Handles the actual processing and sending of the response asynchronously.
        """
        # Rate limiting check
        if not self.check_rate_limit(user_phone):
            rate_limit_msg = "You're sending messages too quickly! 😅 Please wait a moment and try again."
            await self._send_message_async(user_phone, rate_limit_msg)
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="rate_limited").inc()
            return

        # Acquire per-session lock to avoid concurrent processing for same user
        session_id = f"whatsapp_{user_phone}"
        with RedisLock(session_lock_key(session_id), ttl_seconds=15.0, wait_timeout=5.0):
            # Check if the stage is "live_agent" before an intent is set by the incoming message
            if self._mongo_session_manager.get_session(session_id).get("live_agent_status") == True:
                manager = EngagementManager.get_by_session(user_phone)
                if not manager:
                    logger.error(f"Zoom Chat session for contact '{user_phone}' not found.")
                    await self._send_message_async(user_phone, "Unfortunately the agent has disconnected. Please try again later.")

                if not manager.is_agent_connected:
                    logger.error("Agent has not connected yet.")
                    await self._send_message_async(user_phone, "You're in a queue. Please wait while we are trying to connect you to an agent.")
                else:
                    await manager.send_message(message)

            else:
                # Process message
                response = await self.handle_message(message, user_phone, metadata)
                
                # Send response
                await self._send_message_async(user_phone, response)
                WA_MESSAGES_PROCESSED_TOTAL.labels(result="ok").inc()

                #Check if the stage is "live_agent" after the intent is set by the incoming message
                if self._mongo_session_manager.get_session(session_id).get("live_agent_status") == True:
                    if user_phone in EngagementManager._active_engagements:
                        logger.error(f"Chat session '{session_id}' is already active.")

                    logger.info(f"Creating and storing new engagement for session: {user_phone}")
                    callback_with_session_context = partial(self.handle_agent_response, user_phone)
                    
                    temp_name = uuid.uuid4().hex[:6]
                    temp_email = f"{temp_name}@hlastest.com"

                    manager = EngagementManager.create_and_register(
                        session_id=user_phone,
                        nick_name=temp_name,
                        email=temp_email,
                        base_api_url=BASE_URL,
                        on_agent_message_callback=callback_with_session_context
                    )
                    asyncio.create_task(manager.initiate_engagement())

    async def process_webhook(self, request: Request) -> Response:
        """
        Main webhook processing function. It acknowledges the request immediately
        and then processes the message in the background.
        """
        try:
            # Verify request signature if app secret is configured
            raw_body = await request.body()
            app_secret = os.environ.get("META_APP_SECRET")
            sig_header = request.headers.get("X-Hub-Signature-256")
            if app_secret:
                if not sig_header or not sig_header.startswith("sha256="):
                    logger.error("Webhook signature missing or malformed; rejecting")
                    return Response(status_code=403)
                expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
                provided = sig_header.split("=", 1)[1]
                if not hmac.compare_digest(expected, provided):
                    logger.error("Webhook signature mismatch; rejecting")
                    return Response(status_code=403)

            data = await request.json()
            logger.debug(f"Received webhook data: {data}")
            
            message, user_phone, metadata = self.extract_message_data(data)
            
            if message and user_phone:
                # De-duplication using WhatsApp message_id when available
                message_id = (metadata.get('message_id') if isinstance(metadata, dict) else None) or ""
                if message_id:
                    try:
                        if not self.deduper.is_new(message_id):
                            logger.info("Duplicate message detected (message_id=%s). Ignoring.", message_id)
                            WA_MESSAGES_PROCESSED_TOTAL.labels(result="duplicate").inc()
                            return Response(status_code=200)
                    except Exception:
                        pass
                
                # Basic ordering: drop messages older than last processed
                try:
                    ts = int(metadata.get('timestamp')) if metadata.get('timestamp') else int(time.time())
                except Exception:
                    ts = int(time.time())
                if not self.order_guard.allow(user_phone, ts):
                    logger.info("Out-of-order message dropped for %s (ts=%s)", user_phone, ts)
                    WA_MESSAGES_PROCESSED_TOTAL.labels(result="out_of_order").inc()
                    return Response(status_code=200)

                # Acknowledge immediately and process in the background
                asyncio.create_task(self._process_and_respond(message, user_phone, metadata))
            
            # Always return 200 to acknowledge receipt of the event
            return Response(status_code=200)
            
        except TimeoutError as e:
            logger.error(f"Redis lock timeout for WhatsApp session: {e}")
            REDIS_LOCK_TIMEOUTS.labels(scope="whatsapp").inc()
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="error").inc()
            return Response(status_code=200)
        except Exception as e:
            logger.error(f"Critical error in webhook processing: {str(e)}")
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="error").inc()
            # Still return 200 to avoid webhook disabling, but log the error
            return Response(status_code=200)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status for monitoring.
        """
        try:
            # Session is persisted in Redis; per-user rate limiting stats are tracked in Prometheus
            return {
                "status": "healthy",
                "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "webhook_verification_token_configured": bool(self.verify_token)
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "error": str(e)
            }

    # Handlers for Zoom Engagement setup
    async def handle_agent_response(self, user_phone: str, message: dict | str):
        """Callback triggered by EngagementManager for agent messages."""
        logger.info(f"Callback for number '{user_phone}': Received message from agent: {message}")
        # --- FORWARD MESSAGE TO CUSTOMER THROUGH WHATSAPP ---
        await self._send_message_async(user_phone, message)
        if(message == "This chat has been closed."):
            session = self._mongo_session_manager.get_session(f"whatsapp_{user_phone}")
            session["live_agent_status"] = False
            self._mongo_session_manager.save_session(f"whatsapp_{user_phone}", session)
            await self.close_engagement_and_cleanup(user_phone)
        
        event = message.get("event") if isinstance(message, dict) else None
        if event == "consumer_disconnected":
            logger.info(f"Agent ended chat session for numebr '{user_phone}'. Cleaning up.")
            session = self._mongo_session_manager.get_session(f"whatsapp_{user_phone}")
            session["live_agent_status"] = False
            self._mongo_session_manager.save_session(f"whatsapp_{user_phone}", session)
            await self.close_engagement_and_cleanup(user_phone)


    async def close_engagement_and_cleanup(self, session_id: str):
        """Helper to gracefully close and remove an engagement from the session store."""
        manager = EngagementManager.get_by_session(session_id)
        if manager:
            manager.unregister(session_id)  # Remove from registry as well, if needed. del EngagementManager._active_engagements[session_id]
            await manager.close()
            logger.info(f"Successfully closed and cleaned up session: {session_id}")


# Global handler instance
whatsapp_handler = WhatsAppMessageHandler()

# Expose a close method for FastAPI shutdown
async def close_whatsapp_handler_http_client():
    try:
        if whatsapp_handler._http:
            await whatsapp_handler._http.aclose()
            whatsapp_handler._http = None
            logger.info("Closed WhatsApp AsyncClient")
    except Exception as e:
        logger.error(f"Failed to close WhatsApp AsyncClient: {e}")

# Convenience functions for FastAPI routes
async def handle_whatsapp_verification(request: Request) -> Response:
    """Handle WhatsApp webhook verification."""
    return whatsapp_handler.verify_webhook(request)

async def handle_whatsapp_message(request: Request) -> Response:
    """Handle incoming WhatsApp messages."""
    return await whatsapp_handler.process_webhook(request)

# Unused function removed - use whatsapp_handler.get_health_status() directly if needed

