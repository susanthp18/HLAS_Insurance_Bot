"""
Agentic WhatsApp Handler for HLAS Insurance Chatbot
===================================================

This module provides a parallel WhatsApp handler that uses the new
LangGraph-based agentic flow (agentic_chat) instead of the legacy HlasFlow.
It includes full live agent handoff support via Zoom integration.
"""

import os
import logging
import io
import uuid
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional
from functools import partial

from ..agentic import agentic_chat
from .whatsapp_handler import WhatsAppMessageHandler
from ..redis_utils import RedisLock, session_lock_key
from ..metrics import WA_MESSAGES_PROCESSED_TOTAL

# Import Zoom engagement manager
try:
    from .zoom.engagement import EngagementManager
    ZOOM_AVAILABLE = True
except ImportError:
    EngagementManager = None
    ZOOM_AVAILABLE = False

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("ZOOM_BASE_URL", "https://us01cciapi.zoom.us")


class AgenticWhatsAppMessageHandler(WhatsAppMessageHandler):
    """
    WhatsApp handler specialized for the Agentic (LangGraph) runtime.
    Inherits infrastructure (http client, verification, dedupe) from the main handler.
    Includes full Zoom live agent handoff support.
    """

    async def handle_message(self, message: str, user_phone: str, metadata: Dict[str, Any]) -> str:
        """
        Process the message through the Agentic chat system.
        """
        try:
            logger.info(f"AgenticWA: Processing message from {user_phone}: {message[:100]}...")
            
            session_id = f"whatsapp_{user_phone}"
            
            # Check for live agent status first
            if self._mongo_session_manager:
                try:
                    session = self._mongo_session_manager.get_session(session_id)
                    if self._is_live_agent_on(session.get("live_agent_status")):
                        logger.info("AgenticWA: live_agent_status active for %s - short-circuiting", session_id)
                        return "Live agent integration is under development. It will be coming soon. Please say 'hi' exactly to reset the session."
                except Exception as e:
                    logger.warning(f"AgenticWA: Error checking session: {e}")

            # Process through Agentic Chat
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = await agentic_chat(session_id, message)
            
            response = str(result.get("response") or "")
            debug_state = result.get("debug_state", {})
            
            # Check if the bot response indicates live agent handoff
            # The agentic bot detects live agent intent and responds with handoff phrase
            live_agent_requested = debug_state.get("live_agent_requested", False)
            
            # Set live_agent_status if requested
            if live_agent_requested and self._mongo_session_manager:
                try:
                    session = self._mongo_session_manager.get_session(session_id)
                    session["live_agent_status"] = True
                    self._mongo_session_manager.save_session(session_id, session)
                    logger.info(f"AgenticWA: Set live_agent_status=True for {session_id}")
                except Exception as e:
                    logger.error(f"AgenticWA: Failed to set live_agent_status: {e}")
            
            # Log history
            if self._mongo_session_manager:
                try:
                    self._mongo_session_manager.add_history_entry(
                        session_id, 
                        message, 
                        response[:100] if len(response) > 100 else response,
                        response
                    )
                except Exception as e:
                    logger.warning(f"AgenticWA: Failed to log history: {e}")

            if not response:
                response = "I'm sorry, I couldn't process your request. Please try again."
            
            # Truncate if needed
            if len(response) > self.max_message_length:
                response = response[:self.max_message_length-50] + "...\n\n(Message truncated)"
            
            logger.info(f"AgenticWA: Generated response for {user_phone}: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"AgenticWA: Error processing message from {user_phone}: {str(e)}")
            return "I'm sorry, there was an error processing your message. Please try again later."

    async def _process_and_respond(self, message: str, user_phone: str, metadata: Dict[str, Any]):
        """
        Handles the actual processing and sending of the response.
        Includes full live agent handoff support.
        """
        # Rate limiting check
        if not self.check_rate_limit(user_phone):
            rate_limit_msg = "You're sending messages too quickly! Please wait a moment and try again."
            await self._send_message_async(user_phone, rate_limit_msg)
            WA_MESSAGES_PROCESSED_TOTAL.labels(result="rate_limited").inc()
            return

        session_id = f"whatsapp_{user_phone}"
        
        with RedisLock(session_lock_key(session_id), ttl_seconds=15.0, wait_timeout=5.0):
            # Optional quick ack
            try:
                if os.getenv("WA_SEND_ACKS", "false").lower() in ("1", "true", "yes"):
                    await self._send_message_async(user_phone, "Got it - let me check that for you...")
            except Exception:
                pass

            # Check if live agent is already active BEFORE processing
            if self._mongo_session_manager:
                session = self._mongo_session_manager.get_session(session_id)
                if self._is_live_agent_on(session.get("live_agent_status")):
                    # Forward message to Zoom agent
                    if ZOOM_AVAILABLE and EngagementManager:
                        manager = EngagementManager.get_by_session(user_phone)
                        if not manager:
                            logger.error(f"AgenticWA: Zoom session for '{user_phone}' not found.")
                            await self._send_message_async(
                                user_phone, 
                                "Unfortunately the agent has disconnected. Please try again later or say 'hi' to restart."
                            )
                            # Reset live agent status
                            session["live_agent_status"] = False
                            self._mongo_session_manager.save_session(session_id, session)
                            WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_error").inc()
                            return

                        if not manager.is_agent_connected:
                            logger.info("AgenticWA: Agent not yet connected for %s", user_phone)
                            await self._send_message_async(
                                user_phone, 
                                "You're in a queue. Please wait while we connect you to an agent."
                            )
                            WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_queue").inc()
                        else:
                            # Forward message to agent
                            await manager.send_message(message)
                            WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_forwarded").inc()
                    else:
                        # Zoom not available
                        await self._send_message_async(
                            user_phone,
                            "Live agent integration is under development. Please say 'hi' to reset the session."
                        )
                        WA_MESSAGES_PROCESSED_TOTAL.labels(result="live_agent_unavailable").inc()
                    return

            # Process message through agentic bot
            response = await self.handle_message(message, user_phone, metadata)

            # Send response
            if response and response.strip():
                await self._send_message_async(user_phone, response)
                WA_MESSAGES_PROCESSED_TOTAL.labels(result="ok").inc()
            else:
                logger.info("AgenticWA: No response sent for %s (empty)", user_phone)
                WA_MESSAGES_PROCESSED_TOTAL.labels(result="empty").inc()

            # Check if live agent was requested AFTER processing
            if self._mongo_session_manager:
                session = self._mongo_session_manager.get_session(session_id)
                if self._is_live_agent_on(session.get("live_agent_status")):
                    # Initiate Zoom engagement
                    if ZOOM_AVAILABLE and EngagementManager:
                        if user_phone in EngagementManager._active_engagements:
                            logger.warning(f"AgenticWA: Engagement already exists for {user_phone}")
                            return

                        logger.info(f"AgenticWA: Creating Zoom engagement for {user_phone}")
                        callback = partial(self.handle_agent_response, user_phone)
                        
                        temp_name = uuid.uuid4().hex[:6]
                        temp_email = f"{temp_name}@hlastest.com"

                        manager = EngagementManager.create_and_register(
                            session_id=user_phone,
                            nick_name=temp_name,
                            email=temp_email,
                            base_api_url=BASE_URL,
                            on_agent_message_callback=callback
                        )
                        asyncio.create_task(manager.initiate_engagement())
                    else:
                        logger.warning("AgenticWA: Zoom not available for live agent handoff")

    async def handle_agent_response(self, user_phone: str, message):
        """Callback triggered by EngagementManager for agent messages."""
        logger.info(f"AgenticWA: Agent message for '{user_phone}': {message}")
        
        # Forward message to customer via WhatsApp
        if isinstance(message, str):
            await self._send_message_async(user_phone, message)
        elif isinstance(message, dict):
            text = message.get("text") or message.get("message") or str(message)
            await self._send_message_async(user_phone, text)
        
        # Check for chat closed
        session_id = f"whatsapp_{user_phone}"
        
        if isinstance(message, str) and message == "This chat has been closed.":
            logger.info(f"AgenticWA: Agent closed chat for {user_phone}")
            if self._mongo_session_manager:
                session = self._mongo_session_manager.get_session(session_id)
                session["live_agent_status"] = False
                self._mongo_session_manager.save_session(session_id, session)
            await self.close_engagement_and_cleanup(user_phone)
        
        # Check for consumer_disconnected event
        if isinstance(message, dict):
            event = message.get("event")
            if event == "consumer_disconnected":
                logger.info(f"AgenticWA: Agent ended chat for {user_phone}")
                if self._mongo_session_manager:
                    session = self._mongo_session_manager.get_session(session_id)
                    session["live_agent_status"] = False
                    self._mongo_session_manager.save_session(session_id, session)
                await self.close_engagement_and_cleanup(user_phone)

    async def close_engagement_and_cleanup(self, session_id: str):
        """Gracefully close and remove Zoom engagement."""
        if not ZOOM_AVAILABLE or not EngagementManager:
            return
            
        manager = EngagementManager.get_by_session(session_id)
        if manager:
            manager.unregister(session_id)
            await manager.close()
            logger.info(f"AgenticWA: Cleaned up Zoom engagement for {session_id}")


# Global agentic handler instance
agentic_whatsapp_handler = AgenticWhatsAppMessageHandler()
