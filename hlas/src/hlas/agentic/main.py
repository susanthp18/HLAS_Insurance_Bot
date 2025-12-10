#!/usr/bin/env python3
"""
HLAS Agentic Chatbot - Standalone FastAPI Application
=====================================================

Production-ready FastAPI server for the agentic chatbot.
Can be deployed independently from the legacy HLAS system.

Usage:
    uvicorn hlas.agentic.main:app --host 0.0.0.0 --port 8000
    
Or with auto-reload for development:
    uvicorn hlas.agentic.main:app --reload --port 8000
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import agentic components
from . import agentic_chat
from .infrastructure import (
    SessionManager,
    llm_cleanup,
    initialize_models,
)
from .infrastructure.metrics import AGENTIC_MESSAGES_TOTAL, AGENTIC_LATENCY

# Import handlers
from .handlers import (
    handle_agentic_whatsapp_verification,
    handle_agentic_whatsapp_message,
    close_agentic_whatsapp_client,
    agentic_whatsapp_handler,
)
from .infrastructure.idle_monitor import (
    idle_monitor_loop,
    set_whatsapp_handler,
    ENABLE_IDLE_FAREWELL,
)

# Optional: Prometheus metrics
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================
# Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting HLAS Agentic Chatbot...")
    
    # Pre-initialize models (optional - can also lazy load)
    try:
        initialize_models()
        logger.info("LLM models pre-initialized")
    except Exception as e:
        logger.warning(f"LLM pre-initialization failed (will lazy load): {e}")
    
    # Register WhatsApp handler for idle monitor
    set_whatsapp_handler(agentic_whatsapp_handler)
    
    # Start idle monitor background task
    idle_monitor_task = None
    if ENABLE_IDLE_FAREWELL:
        idle_monitor_task = asyncio.create_task(idle_monitor_loop())
        logger.info("Idle monitor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HLAS Agentic Chatbot...")
    
    # Cancel idle monitor
    if idle_monitor_task:
        idle_monitor_task.cancel()
        try:
            await idle_monitor_task
        except asyncio.CancelledError:
            pass
    
    await close_agentic_whatsapp_client()
    llm_cleanup()
    logger.info("Shutdown complete")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="HLAS Agentic Chatbot",
    description="Production-ready LangGraph-based insurance chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: Optional[str] = ""
    debug_state: Optional[dict] = None


# ============================================
# Health & Metrics Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hlas-agentic",
        "version": "1.0.0",
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - verifies dependencies."""
    checks = {}
    
    # Check Redis
    try:
        from .infrastructure import get_redis
        redis = get_redis()
        redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    
    # Check if models can be initialized
    try:
        from .infrastructure import get_chat_llm
        llm = get_chat_llm()
        checks["llm"] = "ok" if llm else "not initialized"
    except Exception as e:
        checks["llm"] = f"error: {e}"
    
    all_ok = all(v == "ok" for v in checks.values())
    
    return {
        "ready": all_ok,
        "checks": checks,
    }


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


# ============================================
# Chat Endpoints
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for the agentic chatbot.
    
    Args:
        request: ChatRequest with session_id and message
        
    Returns:
        ChatResponse with bot response and debug info
    """
    import time
    start_time = time.time()
    
    try:
        result = await agentic_chat(request.session_id, request.message)
        
        # Record metrics
        latency = time.time() - start_time
        AGENTIC_LATENCY.labels(endpoint="chat").observe(latency)
        AGENTIC_MESSAGES_TOTAL.labels(
            result="ok",
            product=result.get("debug_state", {}).get("product") or "unknown"
        ).inc()
        
        return ChatResponse(
            response=result.get("response", ""),
            sources=result.get("sources", ""),
            debug_state=result.get("debug_state"),
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        AGENTIC_MESSAGES_TOTAL.labels(result="error", product="unknown").inc()
        return ChatResponse(
            response="I'm sorry, something went wrong. Please try again.",
            debug_state={"error": str(e)},
        )


@app.post("/agent-chat", response_model=ChatResponse)
async def agent_chat_endpoint(request: ChatRequest):
    """Alias for /chat endpoint (backwards compatibility)."""
    return await chat_endpoint(request)


# ============================================
# WhatsApp Webhook Endpoints
# ============================================

@app.get("/webhook/whatsapp")
async def whatsapp_verification(request: Request):
    """WhatsApp webhook verification (GET)."""
    return handle_agentic_whatsapp_verification(request)


@app.post("/webhook/whatsapp")
async def whatsapp_message(request: Request):
    """WhatsApp message handler (POST)."""
    return await handle_agentic_whatsapp_message(request)


# Alternative webhook paths (for flexibility)
@app.get("/agentic-webhook")
async def agentic_webhook_verification(request: Request):
    """Alternative WhatsApp webhook verification."""
    return handle_agentic_whatsapp_verification(request)


@app.post("/agentic-webhook")
async def agentic_webhook_message(request: Request):
    """Alternative WhatsApp message handler."""
    return await handle_agentic_whatsapp_message(request)


# ============================================
# Session Management Endpoints
# ============================================

@app.post("/session/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a session to initial state."""
    try:
        session_manager = SessionManager()
        session_manager.reset_session(session_id)
        return {"status": "ok", "message": f"Session {session_id} reset"}
    except Exception as e:
        logger.error(f"Session reset error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session state (for debugging)."""
    try:
        session_manager = SessionManager()
        session = session_manager.get_session(session_id)
        return {"status": "ok", "session": session}
    except Exception as e:
        logger.error(f"Session get error: {e}")
        return {"status": "error", "message": str(e)}


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "hlas.agentic.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
    )
