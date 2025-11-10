from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
from .session import MongoSessionManager
from .logging_config import setup_logging
from .llm import initialize_models
from .utils.greeting import get_time_based_greeting
import sys
import uvicorn
from dotenv import load_dotenv
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr
import io

load_dotenv()
setup_logging()

# Initialize models
try:
    initialize_models()
except Exception as e:
    logging.critical("Application startup failed: Could not initialize LLM models.")
    sys.exit(1)

# Import LLM components AFTER logging is configured
from .flow import HlasFlow
from .llm import azure_llm, azure_embeddings
from .redis_utils import RedisLock, session_lock_key, get_redis
from .metrics import REQUESTS_TOTAL, REDIS_LOCK_TIMEOUTS
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Suppress noisy pydantic serializer warnings from underlying LLM/tooling libs
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing special (models pre-initialized)
    yield
    # Shutdown: close reusable HTTP clients
    await close_whatsapp_handler_http_client()

app = FastAPI(lifespan=lifespan)
# Redis-only session manager (backwards alias retained)
mongo_session_manager = MongoSessionManager()
logger = logging.getLogger(__name__)
# Always log startup for visibility
logger.info("Application startup: FastAPI app created. LLMs are pre-initialized via hlas.llm import.")

class ChatInput(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(payload: ChatInput):
    logger = logging.getLogger(__name__)
    logger.info("Chat.request: session_id=%s message='%s'", payload.session_id, payload.message)
    try:
        # Placeholder for tracing
        # with tracer.start_as_current_span("chat") as span:
        #     span.set_attribute("session_id", payload.session_id)

        # Check for "Hi" greeting BEFORE loading session to avoid using old state
        if payload.message.strip().lower() == "hi":
            logger.info("Chat.handler: Received 'hi' greeting - resetting session before processing")
            try:
                mongo_session_manager.reset_session(payload.session_id)
            except Exception as e:
                logger.error("Chat.handler: Failed to reset session for 'hi' greeting - %s", e)

            greeting = get_time_based_greeting()
            logger.info("Chat.handler: Responding with time-based greeting")
            return {"response": greeting, "sources": ""}

        # Execute HlasFlow for fully LLM-driven orchestration under a per-session lock
        flow = HlasFlow()
        lock_key = session_lock_key(payload.session_id)
        with RedisLock(lock_key, ttl_seconds=15.0, wait_timeout=5.0):
            session = mongo_session_manager.get_session(payload.session_id)
            logger.info("Chat.session_loaded: pending_slot='%s' product='%s' keys=%s",
                       session.get("pending_slot"), session.get("product"), list(session.keys()))
            # Suppress third-party console UIs from libraries during flow execution
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = await flow.kickoff_async(inputs={"message": payload.message, "session": session})
            # The flow's final state contains the complete, updated session
            final_session = flow.state.session

            # Build assistant reply payloads for history and persistence (no trimming)
            assistant_reply_full = str(flow.state.reply)
            assistant_reply_hist = assistant_reply_full
            
            # Persist the entire updated session state from the flow FIRST (to keep flags/slots)
            mongo_session_manager.save_session(payload.session_id, final_session)

            # Add the current turn to the history via the session manager (ensures history is preserved)
            mongo_session_manager.add_history_entry(payload.session_id, payload.message, assistant_reply_hist, assistant_reply_full)
            
            # Log the final session state
            logger.info("Chat.session_persist.final: rec_status='%s' cmp_status='%s' sum_status='%s' keys=%s",
                       final_session.get("recommendation_status"),
                       final_session.get("comparison_status"),
                       final_session.get("summary_status"),
                       list(final_session.keys()))
        logger.info("Chat.completed: product=%s reply_len=%d sources=%s",
                   final_session.get("product"), len(str(flow.state.reply or "")), str(flow.state.sources))
        REQUESTS_TOTAL.labels(endpoint="/chat", status="200").inc()
        return {"response": str(flow.state.reply), "sources": flow.state.sources}
    except TimeoutError as e:
        logger.error(f"Redis lock timeout: {e}", exc_info=True)
        REDIS_LOCK_TIMEOUTS.labels(scope="chat").inc()
        REQUESTS_TOTAL.labels(endpoint="/chat", status="503").inc()
        raise HTTPException(status_code=503, detail="Service busy, please retry")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        REQUESTS_TOTAL.labels(endpoint="/chat", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

# WhatsApp Integration Endpoints
from .utils.whatsapp_handler import whatsapp_handler, close_whatsapp_handler_http_client

@app.get("/health")
def health_check():
    """General liveness endpoint."""
    return {"status": "ok", "service": "BT Insurance Chatbot"}

@app.get("/ready")
def readiness_check():
    """
    Readiness: verify Redis connectivity. MongoDB is not used for sessions anymore.
    """
    details = {"mongo": "disabled", "redis": "unknown"}
    ok = True
    try:
        r = get_redis()
        r.ping()
        details["redis"] = "ok"
    except Exception as e:
        details["redis"] = f"error: {e}"
        ok = False
    status = "ok" if ok else "error"
    return {"status": status, **details}
@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/meta-whatsapp")
def meta_whatsapp_webhook_verification(request: Request):
    """
    Enhanced webhook verification for WhatsApp with comprehensive error handling.
    """
    return whatsapp_handler.verify_webhook(request)

@app.post("/meta-whatsapp")
async def meta_whatsapp_webhook(request: Request):
    """
    Enhanced webhook handler for incoming WhatsApp messages with production-grade features.
    """
    return await whatsapp_handler.process_webhook(request)

@app.get("/whatsapp/health")
def whatsapp_health_check():
    """
    WhatsApp-specific health check with detailed status information.
    """
    return whatsapp_handler.get_health_status()

# Using lifespan instead of deprecated on_event shutdown

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
