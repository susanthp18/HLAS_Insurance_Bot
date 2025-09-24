from fastapi import FastAPI, HTTPException, Request, Response
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

# Suppress noisy pydantic serializer warnings from underlying LLM/tooling libs
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
)

app = FastAPI()
mongo_session_manager = MongoSessionManager()
logger = logging.getLogger(__name__)
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

        # Execute HlasFlow for fully LLM-driven orchestration
        flow = HlasFlow()
        session = mongo_session_manager.get_session(payload.session_id)
        logger.info("Chat.session_loaded: pending_slot='%s' product='%s' keys=%s",
                   session.get("pending_slot"), session.get("product"), list(session.keys()))
        # Suppress third-party console UIs from libraries during flow execution
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = await flow.kickoff_async(inputs={"message": payload.message, "session": session})
        # Persist updated session state if any slots or product were set by the flow
        new_session = dict(session)
        # Trim long assistant replies in history to 100 characters for all responses
        assistant_reply_full = str(flow.state.reply)
        assistant_reply_hist = assistant_reply_full
        try:
            if isinstance(assistant_reply_full, str) and len(assistant_reply_full) > 100:
                assistant_reply_hist = assistant_reply_full[:100]
        except Exception:
            assistant_reply_hist = assistant_reply_full
        
        # Add the current turn to the history via the session manager
        mongo_session_manager.add_history_entry(payload.session_id, payload.message, assistant_reply_hist)
        
        new_session.update({
            "product": flow.state.product or session.get("product"),
        })
        if "slots" in flow.state.session:
            new_session["slots"] = flow.state.session["slots"]
        # Persist recommendation_status only for simplified state management
        if flow.state.session.get("recommendation_status"):
            new_session["recommendation_status"] = flow.state.session.get("recommendation_status")
            logger.info("Chat.session_persist: recommendation_status='%s'", new_session["recommendation_status"])
        
        # Persist last_question for slot extractor context
        if flow.state.session.get("last_question"):
            new_session["last_question"] = flow.state.session.get("last_question")
            logger.info("Chat.session_persist: last_question='%s'", new_session["last_question"])
            
        # Persist InfoFlow edge-case flags for product clarification detection
        if flow.state.session.get("_last_info_prod_q"):
            new_session["_last_info_prod_q"] = flow.state.session.get("_last_info_prod_q")
            logger.info("Chat.session_persist: _last_info_prod_q=%s", new_session["_last_info_prod_q"])
        else:
            new_session.pop("_last_info_prod_q", None)
            
        if flow.state.session.get("_last_info_user_msg"):
            new_session["_last_info_user_msg"] = flow.state.session.get("_last_info_user_msg")
            logger.info("Chat.session_persist: _last_info_user_msg='%s'", new_session["_last_info_user_msg"])
        else:
            new_session.pop("_last_info_user_msg", None)
        # Persist comparison and summary states
        compare_pending_flow = flow.state.session.get("compare_pending")
        if compare_pending_flow:
            new_session["compare_pending"] = compare_pending_flow
        else:
            new_session.pop("compare_pending", None)
            
        summary_pending_flow = flow.state.session.get("summary_pending")
        if summary_pending_flow:
            new_session["summary_pending"] = summary_pending_flow
        else:
            new_session.pop("summary_pending", None)
        # Also persist other comparison state
        if flow.state.session.get("comparison_slot"):
            new_session["comparison_slot"] = flow.state.session.get("comparison_slot")
        if flow.state.session.get("comparison_history"):
            new_session["comparison_history"] = flow.state.session.get("comparison_history")
        # Persist summary state
        if flow.state.session.get("summary_slot"):
            new_session["summary_slot"] = flow.state.session.get("summary_slot")
        if flow.state.session.get("summary_history"):
            new_session["summary_history"] = flow.state.session.get("summary_history")
        # Log the final session state before persisting
        logger.info("Chat.session_persist.final: rec_status='%s' keys=%s",
                   new_session.get("recommendation_status"), list(new_session.keys()))
        mongo_session_manager.save_session(payload.session_id, new_session)
        logger.info("Chat.completed: product=%s reply_len=%d sources=%s",
                   flow.state.product, len(str(flow.state.reply or "")), str(flow.state.sources))
        return {"response": str(flow.state.reply), "sources": flow.state.sources}
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        # Placeholder for metrics
        # chat_counter.add(1, {"status": "error"})
        raise HTTPException(status_code=500, detail=str(e))

# WhatsApp Integration Endpoints
from .utils.whatsapp_handler import whatsapp_handler

@app.get("/health")
def health_check():
    """General health check endpoint."""
    return {"status": "ok", "service": "HLAS Insurance Chatbot"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)