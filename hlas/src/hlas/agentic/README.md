# HLAS Agentic Chatbot

Production-ready LangGraph-based chatbot for HLAS Insurance with Redis session management, MongoDB conversation history, and optional Zoom live agent integration.

## Architecture

```
agentic/
├── __init__.py           # Main agentic_chat() entry point
├── config.py             # Configuration loaders
├── graph.py              # LangGraph graph definition
├── state.py              # AgentState type definitions
├── configs/              # YAML templates and configs
├── handlers/             # Channel handlers (WhatsApp)
├── infrastructure/       # Core infrastructure
│   ├── llm.py           # Azure OpenAI LLM setup
│   ├── vector_store.py  # Weaviate client
│   ├── redis_utils.py   # Redis utilities
│   ├── redis_checkpointer.py  # LangGraph Redis persistence
│   ├── session.py       # Session management
│   ├── mongo_history.py # MongoDB history logging
│   └── metrics.py       # Prometheus metrics
├── integrations/         # Optional integrations
│   └── zoom/            # Zoom Contact Center (live agent)
├── nodes/               # LangGraph nodes
├── tools/               # LangChain tools
├── utils/               # Utility functions
└── scripts/             # Initialization scripts
```

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-openai langgraph redis pymongo weaviate-client prometheus-client httpx orjson
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Initialize Infrastructure

```bash
# Verify all services
python -m hlas.agentic.scripts.healthcheck

# Initialize MongoDB collections and indexes
python -m hlas.agentic.scripts.init_mongodb

# Verify Redis (optional: --clear to reset)
python -m hlas.agentic.scripts.init_redis
```

### 4. Run

```python
from hlas.agentic import agentic_chat

# Async usage
result = await agentic_chat(session_id="user123", message="Hello!")
print(result["response"])
```

## Required Services

| Service | Purpose | Required |
|---------|---------|----------|
| Redis | Session state, LangGraph checkpoints, rate limiting | **Yes** |
| MongoDB | Conversation history persistence | **Yes** |
| Azure OpenAI | LLM for chat and embeddings | **Yes** |
| Weaviate | Vector store for RAG | Optional |
| Zoom Contact Center | Live agent handoff | Optional |

## Environment Variables

See `.env.example` for all available configuration options.

### Critical Variables

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Redis (Required)
REDIS_URL=redis://localhost:6379/0

# MongoDB (Required)
MONGO_URI=mongodb://localhost:27017
DB_NAME=hlas
```

## API Reference

### Main Entry Point

```python
async def agentic_chat(session_id: str, message: str) -> Dict[str, Any]:
    """
    Process a user message through the agentic chatbot.
    
    Args:
        session_id: Unique session identifier (e.g., "whatsapp_1234567890")
        message: User's message text
        
    Returns:
        {
            "response": str,           # Bot's response
            "sources": str,            # RAG sources (if any)
            "debug_state": {
                "intent": str,         # Detected intent
                "product": str,        # Detected product
                "rec_ready": bool,     # Recommendation ready
                "live_agent_requested": bool  # Live agent handoff
            }
        }
    """
```

### WhatsApp Handler

```python
from hlas.agentic.handlers import (
    handle_agentic_whatsapp_verification,
    handle_agentic_whatsapp_message,
)

# FastAPI routes
app.get("/agentic-webhook")(handle_agentic_whatsapp_verification)
app.post("/agentic-webhook")(handle_agentic_whatsapp_message)
```

## Session Management

Sessions are stored in Redis with configurable TTL:

```python
from hlas.agentic.infrastructure import SessionManager

manager = SessionManager()

# Get or create session
session = manager.get_session("user123")

# Update session
session["product"] = "Travel"
manager.save_session("user123", session)

# Check live agent status
is_live = manager.is_live_agent_active("user123")
```

## Conversation Persistence

LangGraph conversation state persists in Redis (survives restarts):

```bash
# Enable Redis checkpointer (default: true)
AGENTIC_USE_REDIS_CHECKPOINTER=true
```

Conversation history is logged to MongoDB:

```python
from hlas.agentic.infrastructure import log_history, get_history

# Log a turn
log_history(
    session_id="user123",
    user_message="What's covered?",
    assistant_message="Travel insurance covers...",
    metadata={"product": "Travel"}
)

# Retrieve history
history = get_history("user123", limit=20)
```

## Live Agent Handoff

When the user requests to speak to a human, the bot responds with a handoff message and sets `live_agent_requested=True`:

```python
result = await agentic_chat("user123", "I want to speak to a human")
if result["debug_state"].get("live_agent_requested"):
    # Initiate Zoom engagement
    pass
```

## Metrics

Prometheus metrics available:

- `agentic_messages_total` - Total messages processed
- `agentic_latency_seconds` - Response latency
- `agentic_session_cache_hits_total` - Session cache hits
- `agentic_live_agent_handoffs_total` - Live agent requests
- `agentic_policy_violations_total` - Policy violations detected

## Docker Deployment

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: hlas

  weaviate:
    image: semitechnologies/weaviate:1.24.1
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  redis_data:
  mongo_data:
  weaviate_data:
```

## Troubleshooting

### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping
# Should return: PONG
```

### MongoDB Connection Failed
```bash
# Check MongoDB is running
mongosh --eval "db.adminCommand('ping')"
```

### LLM Initialization Failed
- Verify `AZURE_OPENAI_ENDPOINT` includes trailing slash
- Check API key is valid
- Verify deployment names match your Azure resources

### Weaviate Connection Failed
- Check gRPC port (default: 50051)
- Verify `WEAVIATE_URL` is correct
- RAG features will be disabled if Weaviate is unavailable

## License

Proprietary - HLAS Insurance
