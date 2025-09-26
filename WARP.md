# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Language: Python
- App: FastAPI service exposing /chat and WhatsApp webhook endpoints
- Core domains: LLM orchestration (CrewAI Flows), Retrieval-Augmented Generation (Weaviate), session persistence (MongoDB)
- Key entrypoint: hlas/src/hlas/main.py

Prerequisites and environment
- Python 3.9+ recommended (zoneinfo used)
- Dependencies are pinned in requirements.txt
- Required services and env vars (must be set in your environment prior to running):
  - Azure OpenAI: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, optional AZURE_OPENAI_TEMPERATURE
  - MongoDB: MONGO_URI, DB_NAME
  - Weaviate: WEAVIATE_URL or WEAVIATE_ENDPOINT, optional WEAVIATE_API_KEY; Weaviate gRPC must be exposed on 50051
  - WhatsApp (Meta): META_VERIFY_TOKEN, META_ACCESS_TOKEN, META_PHONE_NUMBER_ID
  - Optional RAG tuning: RAG_TOP_K, RAG_ALPHA
  - Optional logging: LOGGING_ENABLED=true, LOG_FILE, LOG_LEVEL

Common commands
1) Create venv and install dependencies
```pwsh path=null start=null
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- If you plan to use the Admin crawling agent features that leverage Playwright:
```pwsh path=null start=null
python -m playwright install
```

2) Run the API server
- Simplest (runs uvicorn via __main__):
```pwsh path=null start=null
python hlas/src/hlas/main.py
```

- With auto-reload (set module path then use uvicorn):
```pwsh path=null start=null
$env:PYTHONPATH = "$PWD/hlas/src"
uvicorn hlas.main:app --host 0.0.0.0 --port 8000 --reload
```

- POSIX alternative:
```bash path=null start=null
export PYTHONPATH="$PWD/hlas/src"
uvicorn hlas.main:app --host 0.0.0.0 --port 8000 --reload
```

3) Smoke-test endpoints
- Health check:
```pwsh path=null start=null
Invoke-RestMethod -Method GET http://localhost:8000/health
```

- Chat inference:
```pwsh path=null start=null
$body = @{ session_id = "dev1"; message = "Tell me about Travel insurance benefits" } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri http://localhost:8000/chat -ContentType 'application/json' -Body $body
```

- WhatsApp webhook health:
```pwsh path=null start=null
Invoke-RestMethod -Method GET http://localhost:8000/whatsapp/health
```

4) Initialize MongoDB collections (indexes, optional reset)
```pwsh path=null start=null
python Admin/initialize_mongo.py --log-level INFO
# or, to drop 'sessions' and 'conversation_history' after confirmation
python Admin/initialize_mongo.py --reset --log-level INFO
```

5) Seed or refresh RAG sources (optional, for knowledge base ops)
- Crawl a product URL and extract FAQs/benefits/PDFs:
```pwsh path=null start=null
python Admin/crawling_agent.py https://example.com/insurance/travel-insurance
```

Notes on linting and tests
- No linter configuration or test suite was found in the repository. If you add one (e.g., ruff/pytest), use standard invocation patterns.

High-level architecture
- FastAPI service (hlas/src/hlas/main.py)
  - Pre-initializes Azure OpenAI chat and embeddings via initialize_models(); exits fast on misconfiguration
  - Endpoints:
    - POST /chat: primary chat entry; loads session, executes HlasFlow, persists state; special-case greeting for "hi"
    - GET /health: service health
    - GET/POST /meta-whatsapp and GET /whatsapp/health: webhook verification, async processing, and health for WhatsApp
- Session persistence (hlas/src/hlas/session.py)
  - MongoSessionManager singleton
  - Collections: sessions (session state), conversation_history (recent turns)
  - Methods: get_session, save_session (upsert without history), add_history_entry, reset_session
- LLM integration (hlas/src/hlas/llm.py)
  - Centralized Azure OpenAI config; exposes azure_llm (CrewAI LLM wrapper) and azure_embeddings (LangChain Azure embeddings)
  - initialize_models() validates required env vars and constructs clients
- Orchestration flow (hlas/src/hlas/flow.py → HlasFlow)
  - Router decides among directives: greet, handle_capabilities, handle_information, handle_follow_up, handle_summary, plan_only_comparison, handle_recommendation, handle_other
  - Delegates to helper flows based on state flags in session:
    - InfoFlowHelper (information retrieval)
    - CompareFlowHelper (tier/product comparisons)
    - SummaryFlowHelper (summaries)
    - RecFlowHelper (simplified recommendation flow; optional, guarded by import availability)
  - Uses run_direct_task to compose prompts from YAML specs (config/agents.yaml, config/tasks.yaml) and call LLMs consistently
- Flows and tasks (hlas/src/hlas/flows/*.py, hlas/src/hlas/tasks.py, hlas/src/hlas/agents.py)
  - CrewAI Agents constructed from config/agents.yaml with a uniform system/prompt template
  - Tasks built from config/tasks.yaml and mapped to agents in code
  - InfoFlow: hybrid RAG over Weaviate (BM25 + multi-vector) filtered by product; synthesizes using product-specific templates (config/ir_response.yaml)
  - CompareFlow: gathers product and tiers, fetches benefits, and synthesizes comparisons using templates (config/cmp_response.yaml)
  - SummaryFlow: similar slot bootstrap + synthesis using benefits and templates (config/summary_response.yaml)
  - RecFlow: collects product-specific slots with validators and question generation; final synthesis using templates (config/recommendation_response.yaml)
- Tools (hlas/src/hlas/tools/*.py)
  - RAG tool: Weaviate hybrid search leveraging azure_embeddings; filters by product and optional doc_type
  - Benefits tool: fetches benefit chunks by product
  - Comparison, Summary, Explanation, Source attribution helpers for agents
- Prompt and template plumbing (hlas/src/hlas/prompt_runner.py)
  - Loads YAML agent/task specs once; builds [SYSTEM]/[USER] prompts; resilient JSON extraction with regex fallback
- Vector store (hlas/src/hlas/vector_store.py)
  - Singleton Weaviate connection via connect_to_custom; expects HTTP (8080) and gRPC (50051) endpoints
- Logging (hlas/src/hlas/logging_config.py)
  - Package-scoped logger 'hlas' with console + timed rotating file handler when LOGGING_ENABLED=true
- Admin utilities (Admin/*.py)
  - initialize_mongo.py: creates indexes, optional destructive reset with confirmation, robust logging
  - crawling_agent.py: extracts FAQs, benefits (tables), PDFs; optional Gemini + Azure fallback; writes to Admin/source_db
  - embedding_agent.py, migrate_schema.py: present but not detailed here

Conventions and configuration
- YAML-driven behavior in hlas/src/hlas/config/ for:
  - agents.yaml and tasks.yaml (CrewAI agent/task definitions)
  - llms.yaml (model-related settings consumed by higher-level code where applicable)
  - ir_response.yaml, cmp_response.yaml, summary_response.yaml, recommendation_response.yaml (LLM response templates)
  - slot_validation_rules.yaml (per-product validation rules surfaced to validators)
- State fields commonly used in session:
  - product, slots, recommendation_status, comparison_status, summary_status, pending_slot, last_question, history, plus internal flags like _last_info_prod_q, _last_info_user_msg, _fu_query

Repo guidance
- No README.md, WARP.md, CLAUDE.md, Cursor rules, or Copilot instructions were found at the time of writing. If they’re added later, incorporate key instructions here.