# HLAS Insurance Bot – Codebase Overview

## 1) File Structure Analysis
- Top-level files and directories:
  - `Admin/`
    - `source_db/benefits/*_benefits.txt`
    - `source_db/FAQ/*_FAQs.txt`
    - `source_db/policy/*_policy.md`
    - `embedding_agent.py`: Weaviate ingestion and Azure OpenAI embeddings
    - `migrate_schema.py`: Weaviate schema migration utility
    - `initialize_mongo.py`: MongoDB collections/index initialization script
  - `hlas/`
    - `requirements.txt`: runtime deps for app image
    - `src/hlas/`
      - `main.py`: FastAPI app and HTTP endpoints
      - `flow.py`: `HlasFlow` orchestration and routing
      - `flows/compare_flow.py`: plan/tier comparison logic
      - `flows/info_flow.py`: product-aware RAG information responses
      - `flows/summary_flow.py`: plan/tier summaries
      - `flows/rec_flow.py`: recommendation flow and slot management
      - `agents.py`: CrewAI agents built from YAML specs
      - `tasks.py`: CrewAI task mapping to agents from YAML specs
      - `prompt_runner.py`: prompt assembly and JSON parsing for tasks
      - `lc/structured.py`: optional LangChain structured output bridge
      - `tools/benefits_tool.py`: product benefits retrieval from JSON
      - `tools/comparison_tool.py`: simple multi-product/tier comparison
      - `tools/rag_tool.py`: Weaviate hybrid named-vector search tool
      - `utils/whatsapp_handler.py`: WhatsApp webhook, rate-limit, dedupe
      - `utils/zoom/engagement.py`: Zoom Contact Center orchestration
      - `utils/zoom/websocket.py`: Zoom ZPNS websocket lifecycle
      - `utils/product_lex.py`: lexical product candidate detection
      - `llm.py`: LLM/embeddings initialization and configuration
      - `vector_store.py`: Weaviate client
      - `session.py`: Redis-backed session manager (Mongo alias)
      - `redis_utils.py`: Redis client, locks, rate-limit, dedupe, order guard
      - `metrics.py`: Prometheus counters and histograms
      - `logging_config.py`: gated logging setup
      - `config/agents.yaml`: agent specs
      - `config/tasks.yaml`: task specs
      - `config/*_response.yaml`: response templates
      - `config/benefits_raw.json`: corpus for benefits tool
      - `config/slot_validation_rules.yaml`: slot validator rules
      - `config/knowledge_base.txt`: capabilities KB
  - Infra:
    - `Dockerfile.hlas`: app image build
    - `docker-compose.yml`: Caddy, app, Redis, Mongo, Weaviate
    - `Caddyfile`: HTTP reverse proxy
    - `DEPLOYMENT.md`: offline deployment instructions
    - Root `requirements.txt`: pinned deps for tooling/admin

## 2) Main Entry Points and Core Modules
- FastAPI app `hlas/src/hlas/main.py`:
  - `/chat` POST → orchestrates a single turn with session persistence (`hlas/src/hlas/main.py:58`)
  - `/meta-whatsapp` GET/POST → webhook verification and WhatsApp message handling (`hlas/src/hlas/main.py:152`, `hlas/src/hlas/main.py:159`)
  - `/ready`, `/health`, `/metrics` → readiness, liveness and Prometheus (`hlas/src/hlas/main.py:130`, `hlas/src/hlas/main.py:125`, `hlas/src/hlas/main.py:146`)
- Orchestrator `HlasFlow` in `flow.py`: intent routing and multi-flow delegation (`hlas/src/hlas/flow.py:56`)
- Flow helpers:
  - Information: `InfoFlowHelper.handle` (`hlas/src/hlas/flows/info_flow.py:22`)
  - Comparison: `CompareFlowHelper.handle` (`hlas/src/hlas/flows/compare_flow.py:42`)
  - Summary: `SummaryFlowHelper.handle` (`hlas/src/hlas/flows/summary_flow.py:25`)
  - Recommendation: `RecFlowHelper.handle` (`hlas/src/hlas/flows/rec_flow.py:692`)
- Agents/Tasks: `agents.py`, `tasks.py` with YAML specifications
- Tools: Weaviate RAG, benefits and comparison tools under `tools/`
- Integrations: WhatsApp webhook (`utils/whatsapp_handler.py`) and Zoom live chat (`utils/zoom/*`)

## 3) Detailed File Reviews
- `hlas/src/hlas/main.py`
  - Purpose: FastAPI app, model initialization, HTTP endpoints
  - Dependencies: FastAPI, Pydantic, Redis, Prometheus, WhatsApp handler, `HlasFlow`
  - Key constructs:
    - `ChatInput` pydantic model (`hlas/src/hlas/main.py:54`)
    - Chat handler `/chat` → acquire Redis lock, load session, call `HlasFlow.kickoff_async`, persist session and history (`hlas/src/hlas/main.py:58`–`hlas/src/hlas/main.py:111`)
    - Readiness ping Redis (`hlas/src/hlas/main.py:130`–`hlas/src/hlas/main.py:145`)
    - WhatsApp webhook routes bound to `whatsapp_handler` (`hlas/src/hlas/main.py:152`–`hlas/src/hlas/main.py:164`)
  - Config: `.env` for Azure, Redis, Mongo; logging gated by `DEBUG`
  - Data flow: HTTP → session lock → `HlasFlow` → session update → history append (Mongo) → response JSON

- `hlas/src/hlas/flow.py`
  - Purpose: Central flow class that routes intents and coordinates sub-flows
  - Dependencies: CrewAI Flow decorators, YAML config loader, tasks/agents, flow helpers
  - Key methods:
    - `HlasFlow.ingest` start node (`hlas/src/hlas/flow.py:135`)
    - `HlasFlow.decide` router with product switch detection and directives (`hlas/src/hlas/flow.py:147`–`hlas/src/hlas/flow.py:743`)
  - Config: cached agents/tasks via `config_loader`
  - Data flow: message + session → directive → delegated helper → reply/sources + session mutation

- `hlas/src/hlas/flows/info_flow.py`
  - Purpose: One-turn information flow using RAG
  - Dependencies: Azure embeddings/LLM, Weaviate client, YAML IR templates
  - Key function: `InfoFlowHelper.handle` (`hlas/src/hlas/flows/info_flow.py:22`)
  - Data flow: ensure product → embed query → hybrid search with named vectors → synthesize answer via response LLM → sources list

- `hlas/src/hlas/flows/compare_flow.py`
  - Purpose: Stateful comparison across tiers for a product
  - Dependencies: tasks for product/tier identification, benefits tool, response LLM, YAML comparison templates
  - Key function: `CompareFlowHelper.handle` (`hlas/src/hlas/flows/compare_flow.py:42`)
  - Data flow: maintain `comparison_slot` → ensure product + tiers → clarify if missing → fetch benefits → synthesize comparison → mark `comparison_status=done`

- `hlas/src/hlas/flows/summary_flow.py`
  - Purpose: Flexible summary for product and tiers (1+ tiers)
  - Dependencies: tasks for identification, benefits tool, response LLM, YAML summary templates
  - Key function: `SummaryFlowHelper.handle` (`hlas/src/hlas/flows/summary_flow.py:25`)
  - Data flow: maintain `summary_slot` → ensure product + tier(s) → clarify → retrieve benefits → synthesize summary → mark `summary_status=done`

- `hlas/src/hlas/flows/rec_flow.py`
  - Purpose: Slot-driven recommendation flow with validation and generation
  - Dependencies: benefits tool, agents for extraction/validation/question, response LLM, YAML recommendation templates
  - Key functions:
    - `RecFlowHelper.handle` main entry (`hlas/src/hlas/flows/rec_flow.py:692`)
    - `_extract_slots` (`hlas/src/hlas/flows/rec_flow.py:231`)
    - `_validate_slot` (`hlas/src/hlas/flows/rec_flow.py:304`)
    - `_ask_next_question` (`hlas/src/hlas/flows/rec_flow.py:362`)
    - `_generate_recommendation` (`hlas/src/hlas/flows/rec_flow.py:408`)
  - Data flow: product identification → slot extraction → validation → ask next or generate → mark `recommendation_status`

- `hlas/src/hlas/agents.py`
  - Purpose: Build CrewAI agents from `config/agents.yaml`
  - Key constructs: `build_agent_from_config` (`hlas/src/hlas/agents.py:29`), prebuilt agents like `product_identifier`, `orchestrator`, `slot_validator`, response agents (`hlas/src/hlas/agents.py:46`–`hlas/src/hlas/agents.py:56`)

- `hlas/src/hlas/tasks.py`
  - Purpose: Map `config/tasks.yaml` to CrewAI `Task` objects and expose names
  - Key constructs: `build_task` (`hlas/src/hlas/tasks.py:40`), named tasks (
    `identify_product_task`, `route_decision_task`, `construct_follow_up_query_task`, …) (`hlas/src/hlas/tasks.py:53`–`hlas/src/hlas/tasks.py:61`)

- `hlas/src/hlas/prompt_runner.py`
  - Purpose: Build prompts from YAML specs, run LLMs and parse JSON
  - Key functions: `build_prompts` (`hlas/src/hlas/prompt_runner.py:25`), `run_direct_task` (`hlas/src/hlas/prompt_runner.py:149`)

- `hlas/src/hlas/lc/structured.py`
  - Purpose: Optional LangChain structured output path for specific tasks
  - Key function: `structured_invoke` (`hlas/src/hlas/lc/structured.py:74`)

- `hlas/src/hlas/tools/rag_tool.py`
  - Purpose: RAG hybrid search with named vectors and optional Redis cache
  - Key constructs: `RAGTool._run` (`hlas/src/hlas/tools/rag_tool.py:24`)

- `hlas/src/hlas/tools/benefits_tool.py`
  - Purpose: Read `benefits_raw.json` and return product benefits text
  - Key constructs: `BenefitsTool._run` (`hlas/src/hlas/tools/benefits_tool.py:52`)

- `hlas/src/hlas/tools/comparison_tool.py`
  - Purpose: Compare up to three product/tier inputs via benefits tool
  - Key constructs: `_parse_product_and_tier` (`hlas/src/hlas/tools/comparison_tool.py:33`), `ComparisonTool._run` (`hlas/src/hlas/tools/comparison_tool.py:15`)

- `hlas/src/hlas/utils/whatsapp_handler.py`
  - Purpose: WhatsApp webhook verification and message processing
  - Key functions:
    - `verify_webhook` (`hlas/src/hlas/utils/whatsapp_handler.py:96`)
    - `process_webhook` (`hlas/src/hlas/utils/whatsapp_handler.py:523`)
    - `handle_message` (`hlas/src/hlas/utils/whatsapp_handler.py:253`)
    - `_process_and_respond` sends async replies and manages locks (`hlas/src/hlas/utils/whatsapp_handler.py:462`)
  - Data flow: verify → dedupe/rate-limit/order-guard → lock → `HlasFlow` → send message → optional Zoom engagement

- `hlas/src/hlas/utils/zoom/engagement.py`
  - Purpose: Manage Zoom Contact Center engagement lifecycle
  - Key constructs: `EngagementManager.create_and_register` (`hlas/src/hlas/utils/zoom/engagement.py:70`), `initiate_engagement` (`hlas/src/hlas/utils/zoom/engagement.py:227`), websocket message handling (`hlas/src/hlas/utils/zoom/engagement.py:103`)

- `hlas/src/hlas/utils/zoom/websocket.py`
  - Purpose: WebSocket connection management to ZPNS
  - Key constructs: `WebSocketManager.connect` (`hlas/src/hlas/utils/zoom/websocket.py:41`), `_listen_for_messages` (`hlas/src/hlas/utils/zoom/websocket.py:93`), `_send_pings` (`hlas/src/hlas/utils/zoom/websocket.py:118`)

- `hlas/src/hlas/utils/product_lex.py`
  - Purpose: Lexical detection for product candidates and hints
  - Key functions: `lexical_product_candidates` (`hlas/src/hlas/utils/product_lex.py:171`), `lexical_product_hint` (`hlas/src/hlas/utils/product_lex.py:148`)

- `hlas/src/hlas/llm.py`
  - Purpose: Initialize LLM and embeddings for Azure/Grok providers
  - Key function: `initialize_models` (`hlas/src/hlas/llm.py:54`)
  - Config: environment for endpoints, deployments, temperatures; separate response LLM

- `hlas/src/hlas/vector_store.py`
  - Purpose: Connect to Weaviate using HTTP+gRPC with disabled init checks
  - Key functions: `get_weaviate_client` (`hlas/src/hlas/vector_store.py:14`), `close_weaviate_client` (`hlas/src/hlas/vector_store.py:54`)

- `hlas/src/hlas/session.py`
  - Purpose: Redis-backed session manager with rolling history and idle reset
  - Key functions: `get_session` (`hlas/src/hlas/session.py:63`), `save_session` (`hlas/src/hlas/session.py:119`), `add_history_entry` (`hlas/src/hlas/session.py:132`), `reset_session` (`hlas/src/hlas/session.py:160`)

- `hlas/src/hlas/redis_utils.py`
  - Purpose: Redis client, distributed locks, rate-limiter, dedupe, order guard
  - Key constructs: `get_redis` (`hlas/src/hlas/redis_utils.py:33`), `RedisLock` (`hlas/src/hlas/redis_utils.py:55`), `RateLimiter` (`hlas/src/hlas/redis_utils.py:148`), `Deduplicator` (`hlas/src/hlas/redis_utils.py:173`), `OrderGuard` (`hlas/src/hlas/redis_utils.py:193`)

- `hlas/src/hlas/metrics.py`
  - Purpose: Prometheus counters for HTTP and WhatsApp processing, session cache, locks

- `hlas/src/hlas/logging_config.py`
  - Purpose: Configure `hlas` logger for console-only when `DEBUG=true`

- `hlas/src/hlas/mongo_history.py`
  - Purpose: Optional append-only Mongo history writer with graceful no-op
  - Key function: `log_history` (`hlas/src/hlas/mongo_history.py:50`)

- `Admin/embedding_agent.py`
  - Purpose: Chunk corpus, generate embeddings via Azure, ingest into Weaviate
  - Key functions: `embed_product` (`Admin/embedding_agent.py:340`), `generate_hypothetical_questions` (`Admin/embedding_agent.py:183`)

- `Admin/migrate_schema.py`
  - Purpose: Example schema migration script for Weaviate

- `Admin/initialize_mongo.py`
  - Purpose: Initialize MongoDB collections and indexes; optional reset

- Infra files: `Dockerfile.hlas`, `docker-compose.yml`, `Caddyfile`, `DEPLOYMENT.md`

## 4) Documentation: Architecture Diagram and Dependency Graph
- Architecture (ASCII):
  - Client → Caddy → FastAPI `main.py` → `HlasFlow.decide` → [Info | Compare | Summary | Rec | Capabilities]
  - Info → `vector_store` + Azure embeddings → Weaviate hybrid → response LLM → reply
  - Compare/Summary → `benefits_tool` + templates → response LLM → reply
  - Rec → slot extractor/validator/question → benefits → response LLM → reply
  - State → Redis session cache; History → Mongo append-only
  - WhatsApp → webhook → `whatsapp_handler` → `HlasFlow` → async reply via Meta API
  - Zoom → engagement manager + websocket
- Module dependency map (key edges):
  - `main.py` → `flow.py`, `session.py`, `redis_utils.py`, `metrics.py`, `utils/whatsapp_handler.py`
  - `flow.py` → `config_loader`, `tasks.py`, `agents.py`, `prompt_runner.py`, `flows/*`, `tools/*`, `utils/*`
  - `flows/info_flow.py` → `vector_store.py`, `llm.py`, YAML templates
  - `flows/compare_flow.py`/`summary_flow.py` → `tasks.py`, `benefits_tool.py`, YAML templates
  - `flows/rec_flow.py` → `tasks.py`, `benefits_tool.py`, `llm.py`
  - `utils/whatsapp_handler.py` → `redis_utils.py`, `flow.py`, `session.py`, `metrics.py`, `utils/zoom/*`
  - `tools/rag_tool.py` → `vector_store.py`, `llm.py`, `redis_utils.get_redis`

## 5) Quality Assessment
- Code style consistency: Python 3.11 idioms, Pydantic models, explicit logging and defensive error handling across flows and integrations
- Error handling patterns:
  - Redis locks with token verification and timeout exceptions (`hlas/src/hlas/redis_utils.py:55`)
  - WhatsApp webhook signature verification and robust message extraction (`hlas/src/hlas/utils/whatsapp_handler.py:523`)
  - Weaviate connection guarded; embeddings strict mode avoids BM25 fallback (`hlas/src/hlas/flows/info_flow.py:255`)
- Test coverage: no unit tests present; recommend adding coverage for `flow.py`, `flows/*`, `utils/whatsapp_handler.py`, `redis_utils.py`, `tools/*`
- Performance considerations:
  - Optional Redis cache in RAG tool; controlled top‑k and alpha
  - Named vectors for multi‑vector hybrid search; single embedding per query reused
  - Session cache TTL and idle reset to prevent stale contexts
  - Async HTTP client reuse for WhatsApp outbound messages
- Security:
  - Env‑driven secrets; compose currently exposes Mongo/Weaviate ports; recommend internal-only exposure in production
  - Caddy configured HTTP‑only; enable TLS for internet-facing deployments when feasible

## 6) Cross‑File Relationships and Data Flows
- Flows ↔ Tasks/Agents: YAML‑defined contracts executed via `prompt_runner` with JSON outputs
- Flows ↔ Tools: benefits/RAG tools provide domain content for synthesis
- Orchestrator ↔ Session: slot states, flow statuses, ephemeral flags managed in Redis
- WhatsApp ↔ Flow: webhook routes into `HlasFlow`; responses sent via Meta API; live agent integration via Zoom
- Mongo history: append‑only turn logging; non‑critical if disabled

## 7) Potential Architectural Issues and Improvements
- Testing: introduce unit tests and light integration tests for flows and handlers
- Dependency pinning: reconcile `weaviate-client` vs `weaviate_client`; remove duplicate `PyYAML` entry in root `requirements.txt`
- Strict JSON parsing: centralize parsing utilities and ensure consistent fallbacks across flows
- Observability: expand Prometheus metrics for per‑directive counts and flow durations
- Security hardening: default compose to internal DB ports; provide TLS Caddy profile

## 8) Core Business Logic Flows
- Information: product confirmation → RAG hybrid search → templated answer → concise sources
- Comparison: ensure product and 2+ tiers → benefits retrieval → templated comparison → mark done
- Summary: ensure product and ≥1 tier (Car/Early ignore tiers) → templated summary → mark done
- Recommendation: slot collection/validation → tier selection rules → templated recommendation → mark done

## 9) Appendix: Configuration and Templates
- Agent/task specs in `config/agents.yaml`, `config/tasks.yaml`
- Response templates: `config/recommendation_response.yaml`, `config/cmp_response.yaml`, `config/ir_response.yaml`, `config/summary_response.yaml`
- Validation rules: `config/slot_validation_rules.yaml`
- Benefits corpus: `config/benefits_raw.json`; admin `source_db/*` as upstream inputs