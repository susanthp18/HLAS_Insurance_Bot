## Scope and Deliverables
- Perform a comprehensive, read-only audit of the repository, covering inventory, entry points, per‑file reviews, architecture, dependencies, quality assessment, and cross‑file data flows
- Produce a single structured document with:
  - Project architecture diagram (ASCII)
  - Module dependency graph (textual map)
  - Per‑file purpose, dependencies, key functions/classes, configuration and data flow
  - Core component descriptions and business logic flows
  - Code quality assessment and improvement recommendations

## Repository Structure (High‑Level)
- Top‑level: `Admin/`, `hlas/`, `Dockerfile.hlas`, `docker-compose.yml`, `Caddyfile`, `DEPLOYMENT.md`, root `requirements.txt`
- Runtime app: `hlas/src/hlas` with submodules:
  - Entry point `main.py` (FastAPI app, chat and WhatsApp routes) → `hlas/src/hlas/main.py:58`
  - Orchestration `flow.py` (`HlasFlow` router and handlers) → `hlas/src/hlas/flow.py:56`
  - Flows: `flows/info_flow.py`, `flows/compare_flow.py`, `flows/summary_flow.py`, `flows/rec_flow.py`
  - Agents/Tasks: `agents.py`, `tasks.py`, templates in `config/agents.yaml`, `config/tasks.yaml`
  - Tools: RAG and benefits comparison under `tools/`
  - Integrations: WhatsApp (`utils/whatsapp_handler.py`), Zoom live chat (`utils/zoom/*`)
  - LLM/Embeddings: `llm.py`, `vector_store.py`
  - Session & infra: `session.py` (Redis‑based), `redis_utils.py`, `metrics.py`, `logging_config.py`
- Data source corpus for ingestion: `Admin/source_db/*`; ingestion scripts in `Admin/embedding_agent.py`, `Admin/migrate_schema.py`

## Methodology
- Inventory: Enumerate all files/dirs; note hierarchy and organization patterns
- Entry points: Identify and verify external interfaces (`/chat`, `/meta-whatsapp`, `/ready`, `/metrics`) and their handlers
- Per‑file review (for every `.py`, `.yaml`, `.json`, and infra file):
  - Purpose/functionality
  - Dependencies/import graph (internal/external)
  - Key classes/functions with role and references (`file_path:line_number`)
  - Configuration usage (env vars, YAML templates)
  - Data flow (inputs/outputs, session mutations, external calls)
- Cross‑file analysis: Map flows ↔ tasks ↔ agents ↔ tools ↔ configs; identify integration points (Redis, Mongo history, Weaviate, Azure OpenAI, WhatsApp, Zoom)
- Documentation outputs: Architecture diagram (ASCII), dependency map, core flows narrative, per‑file summaries, quality findings

## Architecture Overview (to be documented)
- Request paths:
  - FastAPI `/chat` → `HlasFlow.kickoff_async` orchestrates intents and delegates to flow helpers
  - WhatsApp webhook `/meta-whatsapp` → `utils.whatsapp_handler` performs rate‑limit, dedupe, session, then calls `HlasFlow`
- Orchestration:
  - `HlasFlow.decide` routes to handlers: information, follow‑up, comparison, summary, recommendation, capabilities
  - Flows use CrewAI `agents` and `tasks` specs via `prompt_runner`
- Retrieval:
  - Weaviate hybrid search with named vectors (`content_vector`, `questions_vector`), Azure embeddings
  - Non‑RAG benefits via `benefits_tool`
- State:
  - Redis JSON session with small rolling history; Mongo append‑only conversation history
- Integrations:
  - WhatsApp Cloud API; Zoom Contact Center websocket and REST APIs

## Quality Assessment (planned checks)
- Code style consistency, typing, module boundaries and logging
- Error handling and resilience (Redis locks, external API failures)
- Test coverage (none present) and recommendations for critical modules (flows, tools, WhatsApp handler)
- Performance: caching, embedding strict mode, Weaviate query configuration
- Security: env/secrets handling, container exposure, dependency pinning

## Risks & Improvements (to be delivered with findings)
- Missing tests, duplicate/misaligned dependency pins (`weaviate-client` vs `weaviate_client`), repeated `PyYAML`, HTTP‑only Caddy, exposed DB ports in compose
- Strengthen JSON contract handling; unify response LLM selection; tighten webhook validation and message truncation policies

## Deliverables & Format
- One comprehensive document with clear sections, bullet summaries, and `file_path:line_number` references
- ASCII diagram for architecture and a textual dependency graph
- Prioritized recommendations list with quick‑win actions

## Timeline
- Immediate delivery upon approval with the full write‑up based on the completed read‑only audit
- Follow‑up iteration to refine diagrams and add any requested deep dives