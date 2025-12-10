# HLAS Insurance Bot - Agentic Layer Testing Instructions

This project includes a new **Agentic Layer** built with **LangGraph**, which acts as an intelligent orchestrator using LangChain tools and a multi-agent supervisor pattern.

## Prerequisites

1.  **Environment**: Ensure you have the necessary API keys in your `.env` file.
    *   `AZURE_OPENAI_API_KEY`
    *   `AZURE_OPENAI_ENDPOINT`
    *   `AZURE_OPENAI_API_VERSION`
    *   `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` (or `AZURE_OPENAI_CHAT_DEPLOYMENT`)
    *   `WEAVIATE_URL` (if using Weaviate for info tool)
    *   `WEAVIATE_API_KEY` (if using Weaviate cloud)

2.  **Docker Services**: Ensure Redis, MongoDB, and Weaviate are running.
    ```bash
    docker-compose up -d
    ```

3.  **Dependencies**: Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Server

Start the FastAPI server using Uvicorn:

```bash
python -m uvicorn hlas.src.hlas.main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing with Postman

The new agentic behavior is exposed via the `/agent-chat` endpoint. This endpoint is separate from the legacy `/chat` endpoint and `WhatsApp` webhooks, allowing independent testing.

### Endpoint Details

*   **URL**: `http://localhost:8000/agent-chat`
*   **Method**: `POST`
*   **Header**: `Content-Type: application/json`

### Request Body

```json
{
  "session_id": "test-session-001",
  "message": "I am planning a trip to Japan next week."
}
```

### Test Scenarios

Run the following conversation sequences to verify the intelligence and multi-agent orchestration.

#### Scenario 1: Recommendation Flow (Travel)
1.  **User**: "I want to buy travel insurance."
    *   *Expectation*: Agent asks for destination (e.g., "Where are you travelling to?").
    *   *Note*: The agent should use the `RecommendationAgent` subgraph.
2.  **User**: "Japan."
    *   *Expectation*: Agent generates a recommendation (likely "Gold" tier) with benefits text and advisory.

#### Scenario 2: Information Query (RAG)
1.  **User**: "Does the travel plan cover lost baggage?"
    *   *Expectation*: Agent uses `InfoAgent` (RAG tool) to search Weaviate and provide a specific answer based on the policy wording.

#### Scenario 3: Comparison
1.  **User**: "Compare the Basic and Gold plans for Travel."
    *   *Expectation*: Agent uses `CompareAgent` to provide a side-by-side comparison of benefits.

#### Scenario 4: Summary
1.  **User**: "Give me a summary of the Maid insurance."
    *   *Expectation*: Agent uses `SummaryAgent` to provide a high-level overview.

#### Scenario 5: Context Switching & Intelligence
1.  **User**: "I'm looking for Maid insurance."
    *   *Expectation*: Agent asks for `maid_country` or `duration`.
2.  **User**: "Actually, tell me about Home insurance instead."
    *   *Expectation*: Agent switches context to Home insurance and starts that flow (or answers info request), dropping the Maid context appropriately.

#### Scenario 6: Negative Feedback & Reflection
1.  **User**: "That answer was not helpful at all."
    *   *Expectation*: The `Supervisor` detects negative feedback, triggers `SelfCritique`, and the agent attempts to rephrase or ask clarifying questions to recover.

## Debugging

The response from `/agent-chat` includes a `debug_state` field which shows:
*   `intent`: The classified intent (e.g., `recommend`, `info`).
*   `product`: The detected product.
*   `rec_ready`: Status of the recommendation flow.

Use this to verify if the agent is routing correctly.
