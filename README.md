# MCP Memory Server

A semantic memory layer for AI coding agents. It allows agents to store and retrieve memories with automatic summarization, reducing token usage during long coding sessions.

## Features

- **Automatic Summarization** — Stores the original text and generates a concise summary in the background (Groq primary, Anthropic fallback).
- **Semantic Retrieval** — Uses vector embeddings based on the raw text for accurate relevance matching. Returns summaries by default to save tokens; full text is opt-in via `include_full_text`.
- **Recency-Weighted Reranking** — Retrieval blends vector similarity with a recency score so recent memories get a slight ranking boost over older ones with similar content.
- **Near-Duplicate Detection** — Storing a memory that closely matches an existing one updates the existing record instead of creating a duplicate.
- **Non-Blocking Writes** — Store calls return immediately after embedding; summarization runs asynchronously and populates a few seconds later.
- **MCP Native Support** — Exposes `store_memory`, `retrieve_memory`, and `delete_memory` as MCP tools.
- **REST API** — Simple HTTP endpoints for testing and integration (`/store`, `/retrieve`, `/health`).
- **Session & Project Support** — Optional `session_id` and `project_id` fields for organizing memories.
- **Efficient** — Uses ONNX backend with quantized embeddings for lower memory usage.

## Quick Start

```bash
pip install -r requirements.txt
python -m memory_server
```

That's it — no database to run. Memories live in a single SQLite file at
`~/.memory_server/memories.db` (override with `MEMORY_DB_PATH`). The server
runs at `http://localhost:8000`.

<details>
<summary>Legacy Weaviate backend</summary>

The original Weaviate backend is still available:

```bash
pip install weaviate-client
docker compose up -d
MEMORY_BACKEND=weaviate python -m memory_server
```

</details>

### Interactive Documentation

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to test the API.

## Usage

### REST Endpoints

**Store a memory**

```bash
curl -X POST http://localhost:8000/store \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The agent is building a memory layer using Weaviate and FastAPI.",
    "source": "conversation",
    "category": "project",
    "session_id": "session-123",
    "project_id": "mcp-memory"
  }'
```

**Retrieve memories** (with optional filters)

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "memory layer",
    "top_k": 5,
    "session_id": "session-123",
    "category": "project",
    "source": "conversation",
    "include_full_text": false
  }'
```

> **Note:** Retrieval returns summaries and metadata by default. Set `"include_full_text": true` to include the original text in results. This keeps token usage low when only the summary is needed.

**Delete a memory**

```bash
curl -X DELETE http://localhost:8000/memory/<memory-id>
```

### MCP Tools

The server exposes tools via the Model Context Protocol at:

```
http://localhost:8000/mcp
```

**Available tools:**

- `store_memory(text, source, category, tags, session_id, project_id)` — Returns immediately; summarization runs in the background. The summary field on a freshly stored memory may be empty for a few seconds until the background task completes.
- `retrieve_memory(query, top_k, session_id, project_id, category, source, include_full_text)` — Returns summary + metadata by default. Pass `include_full_text=true` to include original text.
- `delete_memory(memory_id)`

These can be called directly by any MCP-compatible agent.

### Behavior Notes

- **Async summarization**: Both `store_memory` and `POST /store` return as soon as the embedding is computed and the record is inserted. The LLM-generated summary is populated asynchronously a few seconds later. If you retrieve a memory immediately after storing it, the `summary` field may be empty.
- **Near-duplicate merging**: If a new memory's embedding is within a cosine distance of 0.05 of an existing record, the existing record is updated rather than creating a new one. This prevents redundant entries when the same fact is stored with minor wording differences.
- **Recency reranking**: Retrieved results are reranked using a combined score of vector similarity and recency. Recent memories receive a slight boost. The decay weight is small enough (0.01 per day) that strong semantic matches still outrank recent but less relevant ones.

## Environment Variables

| Variable              | Description                                      | Default     |
|-----------------------|--------------------------------------------------|-------------|
| `GROQ_API_KEY`        | Groq API key for free summarization              | *(required)* |
| `ANTHROPIC_API_KEY`   | Anthropic API key (paid fallback only)           | *(optional)* |
| `MEMORY_BACKEND`      | Storage backend: `sqlite` or `weaviate`          | `sqlite`     |
| `MEMORY_DB_PATH`      | SQLite database file location                    | `~/.memory_server/memories.db` |
| `WEAVIATE_HOST`       | Weaviate hostname (weaviate backend only)        | `localhost`  |
| `WEAVIATE_PORT`       | Weaviate HTTP port (weaviate backend only)       | `8080`       |
| `WEAVIATE_GRPC_PORT`  | Weaviate gRPC port (weaviate backend only)       | `50051`      |

> **Billing note for `ANTHROPIC_API_KEY`:** Each deployer brings their own Anthropic key. The project does not supply or cover Anthropic API usage on anyone's behalf — whoever sets that env var is the one whose account gets billed if the Groq path fails and the Anthropic fallback fires.

Summarization follows a three-step fallback chain: Groq first (free), Anthropic only if Groq fails (paid, deployer's own key), plain truncation to 600 characters if both fail.

## Architecture

- **Storage**: SQLite, single file, zero infrastructure (default) — brute-force cosine search over float32 blobs, sub-millisecond at this scale. Weaviate available as a legacy backend.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (ONNX backend), computed from raw text
- **Summarization**: Groq (Llama 3.1 8B, primary) / Anthropic Claude (paid fallback), runs asynchronously after insert
- **Framework**: FastAPI + FastMCP

**Tunable constants** (hardcoded in `sqlite_store.py`/`store.py`, not env vars):

| Constant                       | Default | Purpose |
|-------------------------------|---------|---------|
| `RECENCY_DECAY_WEIGHT`        | `0.01`  | Score penalty per day of age during retrieval reranking |
| `DUPLICATE_DISTANCE_THRESHOLD` | `0.05`  | Max cosine distance to consider a new memory a duplicate of an existing one |

## Project Structure

```
mcp-memory-server/
├── memory_server/
│   ├── __init__.py
│   ├── main.py
│   ├── sqlite_store.py   # default backend (zero infra)
│   ├── store.py          # legacy Weaviate backend
│   └── summarize.py
├── docker-compose.yml    # only needed for the Weaviate backend
├── requirements.txt
├── diagnose.py
├── test_e2e.py
├── eval_retrieval.py
└── README.md
```

## License

MIT License
```
