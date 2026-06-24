# MCP Memory Server

A semantic memory layer for AI coding agents. It allows agents to store and retrieve memories with automatic summarization, reducing token usage during long coding sessions.

## Features

- **Automatic Summarization** — Stores both the original text and a concise summary (Groq primary, Anthropic fallback).
- **Semantic Retrieval** — Uses vector embeddings based on summaries for better relevance.
- **MCP Native Support** — Exposes `store_memory` and `retrieve_memory` as MCP tools.
- **REST API** — Simple HTTP endpoints for testing and integration (`/store`, `/retrieve`, `/health`).
- **Session & Project Support** — Optional `session_id` and `project_id` fields for organizing memories.
- **Efficient** — Uses ONNX backend with quantized embeddings for lower memory usage.

## Quick Start

### 1. Start Weaviate

```bash
docker compose up -d
```

### 2. Start the Server

```bash
python -m memory_server
```

The server runs at `http://localhost:8000`.

### 3. Interactive Documentation

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
    "source": "conversation"
  }'
```

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

- `store_memory(text, source, category, tags, session_id, project_id)`
- `retrieve_memory(query, top_k, session_id, project_id, category, source)`
- `delete_memory(memory_id)`

These can be called directly by any MCP-compatible agent.

## Environment Variables

| Variable              | Description                              | Default     |
|-----------------------|------------------------------------------|-------------|
| `GROQ_API_KEY`        | Groq API key for free summarization      | *(required)* |
| `ANTHROPIC_API_KEY`   | Anthropic API key (paid fallback only)   | *(optional)* |
| `WEAVIATE_HOST`       | Weaviate hostname                        | `localhost`  |
| `WEAVIATE_PORT`       | Weaviate HTTP port                       | `8080`       |
| `WEAVIATE_GRPC_PORT`  | Weaviate gRPC port                       | `50051`      |

> **Billing note for `ANTHROPIC_API_KEY`:** Each deployer brings their own Anthropic key. The project does not supply or cover Anthropic API usage on anyone's behalf — whoever sets that env var is the one whose account gets billed if the Groq path fails and the Anthropic fallback fires.

Summarization follows a three-step fallback chain: Groq first (free), Anthropic only if Groq fails (paid, deployer's own key), plain truncation to 600 characters if both fail.

## Architecture

- **Vector Database**: Weaviate (local)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (ONNX backend)
- **Summarization**: Groq (Llama 3.1 8B, primary) / Anthropic Claude (paid fallback)
- **Framework**: FastAPI + FastMCP

## Project Structure

```
mcp-memory-server/
├── memory_server/
│   ├── __init__.py
│   ├── main.py
│   └── store.py
├── docker-compose.yml
├── requirements.txt
├── diagnose.py
├── test_e2e.py
└── README.md
```

## License

MIT License
```
