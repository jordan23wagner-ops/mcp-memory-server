# mcp-memory-server

A persistent semantic memory layer for AI coding agents, built with FastAPI and the Model Context Protocol (MCP).

Agents connect over MCP and use two tools -- `store_memory` to save text with vector embeddings, and `retrieve_memory` to search by meaning. Weaviate handles vector storage; sentence-transformers generates embeddings client-side.

## Architecture

```
MCP Client (agent)
    |
    |  streamable HTTP  (/mcp)
    v
FastAPI + FastMCP
    |
    |-- sentence-transformers  (all-MiniLM-L6-v2)
    |-- Weaviate               (local, no vectorizer modules)
```

## Prerequisites

- Python 3.11+
- Docker (for Weaviate)
- `gh` CLI (optional, for creating the GitHub repo)

## Quickstart

```bash
# 1. Start Weaviate
docker compose up -d

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt

# 3. Run the server
python -m memory_server
```

The server starts on `http://localhost:8000`.

| Endpoint    | Purpose                              |
|-------------|--------------------------------------|
| `/mcp`      | MCP streamable HTTP transport        |
| `/health`   | Liveness check                       |
| `/docs`     | Interactive API docs (Swagger UI)    |

## MCP Tools

### `store_memory`

Save text with an auto-generated embedding.

| Parameter  | Type       | Default | Description                        |
|------------|------------|---------|------------------------------------|
| `text`     | `str`      | --      | The memory content (required)      |
| `source`   | `str`      | `""`    | Origin label (e.g. "conversation") |
| `category` | `str`      | `""`    | Grouping label                     |
| `tags`     | `list[str]`| `[]`    | Finer-grained labels               |

Returns `{ "id": "<uuid>", "status": "stored" }`.

### `retrieve_memory`

Semantic search over stored memories.

| Parameter | Type  | Default | Description                  |
|-----------|-------|---------|------------------------------|
| `query`   | `str` | --      | Natural-language search text |
| `top_k`   | `int` | `5`     | Max results to return        |

Returns a list of matches with `text`, metadata fields, and a `distance` score (lower = more relevant).

## Project Structure

```
mcp-memory-server/
  memory_server/
    __init__.py
    __main__.py       # python -m entrypoint
    main.py           # FastAPI app + MCP tool definitions
    store.py          # Weaviate client + embedding logic
  docker-compose.yml  # Local Weaviate instance
  requirements.txt
```

## License

MIT
