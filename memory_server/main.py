"""FastAPI + MCP server for semantic memory storage and retrieval.

Run directly:
    python -m memory_server.main

MCP tools exposed:
    store_memory - persist text + metadata with vector embeddings
    retrieve_memory - semantic search over stored memories

The MCP endpoint is mounted at /mcp (streamable HTTP transport).
A /health endpoint is available for liveness checks.
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP
from memory_server.store import MemoryStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
store = MemoryStore()
mcp = FastMCP("Memory Server")


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def store_memory(
    text: str,
    source: str = "",
    category: str = "",
    tags: list[str] | None = None,
) -> dict:
    """Store a piece of text as a searchable memory with semantic embedding.

    Args:
        text: The memory content to store.
        source: Where this memory came from (e.g. "conversation", "document").
        category: A label for organising memories (e.g. "project-x", "personal").
        tags: Optional list of tags for finer-grained filtering.

    Returns:
        A dict with the new memory's UUID and confirmation status.
    """
    metadata = {"source": source, "category": category, "tags": tags or []}
    memory_id = store.store(text, metadata)
    return {"id": memory_id, "status": "stored"}


@mcp.tool()
def retrieve_memory(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve memories most relevant to a query using semantic search.

    Args:
        query: Natural-language search text.
        top_k: Maximum number of results to return (default 5).

    Returns:
        A list of matching memories with text, metadata, and similarity distance.
        Lower distance = higher relevance.
    """
    return store.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Temporary REST endpoints for easy testing (can be removed later)
# ---------------------------------------------------------------------------
class StoreRequest(BaseModel):
    text: str
    metadata: dict = {}


class RetrieveRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/store")
async def store_memory_rest(request: StoreRequest):
    memory_id = store.store(request.text, request.metadata)
    return {"id": memory_id, "status": "stored"}


@app.post("/retrieve")
async def retrieve_memory_rest(request: RetrieveRequest):
    results = store.retrieve(request.query, top_k=request.limit)
    return {"results": results}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the Weaviate connection alongside the FastAPI server."""
    store.connect()
    yield
    store.close()


app = FastAPI(
    title="Memory MCP Server",
    description="Semantic memory storage and retrieval via MCP tools",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount the MCP protocol handler under /mcp
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "memory-mcp-server"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("memory_server.main:app", host="0.0.0.0", port=8000, reload=True)
