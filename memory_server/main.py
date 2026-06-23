"""FastAPI + MCP server for semantic memory storage and retrieval."""

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared objects
# ---------------------------------------------------------------------------
store = MemoryStore()
mcp = FastMCP("Memory Server")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def store_memory(
    text: str,
    source: str = "",
    category: str = "",
    tags: list[str] | None = None,
) -> dict:
    """Store text as a searchable memory with semantic embedding."""
    metadata = {"source": source, "category": category, "tags": tags or []}
    memory_id = store.store(text, metadata)
    return {"id": memory_id, "status": "stored"}


@mcp.tool()
def retrieve_memory(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve most relevant memories using semantic search."""
    return store.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# FastAPI App + REST Endpoints (for easy testing)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    store.connect()
    yield
    store.close()


app = FastAPI(title="MCP Memory Server", version="0.1.0", lifespan=lifespan)

# Mount MCP
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/health")
async def health():
    return {"status": "ok"}


# Temporary REST endpoints (keep for now)
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
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("memory_server.main:app", host="0.0.0.0", port=8000, reload=True)
