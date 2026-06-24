"""FastAPI + MCP server for semantic memory storage and retrieval.

Features:
- MCP tools for agents (store_memory, retrieve_memory)
- REST endpoints for easy testing (/store, /retrieve)
- Automatic summarization on storage
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
    session_id: str = "",
    project_id: str = "",
) -> dict:
    """Store text as a memory with automatic summarization."""
    metadata = {
        "source": source,
        "category": category,
        "tags": tags or [],
        "session_id": session_id,
        "project_id": project_id,
    }
    memory_id = store.store(text, metadata)
    return {"id": memory_id, "status": "stored"}


@mcp.tool()
def retrieve_memory(
    query: str,
    top_k: int = 5,
    session_id: str = None,
    project_id: str = None,
) -> list[dict]:
    """Retrieve most relevant memories (optionally filtered by session/project)."""
    return store.retrieve(query, top_k=top_k, session_id=session_id, project_id=project_id)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    store.connect()
    yield
    store.close()


app = FastAPI(
    title="MCP Memory Server",
    description="Semantic memory layer for AI coding agents",
    version="0.3.0",
    lifespan=lifespan,
)

# Mount MCP endpoint
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/health")
async def health():
    return {"status": "ok", "service": "memory-mcp-server"}


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
class StoreRequest(BaseModel):
    text: str
    source: str = ""
    category: str = ""
    tags: list[str] = []
    session_id: str = ""
    project_id: str = ""


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str = None
    project_id: str = None


@app.post("/store")
async def store_memory_rest(req: StoreRequest):
    metadata = {
        "source": req.source,
        "category": req.category,
        "tags": req.tags,
        "session_id": req.session_id,
        "project_id": req.project_id,
    }
    memory_id = store.store(req.text, metadata)
    return {"id": memory_id, "status": "stored"}


@app.post("/retrieve")
async def retrieve_memory_rest(req: RetrieveRequest):
    results = store.retrieve(
        req.query,
        top_k=req.top_k,
        session_id=req.session_id,
        project_id=req.project_id,
    )
    return {"results": results}


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("memory_server.main:app", host="0.0.0.0", port=8000, reload=True)
