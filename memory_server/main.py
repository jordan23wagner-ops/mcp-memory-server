"""FastAPI + MCP server for semantic memory storage and retrieval.

Features:
- MCP tools for agents (store_memory, retrieve_memory, delete_memory)
- REST endpoints for easy testing (/store, /retrieve, /memory/{id})
- Automatic summarization on storage
- Filtering by session_id, project_id, category, source
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI
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
mcp = FastMCP("Memory Server", streamable_http_path="/")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def store_memory(
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
    asyncio.get_event_loop().run_in_executor(
        None, store.finalize_summary, memory_id, text
    )
    return {"id": memory_id, "status": "stored"}


@mcp.tool()
def retrieve_memory(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
    project_id: str | None = None,
    category: str | None = None,
    source: str | None = None,
    include_full_text: bool = False,
) -> list[dict]:
    """Retrieve most relevant memories (optionally filtered by session/project/category/source)."""
    return store.retrieve(
        query,
        top_k=top_k,
        session_id=session_id,
        project_id=project_id,
        category=category,
        source=source,
        include_full_text=include_full_text,
    )


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Delete a memory by its ID."""
    success = store.delete(memory_id)
    if success:
        return {"id": memory_id, "status": "deleted"}
    return {"id": memory_id, "status": "not_found"}


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    store.connect()
    async with mcp.session_manager.run():
        yield
    store.close()


app = FastAPI(
    title="MCP Memory Server",
    description="Semantic memory layer for AI coding agents",
    version="0.4.0",
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
    session_id: str | None = ""
    project_id: str | None = ""


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str | None = None
    project_id: str | None = None
    category: str | None = None
    source: str | None = None
    include_full_text: bool = False


@app.post("/store")
async def store_memory_rest(req: StoreRequest, background_tasks: BackgroundTasks):
    metadata = {
        "source": req.source,
        "category": req.category,
        "tags": req.tags,
        "session_id": req.session_id,
        "project_id": req.project_id,
    }
    memory_id = store.store(req.text, metadata)
    background_tasks.add_task(store.finalize_summary, memory_id, req.text)
    return {"id": memory_id, "status": "stored"}


@app.post("/retrieve")
async def retrieve_memory_rest(req: RetrieveRequest):
    results = store.retrieve(
        req.query,
        top_k=req.top_k,
        session_id=req.session_id,
        project_id=req.project_id,
        category=req.category,
        source=req.source,
        include_full_text=req.include_full_text,
    )
    return {"results": results}


@app.delete("/memory/{memory_id}")
async def delete_memory_rest(memory_id: str):
    success = store.delete(memory_id)
    if success:
        return {"id": memory_id, "status": "deleted"}
    return {"id": memory_id, "status": "not_found"}


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("memory_server.main:app", host="0.0.0.0", port=8000, reload=True)
