"""Allow running with `python -m memory_server`."""

from memory_server.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("memory_server.main:app", host="0.0.0.0", port=8000, reload=True)
