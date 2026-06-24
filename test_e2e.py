"""End-to-end test suite for the MCP Memory Server REST API.

Requires the server to be running at http://localhost:8000 with Weaviate up.
Run: python test_e2e.py
"""

import sys
import time
import requests

BASE_URL = "http://localhost:8000"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  -- {detail}")


def store_memory(text: str, **kwargs) -> dict:
    resp = requests.post(f"{BASE_URL}/store", json={"text": text, **kwargs})
    resp.raise_for_status()
    return resp.json()


def retrieve_memories(query: str, **kwargs) -> list[dict]:
    resp = requests.post(f"{BASE_URL}/retrieve", json={"query": query, **kwargs})
    resp.raise_for_status()
    return resp.json()["results"]


def delete_memory(memory_id: str) -> dict:
    resp = requests.delete(f"{BASE_URL}/memory/{memory_id}")
    resp.raise_for_status()
    return resp.json()


# ── Health Check ──────────────────────────────────────────────────────
def test_health():
    print("\n[Health Check]")
    resp = requests.get(f"{BASE_URL}/health")
    check("server is healthy", resp.status_code == 200)
    check("status is ok", resp.json().get("status") == "ok")


# ── Store & Retrieve (no filters) ────────────────────────────────────
def test_store_and_retrieve_basic():
    print("\n[Store & Retrieve — basic]")
    result = store_memory(
        text="Python decorators allow you to modify functions using the @syntax.",
        source="docs",
        category="python",
    )
    check("store returns id", "id" in result)
    check("status is stored", result.get("status") == "stored")

    time.sleep(1)

    results = retrieve_memories("python decorators", top_k=3)
    check("retrieve returns results", len(results) > 0)

    match = results[0]
    check("result has summary", bool(match.get("summary")))
    check("text excluded by default", "text" not in match)
    check("result has distance", match.get("distance") is not None)

    results_full = retrieve_memories("python decorators", top_k=3, include_full_text=True)
    match_full = results_full[0]
    check("include_full_text returns text", "decorator" in match_full.get("text", "").lower())


# ── Store with session_id and project_id ──────────────────────────────
def test_store_with_session_and_project():
    print("\n[Store & Retrieve — session_id / project_id]")
    sid = f"test-session-{int(time.time())}"
    pid = f"test-project-{int(time.time())}"

    store_memory(
        text="Weaviate uses HNSW for approximate nearest neighbor search.",
        source="research",
        category="database",
        session_id=sid,
        project_id=pid,
    )
    store_memory(
        text="FastAPI supports automatic OpenAPI documentation.",
        source="docs",
        category="framework",
        session_id=sid,
    )
    store_memory(
        text="Unrelated memory about cooking pasta al dente.",
        source="random",
        category="food",
    )

    time.sleep(1)

    # Filter by session_id
    results = retrieve_memories("nearest neighbor search", session_id=sid)
    check("session filter returns results", len(results) > 0)
    check(
        "all results match session_id",
        all(r.get("session_id") == sid for r in results),
    )

    # Filter by project_id
    results = retrieve_memories("nearest neighbor", project_id=pid)
    check("project filter returns results", len(results) > 0)
    check(
        "all results match project_id",
        all(r.get("project_id") == pid for r in results),
    )


# ── Filter by category and source ────────────────────────────────────
def test_filter_by_category_and_source():
    print("\n[Retrieve — category / source filters]")
    ts = str(int(time.time()))
    sid = f"filter-test-{ts}"

    store_memory(
        text="Rust ownership model prevents data races at compile time.",
        source="tutorial",
        category="rust",
        session_id=sid,
    )
    store_memory(
        text="Go goroutines are lightweight concurrent threads.",
        source="blog",
        category="golang",
        session_id=sid,
    )

    time.sleep(1)

    results = retrieve_memories("concurrency", session_id=sid, category="rust")
    check("category filter returns results", len(results) > 0)
    check(
        "category filter is correct",
        all(r.get("category") == "rust" for r in results),
    )

    results = retrieve_memories("concurrency", session_id=sid, source="blog")
    check("source filter returns results", len(results) > 0)
    check(
        "source filter is correct",
        all(r.get("source") == "blog" for r in results),
    )


# ── Delete ────────────────────────────────────────────────────────────
def test_delete():
    print("\n[Delete]")
    result = store_memory(text="This memory will be deleted shortly.")
    memory_id = result["id"]

    time.sleep(1)

    del_result = delete_memory(memory_id)
    check("delete returns deleted status", del_result.get("status") == "deleted")
    check("delete returns correct id", del_result.get("id") == memory_id)

    # Verify it no longer appears in retrieval
    time.sleep(1)
    results = retrieve_memories("memory will be deleted", top_k=10)
    ids = [r["id"] for r in results]
    check("deleted memory is gone from results", memory_id not in ids)


# ── Summary Generation ────────────────────────────────────────────────
def test_summary_generation():
    print("\n[Summary Generation]")
    long_text = (
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Instead of being explicitly programmed, "
        "these systems use algorithms to identify patterns, make decisions, and improve "
        "over time. Common approaches include supervised learning, unsupervised learning, "
        "and reinforcement learning. Applications range from image recognition and natural "
        "language processing to recommendation systems and autonomous vehicles."
    )
    result = store_memory(text=long_text, category="ai")

    time.sleep(1)

    results = retrieve_memories("machine learning", top_k=1)
    check("summary exists", bool(results[0].get("summary")))
    check(
        "summary is shorter than original",
        len(results[0].get("summary", "")) < len(long_text),
    )
    check("original_length is set", results[0].get("original_length") == len(long_text))


# ── Run All ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MCP Memory Server — End-to-End Tests")
    print("=" * 60)

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {BASE_URL}")
        print("Make sure Weaviate and the memory server are running.")
        sys.exit(1)

    test_health()
    test_store_and_retrieve_basic()
    test_store_with_session_and_project()
    test_filter_by_category_and_source()
    test_delete()
    test_summary_generation()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
