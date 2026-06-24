"""Diagnostic script for the MCP Memory Server.

Checks infrastructure health and runs functional tests against the REST API.
Prints clear PASS/FAIL results for each check.

Usage: python diagnose.py
"""

import sys
import time
import requests

BASE_URL = "http://localhost:8000"
WEAVIATE_URL = "http://localhost:8080"

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


# ═══════════════════════════════════════════════════════════════════════
# Infrastructure Checks
# ═══════════════════════════════════════════════════════════════════════

def check_weaviate():
    """Verify Weaviate is reachable."""
    print("\n[1/6] Weaviate Connection")
    try:
        resp = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", timeout=5)
        check("weaviate is ready", resp.status_code == 200)
    except requests.ConnectionError:
        check("weaviate is ready", False, f"cannot connect to {WEAVIATE_URL}")


def check_server():
    """Verify the FastAPI server is reachable and healthy."""
    print("\n[2/6] Memory Server Health")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        check("server responds", resp.status_code == 200)
        check("status is ok", data.get("status") == "ok")
    except requests.ConnectionError:
        check("server responds", False, f"cannot connect to {BASE_URL}")


# ═══════════════════════════════════════════════════════════════════════
# Functional Tests
# ═══════════════════════════════════════════════════════════════════════

def check_store_and_retrieve():
    """Store a memory, then retrieve it and verify the summary was generated."""
    print("\n[3/6] Store & Retrieve")

    # Store
    resp = requests.post(f"{BASE_URL}/store", json={
        "text": "Diagnostic test: Weaviate stores vectors for semantic search.",
        "source": "diagnose",
        "category": "test",
        "session_id": "diag-session",
        "project_id": "diag-project",
    })
    data = resp.json()
    check("store returns 200", resp.status_code == 200)
    check("store returns id", "id" in data)
    check("store status is stored", data.get("status") == "stored")

    memory_id = data.get("id", "")

    time.sleep(1)

    # Retrieve
    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "semantic search vectors",
        "top_k": 3,
    })
    results = resp.json().get("results", [])
    check("retrieve returns results", len(results) > 0)

    if results:
        top = results[0]
        check("summary is populated", bool(top.get("summary")))
        check("source is preserved", top.get("source") == "diagnose")
        check("distance is present", top.get("distance") is not None)

    return memory_id


def check_filtering():
    """Verify filtering by session_id, project_id, category, and source."""
    print("\n[4/6] Retrieval Filters")

    ts = str(int(time.time()))
    sid = f"diag-filter-{ts}"

    # Store two memories with different categories/sources
    requests.post(f"{BASE_URL}/store", json={
        "text": "Alpha memory for filter testing.",
        "source": "source-a",
        "category": "cat-a",
        "session_id": sid,
        "project_id": "proj-a",
    })
    requests.post(f"{BASE_URL}/store", json={
        "text": "Beta memory for filter testing.",
        "source": "source-b",
        "category": "cat-b",
        "session_id": sid,
        "project_id": "proj-b",
    })

    time.sleep(1)

    # Filter by session_id (should get both)
    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "filter testing",
        "session_id": sid,
    })
    results = resp.json().get("results", [])
    check("session_id filter returns both", len(results) >= 2)

    # Filter by project_id (should get only one)
    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "filter testing",
        "session_id": sid,
        "project_id": "proj-a",
    })
    results = resp.json().get("results", [])
    check("project_id filter narrows results", len(results) >= 1)
    check(
        "project_id values match",
        all(r.get("project_id") == "proj-a" for r in results),
    )

    # Filter by category
    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "filter testing",
        "session_id": sid,
        "category": "cat-b",
    })
    results = resp.json().get("results", [])
    check("category filter returns results", len(results) >= 1)
    check(
        "category values match",
        all(r.get("category") == "cat-b" for r in results),
    )

    # Filter by source
    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "filter testing",
        "session_id": sid,
        "source": "source-a",
    })
    results = resp.json().get("results", [])
    check("source filter returns results", len(results) >= 1)
    check(
        "source values match",
        all(r.get("source") == "source-a" for r in results),
    )


def check_delete(memory_id: str):
    """Delete a memory by ID and verify it no longer appears in results."""
    print("\n[5/6] Delete")

    if not memory_id:
        check("have a memory_id to delete", False, "no id from earlier store")
        return

    resp = requests.delete(f"{BASE_URL}/memory/{memory_id}")
    data = resp.json()
    check("delete returns 200", resp.status_code == 200)
    check("delete status is deleted", data.get("status") == "deleted")
    check("delete id matches", data.get("id") == memory_id)

    time.sleep(1)

    # Confirm it's gone
    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "Diagnostic test Weaviate semantic search",
        "top_k": 20,
    })
    ids = [r["id"] for r in resp.json().get("results", [])]
    check("deleted memory absent from results", memory_id not in ids)


def check_summary_quality():
    """Store a long text and verify the summary is shorter than the original."""
    print("\n[6/6] Summary Quality")

    long_text = (
        "Docker containers package applications with their dependencies into "
        "standardized units. They share the host OS kernel, making them lighter "
        "than virtual machines. Docker Compose lets you define multi-container "
        "applications in a single YAML file, simplifying orchestration for local "
        "development and testing environments."
    )
    resp = requests.post(f"{BASE_URL}/store", json={
        "text": long_text,
        "category": "devops",
    })
    check("long text stored", resp.status_code == 200)

    time.sleep(1)

    resp = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "docker containers",
        "top_k": 1,
    })
    results = resp.json().get("results", [])
    if results:
        summary = results[0].get("summary", "")
        check("summary is non-empty", bool(summary))
        check("summary is shorter than original", len(summary) < len(long_text))
        check(
            "original_length matches",
            results[0].get("original_length") == len(long_text),
        )
    else:
        check("retrieve returned a result", False, "empty results")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MCP Memory Server — Diagnostics")
    print("=" * 60)

    check_weaviate()
    check_server()

    # Stop early if the server isn't reachable
    if failed > 0:
        print(f"\nAborting: fix infrastructure issues first ({failed} failed)")
        sys.exit(1)

    memory_id = check_store_and_retrieve()
    check_filtering()
    check_delete(memory_id)
    check_summary_quality()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
