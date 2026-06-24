"""Retrieval evaluation harness for the MCP Memory Server.

Stores a set of hardcoded facts into a clean test collection, runs queries
with known expected top hits, and reports hit-rate.

Run: python eval_retrieval.py
Requires: server running at localhost:8000 with Weaviate up.
"""

import sys
import time
import requests

BASE_URL = "http://localhost:8000"

EVAL_SESSION = f"eval-retrieval-{int(time.time())}"

MEMORIES = [
    {
        "id_tag": "python-gc",
        "text": "Python uses reference counting combined with a cyclic garbage collector to manage memory automatically.",
        "category": "python",
    },
    {
        "id_tag": "python-gil",
        "text": "The Global Interpreter Lock (GIL) in CPython prevents multiple native threads from executing Python bytecodes simultaneously.",
        "category": "python",
    },
    {
        "id_tag": "rust-ownership",
        "text": "Rust's ownership model ensures memory safety at compile time without needing a garbage collector.",
        "category": "rust",
    },
    {
        "id_tag": "rust-borrow",
        "text": "Rust's borrow checker enforces that references do not outlive the data they point to, preventing dangling pointers.",
        "category": "rust",
    },
    {
        "id_tag": "docker-containers",
        "text": "Docker containers package applications with their dependencies into isolated units that run consistently across environments.",
        "category": "devops",
    },
    {
        "id_tag": "k8s-orchestration",
        "text": "Kubernetes orchestrates containerized workloads, handling scaling, load balancing, and self-healing of failed pods.",
        "category": "devops",
    },
    {
        "id_tag": "postgres-mvcc",
        "text": "PostgreSQL implements Multi-Version Concurrency Control (MVCC) to handle concurrent transactions without read locks.",
        "category": "database",
    },
    {
        "id_tag": "redis-cache",
        "text": "Redis is an in-memory key-value store often used as a cache, message broker, or session store with sub-millisecond latency.",
        "category": "database",
    },
    {
        "id_tag": "react-hooks",
        "text": "React hooks like useState and useEffect let functional components manage state and side effects without class syntax.",
        "category": "frontend",
    },
    {
        "id_tag": "css-grid",
        "text": "CSS Grid Layout provides a two-dimensional grid system for building complex responsive web page layouts.",
        "category": "frontend",
    },
    {
        "id_tag": "jwt-auth",
        "text": "JSON Web Tokens (JWT) encode claims as a JSON object signed with a secret or public/private key pair for stateless authentication.",
        "category": "security",
    },
    {
        "id_tag": "oauth2-flow",
        "text": "OAuth 2.0 authorization framework delegates user authentication to a third-party service using access tokens and refresh tokens.",
        "category": "security",
    },
    # Near-duplicates of existing entries (should be deduped)
    {
        "id_tag": "python-gc-dup",
        "text": "Python uses reference counting combined with a cyclic garbage collector to automatically manage memory.",
        "category": "python",
    },
    {
        "id_tag": "docker-containers-dup",
        "text": "Docker containers package applications with their dependencies into isolated units that run consistently across different environments.",
        "category": "devops",
    },
    # Distinct but related entries
    {
        "id_tag": "graphql-api",
        "text": "GraphQL lets clients request exactly the data they need from an API using a typed query language, reducing over-fetching.",
        "category": "api",
    },
    {
        "id_tag": "rest-api",
        "text": "RESTful APIs use HTTP methods (GET, POST, PUT, DELETE) on resource URLs to provide a stateless interface for web services.",
        "category": "api",
    },
    {
        "id_tag": "git-branching",
        "text": "Git branches allow developers to work on features in isolation and merge changes back through pull requests.",
        "category": "tooling",
    },
    {
        "id_tag": "ci-cd-pipeline",
        "text": "CI/CD pipelines automate building, testing, and deploying code on every commit to catch issues early and ship faster.",
        "category": "devops",
    },
]

QUERIES = [
    {
        "query": "How does Python manage memory?",
        "expected_tag": "python-gc",
    },
    {
        "query": "What prevents Python threads from running in parallel?",
        "expected_tag": "python-gil",
    },
    {
        "query": "How does Rust guarantee memory safety?",
        "expected_tag": "rust-ownership",
    },
    {
        "query": "dangling pointer prevention in Rust",
        "expected_tag": "rust-borrow",
    },
    {
        "query": "application packaging and environment consistency",
        "expected_tag": "docker-containers",
    },
    {
        "query": "container orchestration autoscaling",
        "expected_tag": "k8s-orchestration",
    },
    {
        "query": "concurrent transactions without locking in Postgres",
        "expected_tag": "postgres-mvcc",
    },
    {
        "query": "fast in-memory caching with low latency",
        "expected_tag": "redis-cache",
    },
    {
        "query": "functional component state management in React",
        "expected_tag": "react-hooks",
    },
    {
        "query": "two-dimensional page layout system CSS",
        "expected_tag": "css-grid",
    },
    {
        "query": "stateless authentication with signed tokens",
        "expected_tag": "jwt-auth",
    },
    {
        "query": "delegated login via third-party with access tokens",
        "expected_tag": "oauth2-flow",
    },
    {
        "query": "query language that prevents over-fetching from APIs",
        "expected_tag": "graphql-api",
    },
    {
        "query": "HTTP methods GET POST on resource URLs",
        "expected_tag": "rest-api",
    },
    {
        "query": "automated build test deploy on every commit",
        "expected_tag": "ci-cd-pipeline",
    },
]

ADVERSARIAL_QUERIES = [
    {
        "query": "why can't you run Python code on all your CPU cores at once",
        "expected_tag": "python-gil",
        "note": "Zero shared vocabulary with stored fact (no 'GIL', 'interpreter', 'threads', 'bytecodes')",
    },
    {
        "query": "that database trick where readers never block writers",
        "expected_tag": "postgres-mvcc",
        "note": "Casual phrasing, no technical terms from fact (no 'MVCC', 'concurrent', 'transactions')",
    },
    {
        "query": "I need to call a backend but only get back exactly the fields I ask for",
        "expected_tag": "graphql-api",
        "note": "Developer perspective, different framing than the stored fact",
    },
    {
        "query": "how do you keep bad pointers from crashing your program",
        "expected_tag": "rust-borrow",
        "note": "Ambiguous: rust-ownership (memory safety) and rust-borrow (dangling pointers) both valid",
    },
    {
        "query": "what was that thing where you let another site verify who the user is",
        "expected_tag": "oauth2-flow",
        "note": "Ambiguous: jwt-auth (signed tokens for auth) and oauth2-flow (delegated auth) both plausible",
    },
]


def resolve_tag(result_id, tag_to_id):
    for tag, mid in tag_to_id.items():
        if mid == result_id:
            return tag
    return "?"


def run_query_set(label, queries, tag_to_id, session_id, verbose_misses=False):
    print(f"\n--- {label} ({len(queries)} queries, top_k=3) ---\n")
    hits = 0
    header = f"  {'Query':<55} {'Expected':<20} {'Got':<20} {'Result'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for q in queries:
        resp = requests.post(f"{BASE_URL}/retrieve", json={
            "query": q["query"],
            "session_id": session_id,
            "top_k": 3,
        })
        resp.raise_for_status()
        results = resp.json()["results"]

        expected_id = tag_to_id[q["expected_tag"]]
        result_ids = [r["id"] for r in results]
        hit = expected_id in result_ids

        if hit:
            hits += 1
            rank = result_ids.index(expected_id) + 1
            got_str = f"rank {rank}"
            status = "HIT"
        else:
            got_str = resolve_tag(results[0]["id"], tag_to_id) if results else "empty"
            status = "MISS"

        query_short = q["query"][:53]
        print(f"  {query_short:<53} {q['expected_tag']:<20} {got_str:<20} {status}")

        if not hit and verbose_misses and results:
            print(f"    note: {q.get('note', '')}")
            print(f"    expected_id: {expected_id}")
            for i, r in enumerate(results[:3]):
                rtag = resolve_tag(r["id"], tag_to_id)
                print(f"    result[{i}]: tag={rtag}  id={r['id']}  distance={r['distance']:.4f}")

    return hits, len(queries)


def main():
    print("=" * 70)
    print("MCP Memory Server — Retrieval Evaluation")
    print("=" * 70)

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.ConnectionError:
        print(f"\nERROR: Cannot connect to server at {BASE_URL}")
        sys.exit(1)

    tag_to_id = {}
    print(f"\nStoring {len(MEMORIES)} memories (session={EVAL_SESSION})...")
    for mem in MEMORIES:
        resp = requests.post(f"{BASE_URL}/store", json={
            "text": mem["text"],
            "category": mem["category"],
            "source": "eval",
            "session_id": EVAL_SESSION,
        })
        resp.raise_for_status()
        memory_id = resp.json()["id"]
        tag_to_id[mem["id_tag"]] = memory_id

    dedup_pairs = [
        ("python-gc", "python-gc-dup"),
        ("docker-containers", "docker-containers-dup"),
    ]
    dedup_pass = 0
    print("\n--- Deduplication Check ---")
    for orig, dup in dedup_pairs:
        same = tag_to_id[orig] == tag_to_id[dup]
        status = "DEDUPED" if same else "NOT DEDUPED"
        print(f"  {orig} vs {dup}: {status}")
        if same:
            dedup_pass += 1

    print("\nWaiting 5s for indexing...")
    time.sleep(5)

    orig_hits, orig_total = run_query_set(
        "Original Queries", QUERIES, tag_to_id, EVAL_SESSION
    )
    adv_hits, adv_total = run_query_set(
        "Adversarial Queries", ADVERSARIAL_QUERIES, tag_to_id, EVAL_SESSION,
        verbose_misses=True,
    )

    total_hits = orig_hits + adv_hits
    total_queries = orig_total + adv_total

    print(f"\n{'=' * 70}")
    print(f"Original:    {orig_hits}/{orig_total} = {orig_hits/orig_total*100:.1f}%")
    print(f"Adversarial: {adv_hits}/{adv_total} = {adv_hits/adv_total*100:.1f}%")
    print(f"Combined:    {total_hits}/{total_queries} = {total_hits/total_queries*100:.1f}%")
    print(f"Dedup:       {dedup_pass}/{len(dedup_pairs)} near-duplicates correctly merged")
    print(f"{'=' * 70}")

    return 0 if (orig_hits / orig_total) >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
