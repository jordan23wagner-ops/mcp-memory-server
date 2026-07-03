"""SQLite-backed memory store — zero infrastructure, single file on disk.

Vector search is brute-force cosine similarity over float32 blobs via numpy.
At this store's scale (thousands of memories, 384-dim embeddings) a full scan
is sub-millisecond and avoids any dependency on SQLite loadable extensions.
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from memory_server.summarize import summarize as _summarize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECENCY_DECAY_WEIGHT = 0.01
DUPLICATE_DISTANCE_THRESHOLD = 0.05

DEFAULT_DB_PATH = Path.home() / ".memory_server" / "memories.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    session_id TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL DEFAULT '',
    original_length INTEGER,
    created_at TEXT,
    embedding BLOB NOT NULL
);
"""


class SQLiteMemoryStore:
    """Drop-in replacement for the Weaviate MemoryStore with the same interface."""

    def __init__(self):
        self.db_path = None
        self.model = None
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def connect(self):
        self.db_path = Path(os.environ.get("MEMORY_DB_PATH", DEFAULT_DB_PATH))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Opening SQLite memory store at {self.db_path}")

        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(
            self.embedding_model_name,
            backend="onnx"
        )
        logger.info("SQLiteMemoryStore ready")

    def _conn(self) -> sqlite3.Connection:
        # A connection per operation keeps this safe across FastAPI worker
        # threads and the summarization executor without shared-state locking.
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _embed(self, text: str) -> np.ndarray:
        vec = np.asarray(self.model.encode(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _nearest(embedding: np.ndarray, rows) -> list[tuple[float, sqlite3.Row]]:
        """Return (cosine_distance, row) pairs sorted nearest-first."""
        if not rows:
            return []
        matrix = np.stack([
            np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
        ])
        similarities = matrix @ embedding
        distances = 1.0 - similarities
        order = np.argsort(distances)
        return [(float(distances[i]), rows[i]) for i in order]

    def summarize(self, text: str) -> str:
        return _summarize(text)

    def store(self, text: str, metadata: dict = None) -> str:
        if metadata is None:
            metadata = {}

        created_at = datetime.now(timezone.utc).isoformat()
        embedding = self._embed(text)

        with self._conn() as conn:
            rows = conn.execute("SELECT id, embedding FROM memories").fetchall()
            nearest = self._nearest(embedding, rows)

            if nearest and nearest[0][0] <= DUPLICATE_DISTANCE_THRESHOLD:
                existing_id = nearest[0][1]["id"]
                logger.info(
                    f"Duplicate detected, updating existing memory {existing_id} instead of inserting"
                )
                conn.execute(
                    """UPDATE memories SET text=?, summary='', source=?, category=?, tags=?,
                       session_id=?, project_id=?, original_length=?, created_at=?, embedding=?
                       WHERE id=?""",
                    (
                        text,
                        metadata.get("source", ""),
                        metadata.get("category", ""),
                        json.dumps(metadata.get("tags", [])),
                        metadata.get("session_id") or "",
                        metadata.get("project_id") or "",
                        len(text),
                        created_at,
                        embedding.tobytes(),
                        existing_id,
                    ),
                )
                return existing_id

            memory_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO memories
                   (id, text, summary, source, category, tags, session_id, project_id,
                    original_length, created_at, embedding)
                   VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    text,
                    metadata.get("source", ""),
                    metadata.get("category", ""),
                    json.dumps(metadata.get("tags", [])),
                    metadata.get("session_id") or "",
                    metadata.get("project_id") or "",
                    len(text),
                    created_at,
                    embedding.tobytes(),
                ),
            )
            return memory_id

    def finalize_summary(self, memory_id: str, text: str):
        summary = self.summarize(text)
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET summary=? WHERE id=?", (summary, memory_id)
            )
        logger.info(f"Summary finalized for memory {memory_id}")

    def delete(self, memory_id: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return cursor.rowcount > 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
        project_id: str | None = None,
        category: str | None = None,
        source: str | None = None,
        include_full_text: bool = False,
    ):
        embedding = self._embed(query)

        where = []
        params = []
        for column, value in (
            ("session_id", session_id),
            ("project_id", project_id),
            ("category", category),
            ("source", source),
        ):
            if value:
                where.append(f"{column} = ?")
                params.append(value)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where_sql}", params
            ).fetchall()

        fetch_limit = min(top_k * 3, 30)
        nearest = self._nearest(embedding, rows)[:fetch_limit]

        now = datetime.now(timezone.utc)
        candidates = []
        for distance, row in nearest:
            vector_similarity = 1.0 - distance

            age_days = 0.0
            if row["created_at"]:
                created_dt = datetime.fromisoformat(row["created_at"])
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age_days = (now - created_dt).total_seconds() / 86400.0

            combined_score = vector_similarity - (RECENCY_DECAY_WEIGHT * age_days)

            entry = {
                "id": row["id"],
                "summary": row["summary"],
                "source": row["source"],
                "category": row["category"],
                "tags": json.loads(row["tags"]),
                "session_id": row["session_id"],
                "project_id": row["project_id"],
                "original_length": row["original_length"],
                "created_at": row["created_at"],
                "distance": distance,
            }
            if include_full_text:
                entry["text"] = row["text"]
            candidates.append((combined_score, entry))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:top_k]]

    def close(self):
        pass
