"""Weaviate-backed vector store with automatic summarization."""

import logging
from datetime import datetime, timezone

import weaviate
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self):
        self.client = None
        self.model = None

    def connect(self):
        """Connect to Weaviate and load embedding model."""
        logger.info("Connecting to Weaviate at localhost:8080")
        self.client = weaviate.Client("http://localhost:8080")

        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Ensure collection exists
        if not self.client.schema.exists("Memory"):
            self.client.schema.create_class({
                "class": "Memory",
                "vectorizer": "none",
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "summary", "dataType": ["text"]},
                    {"name": "source", "dataType": ["text"]},
                    {"name": "category", "dataType": ["text"]},
                    {"name": "tags", "dataType": ["text[]"]},
                    {"name": "created_at", "dataType": ["date"]},
                ]
            })
        logger.info("MemoryStore ready")

    def summarize(self, text: str) -> str:
        """Generate a concise summary using Claude."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            prompt = f"""Summarize the following text in 2-3 sentences. Focus on the key points.

Text:
{text}

Summary:"""

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return text[:600]

    def store(self, text: str, metadata: dict = None) -> str:
        """Store text with automatic summarization."""
        if metadata is None:
            metadata = {}

        summary = self.summarize(text)
        metadata["summary"] = summary
        metadata["original_length"] = len(text)
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        embedding = self.model.encode(summary).tolist()

        data_object = {
            "text": text,
            "summary": summary,
            **metadata
        }

        result = self.client.data_object.create(
            data_object=data_object,
            class_name="Memory",
            vector=embedding
        )
        return str(result)

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve most relevant memories."""
        embedding = self.model.encode(query).tolist()

        results = self.client.query.get("Memory", [
            "text", "summary", "source", "category", "tags", "created_at"
        ]).with_near_vector({
            "vector": embedding
        }).with_limit(top_k).do()

        return results.get("data", {}).get("Get", {}).get("Memory", [])

    def close(self):
        if self.client:
            self.client.close()
