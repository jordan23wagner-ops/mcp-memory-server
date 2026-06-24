"""Weaviate-backed memory store with automatic summarization."""

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
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def connect(self):
        """Connect to Weaviate and load the embedding model."""
        logger.info("Connecting to Weaviate at localhost:8080")
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(
            self.embedding_model_name,
            backend="onnx"   # Lower memory usage
        )

        # Ensure collection exists
        self._ensure_collection()
        logger.info("MemoryStore ready")

    def _ensure_collection(self):
        """Create the Memory collection if it doesn't exist."""
        collection_name = "Memory"

        if not self.client.collections.exists(collection_name):
            self.client.collections.create(
                name=collection_name,
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                properties=[
                    weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="summary", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="source", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="category", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="tags", data_type=weaviate.classes.config.DataType.TEXT_ARRAY),
                    weaviate.classes.config.Property(name="original_length", data_type=weaviate.classes.config.DataType.INT),
                    weaviate.classes.config.Property(name="created_at", data_type=weaviate.classes.config.DataType.DATE),
                ]
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' already exists")

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
            return text[:600]  # Fallback

    def store(self, text: str, metadata: dict = None) -> str:
        """Store text with automatic summarization."""
        if metadata is None:
            metadata = {}

        # Generate summary
        summary = self.summarize(text)
        metadata["summary"] = summary
        metadata["original_length"] = len(text)
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        # Create embedding from summary
        embedding = self.model.encode(summary).tolist()

        data_object = {
            "text": text,
            "summary": summary,
            **metadata
        }

        collection = self.client.collections.get("Memory")
        result = collection.data.insert(
            properties=data_object,
            vector=embedding
        )
        return str(result.uuid)

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve most relevant memories using semantic search."""
        embedding = self.model.encode(query).tolist()

        collection = self.client.collections.get("Memory")
        response = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_properties=["text", "summary", "source", "category", "tags", "original_length", "created_at"]
        )

        return [obj.properties for obj in response.objects]

    def close(self):
        if self.client:
            self.client.close()
