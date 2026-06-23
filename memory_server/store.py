"""Weaviate-backed vector store for semantic memory.

Handles embedding generation (sentence-transformers) and CRUD operations
against a local Weaviate instance. Vectors are generated client-side so
Weaviate doesn't need any vectorizer modules installed.
"""

import logging
from datetime import datetime, timezone

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

logger = logging.getLogger(__name__)

COLLECTION_NAME = "Memory"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class MemoryStore:
    """Embeds text via sentence-transformers and stores/retrieves it in Weaviate."""

    def __init__(
        self,
        weaviate_host: str = "localhost",
        weaviate_port: int = 8080,
        weaviate_grpc_port: int = 50051,
        embedding_model: str = DEFAULT_MODEL,
    ):
        self.weaviate_host = weaviate_host
        self.weaviate_port = weaviate_port
        self.weaviate_grpc_port = weaviate_grpc_port
        self.embedding_model_name = embedding_model
        self.client: weaviate.WeaviateClient | None = None
        self.embedder: SentenceTransformer | None = None

    def connect(self) -> None:
        """Load the embedding model and connect to Weaviate."""
        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self.embedder = SentenceTransformer(self.embedding_model_name)

        logger.info(
            "Connecting to Weaviate at %s:%d", self.weaviate_host, self.weaviate_port
        )
        self.client = weaviate.connect_to_local(
            host=self.weaviate_host,
            port=self.weaviate_port,
            grpc_port=self.weaviate_grpc_port,
        )
        self._ensure_collection()
        logger.info("MemoryStore ready")

    def _ensure_collection(self) -> None:
        """Create the Memory collection if it doesn't already exist."""
        if self.client.collections.exists(COLLECTION_NAME):
            logger.info("Collection '%s' already exists", COLLECTION_NAME)
            return

        # Vectorizer.none() because we supply our own embeddings
        self.client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
                Property(name="created_at", data_type=DataType.TEXT),
            ],
        )
        logger.info("Created collection '%s'", COLLECTION_NAME)

    def _embed(self, text: str) -> list[float]:
        """Generate a dense vector for the given text."""
        return self.embedder.encode(text).tolist()

    def store(self, text: str, metadata: dict | None = None) -> str:
        """Embed and store a memory. Returns the new object's UUID."""
        metadata = metadata or {}
        vector = self._embed(text)

        properties = {
            "text": text,
            "source": metadata.get("source", ""),
            "category": metadata.get("category", ""),
            "tags": metadata.get("tags", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        collection = self.client.collections.get(COLLECTION_NAME)
        obj_uuid = collection.data.insert(properties=properties, vector=vector)

        logger.info("Stored memory %s (%d chars)", obj_uuid, len(text))
        return str(obj_uuid)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Find the top-k memories closest to the query by cosine similarity."""
        query_vector = self._embed(query)
        collection = self.client.collections.get(COLLECTION_NAME)

        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        memories = []
        for obj in results.objects:
            memories.append(
                {
                    "id": str(obj.uuid),
                    "text": obj.properties.get("text", ""),
                    "source": obj.properties.get("source", ""),
                    "category": obj.properties.get("category", ""),
                    "tags": obj.properties.get("tags", []),
                    "created_at": obj.properties.get("created_at", ""),
                    "distance": obj.metadata.distance,
                }
            )

        logger.info(
            "Retrieved %d memories for query (%d chars)", len(memories), len(query)
        )
        return memories

    def close(self) -> None:
        """Disconnect from Weaviate."""
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")
