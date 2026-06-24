"""Weaviate-backed memory store with automatic summarization."""

import logging
import os
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
        host = os.environ.get("WEAVIATE_HOST", "localhost")
        port = int(os.environ.get("WEAVIATE_PORT", "8080"))
        grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

        logger.info(f"Connecting to Weaviate at {host}:{port}")
        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port
        )

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(
            self.embedding_model_name,
            backend="onnx"
        )

        self._ensure_collection()
        logger.info("MemoryStore ready")

    def _ensure_collection(self):
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
                    weaviate.classes.config.Property(name="session_id", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="project_id", data_type=weaviate.classes.config.DataType.TEXT),
                    weaviate.classes.config.Property(name="original_length", data_type=weaviate.classes.config.DataType.INT),
                    weaviate.classes.config.Property(name="created_at", data_type=weaviate.classes.config.DataType.DATE),
                ]
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    def summarize(self, text: str) -> str:
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
        if metadata is None:
            metadata = {}

        summary = self.summarize(text)
        metadata["summary"] = summary
        metadata["original_length"] = len(text)
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        metadata["session_id"] = metadata.get("session_id") or ""
        metadata["project_id"] = metadata.get("project_id") or ""

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
        return str(result)

    def delete(self, memory_id: str) -> bool:
        collection = self.client.collections.get("Memory")
        try:
            collection.data.delete_by_id(memory_id)
            return True
        except Exception as e:
            logger.warning(f"Delete failed for {memory_id}: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
        project_id: str | None = None,
        category: str | None = None,
        source: str | None = None,
    ):
        embedding = self.model.encode(query).tolist()

        collection = self.client.collections.get("Memory")

        filters = []
        if session_id:
            filters.append(weaviate.classes.query.Filter.by_property("session_id").equal(session_id))
        if project_id:
            filters.append(weaviate.classes.query.Filter.by_property("project_id").equal(project_id))
        if category:
            filters.append(weaviate.classes.query.Filter.by_property("category").equal(category))
        if source:
            filters.append(weaviate.classes.query.Filter.by_property("source").equal(source))

        response = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_properties=[
                "text", "summary", "source", "category",
                "tags", "session_id", "project_id",
                "original_length", "created_at"
            ],
            filters=weaviate.classes.query.Filter.all_of(filters) if filters else None
        )

        results = []
        for obj in response.objects:
            results.append({
                "id": str(obj.uuid),
                "text": obj.properties.get("text"),
                "summary": obj.properties.get("summary"),
                "source": obj.properties.get("source"),
                "category": obj.properties.get("category"),
                "tags": obj.properties.get("tags", []),
                "session_id": obj.properties.get("session_id", ""),
                "project_id": obj.properties.get("project_id", ""),
                "original_length": obj.properties.get("original_length"),
                "created_at": obj.properties.get("created_at"),
                "distance": obj.metadata.distance if obj.metadata else None
            })
        return results

    def close(self):
        if self.client:
            self.client.close()
