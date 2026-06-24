"""Weaviate-backed memory store with automatic summarization."""

import logging
import os
from datetime import datetime, timezone

import weaviate
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECENCY_DECAY_WEIGHT = 0.01
DUPLICATE_DISTANCE_THRESHOLD = 0.05


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
        prompt = f"Summarize the following text in 2-3 sentences. Focus on the key points.\n\nText:\n{text}\n\nSummary:"

        # 1) Try Groq (free tier)
        try:
            from groq import Groq
            groq_client = Groq()
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.info("Summarization via groq")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq summarization failed: {e}")

        # 2) Fall back to Anthropic (paid, deployer's own key)
        try:
            import anthropic
            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.info("Summarization via anthropic (fallback)")
            return message.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Anthropic summarization failed: {e}")

        # 3) Last resort: truncation
        logger.warning("Summarization via truncation (all providers failed)")
        return text[:600]

    def store(self, text: str, metadata: dict = None) -> str:
        if metadata is None:
            metadata = {}

        metadata["summary"] = ""
        metadata["original_length"] = len(text)
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        metadata["session_id"] = metadata.get("session_id") or ""
        metadata["project_id"] = metadata.get("project_id") or ""

        embedding = self.model.encode(text).tolist()

        collection = self.client.collections.get("Memory")

        existing = collection.query.near_vector(
            near_vector=embedding,
            limit=1,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            return_properties=["text"],
        )
        if (
            existing.objects
            and existing.objects[0].metadata.distance is not None
            and existing.objects[0].metadata.distance <= DUPLICATE_DISTANCE_THRESHOLD
        ):
            existing_id = str(existing.objects[0].uuid)
            logger.info(
                f"Duplicate detected, updating existing memory {existing_id} instead of inserting"
            )
            collection.data.update(
                uuid=existing_id,
                properties={
                    "text": text,
                    "summary": "",
                    **metadata,
                },
                vector=embedding,
            )
            return existing_id

        data_object = {
            "text": text,
            "summary": "",
            **metadata
        }

        result = collection.data.insert(
            properties=data_object,
            vector=embedding
        )
        return str(result)

    def finalize_summary(self, memory_id: str, text: str):
        summary = self.summarize(text)
        collection = self.client.collections.get("Memory")
        collection.data.update(
            uuid=memory_id,
            properties={"summary": summary}
        )
        logger.info(f"Summary finalized for memory {memory_id}")

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
        include_full_text: bool = False,
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

        return_properties = [
            "summary", "source", "category",
            "tags", "session_id", "project_id",
            "original_length", "created_at"
        ]
        if include_full_text:
            return_properties.insert(0, "text")

        fetch_limit = min(top_k * 3, 30)

        response = collection.query.near_vector(
            near_vector=embedding,
            limit=fetch_limit,
            return_properties=return_properties,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            filters=weaviate.classes.query.Filter.all_of(filters) if filters else None
        )

        now = datetime.now(timezone.utc)
        candidates = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata else 0.0
            vector_similarity = 1.0 - (distance or 0.0)

            created_at = obj.properties.get("created_at")
            age_days = 0.0
            if created_at:
                if isinstance(created_at, str):
                    created_dt = datetime.fromisoformat(created_at)
                else:
                    created_dt = created_at
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age_days = (now - created_dt).total_seconds() / 86400.0

            combined_score = vector_similarity - (RECENCY_DECAY_WEIGHT * age_days)

            entry = {
                "id": str(obj.uuid),
                "summary": obj.properties.get("summary"),
                "source": obj.properties.get("source"),
                "category": obj.properties.get("category"),
                "tags": obj.properties.get("tags", []),
                "session_id": obj.properties.get("session_id", ""),
                "project_id": obj.properties.get("project_id", ""),
                "original_length": obj.properties.get("original_length"),
                "created_at": obj.properties.get("created_at"),
                "distance": distance,
            }
            if include_full_text:
                entry["text"] = obj.properties.get("text")
            candidates.append((combined_score, entry))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:top_k]]

    def close(self):
        if self.client:
            self.client.close()
