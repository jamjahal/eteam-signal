import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.core.config import settings
from src.core.logger import get_logger
from src.models.schema import FilingChunk

log = get_logger(__name__)

class VectorStore:
    """
    Wrapper around Qdrant for storing and retrieving SEC filing chunks.
    """
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = 384 # Default for all-MiniLM-L6-v2
        
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Create collection if it doesn't exist.
        """
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            log.info("Creating Qdrant collection", name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def upsert_chunks(self, chunks: List[FilingChunk], embeddings: List[List[float]]):
        """
        Upload chunks + embeddings to Qdrant.
        """
        if not chunks:
            return

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()), # or use chunk.id
                vector=embedding,
                payload=chunk.model_dump(mode='json')
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        log.info("Upserted chunks", count=len(points))

    def search(self, query_vector: List[float], limit: int = 10,
               filter_conditions: Optional[Dict] = None) -> List[models.ScoredPoint]:
        """
        Perform dense vector search via query_points (qdrant-client >= 1.12).
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return response.points
