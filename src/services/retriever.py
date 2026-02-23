from typing import List
from sentence_transformers import SentenceTransformer
from src.services.vector_store import VectorStore
from src.models.schema import RetrievalResult, FilingChunk

class Retriever:
    """
    Orchestrates the retrieval of documents using hybrid search strategies.
    """
    def __init__(self):
        # Load a lightweight efficient model. 
        # In prod, this might be an API call to OpenAI or a service.
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = VectorStore()

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.encoder.encode(texts).tolist()

    def index_filing(self, chunks: List[FilingChunk]):
        """
        Embed and store filing chunks.
        """
        if not chunks:
            return
            
        texts = [f"{c.section_name}: {c.content}" for c in chunks]
        embeddings = self.encode(texts)
        self.vector_store.upsert_chunks(chunks, embeddings)

    def search(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        """
        Semantic search for the query.
        """
        query_vector = self.encode([query])[0]
        
        # Perform retrieval
        points = self.vector_store.search(query_vector=query_vector, limit=limit)
        
        results = []
        for i, point in enumerate(points):
            # Reconstruct FilingChunk from payload
            chunk = FilingChunk(**point.payload)
            
            results.append(RetrievalResult(
                chunk=chunk,
                score=point.score,
                source="dense",
                rank=i + 1
            ))
            
        return results
