from functools import lru_cache
import hashlib
from typing import Optional, Dict, Any


class SemanticQueryCache:
    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 100):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings = None  # Inject embedding function
        self.threshold = similarity_threshold
        self.max_size = max_size

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if similar query exists in cache."""
        if not self.cache:
            return None

        query_emb = self._embed(query)

        for cached_query, cached_result in self.cache.items():
            cached_emb = cached_result["embedding"]
            similarity = self._cosine_similarity(query_emb, cached_emb)

            if similarity >= self.threshold:
                print(f"✨ Cache hit! (similarity: {similarity:.3f})")
                return cached_result["result"]

        return None

    def set(self, query: str, result: Dict[str, Any]):
        """Add query result to cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))

        query_emb = self._embed(query)
        self.cache[query] = {
            "result": result,
            "embedding": query_emb
        }

    def _embed(self, text: str):
        """Generate embedding for text."""
        if not self.embeddings:
            raise ValueError("Embedding function not injected")
        return self.embeddings([text])[0]

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)