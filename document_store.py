from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from hybrid_retriever import create_hybrid_retrieval_pipeline

class DocumentStore:
    """
    Encapsulates the construction and configuration of the retrieval pipeline.
    It acts as a provider for a configured retriever, hiding the complexity
    of its construction (e.g., hybrid retrieval, re-ranking).
    """
    def __init__(
        self,
        vector_store: Chroma,
        bm25_index: Optional[Any] = None,
        bm25_chunks: Optional[List[Document]] = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.bm25_chunks = bm25_chunks or []
        self._retriever: Optional[BaseRetriever] = None

    def get_retriever(self) -> BaseRetriever:
        """
        Returns a fully configured hybrid retriever instance.
        Caches the retriever instance after the first build.
        """
        if self._retriever is None:
            print("🔧 Building hybrid retriever...")
            self._retriever = create_hybrid_retrieval_pipeline(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                bm25_chunks=self.bm25_chunks,
                use_reranking=True, # This could also be a parameter
            )
        return self._retriever

    def get_embedding_function(self):
        """Provides access to the embedding function for other components like the cache."""
        return self.vector_store._embedding_function.embed_documents