from pydantic import Field
from typing import List, Optional
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from config import app_config


class HybridRetriever(BaseRetriever):
    """
    Retriever that combines dense vector search with BM25 sparse retrieval.

    Strategy:
    1. Performs MMR-based vector search for semantic similarity
    2. Supplements with BM25 for keyword matching (if results are sparse)
    3. Deduplicates based on content

    This class focuses ONLY on retrieval, not reranking.
    """

    vector_store: Chroma
    bm25_index: Optional[BM25Okapi] = None
    bm25_chunks: List[Document] = Field(default_factory=list)
    vector_k: int = app_config.retriever.vector_k
    vector_fetch_k: int = app_config.retriever.vector_fetch_k
    bm25_top_k: int = app_config.retriever.bm25_top_k
    lambda_mult: float = app_config.retriever.lambda_mult
    min_docs_before_bm25: int = app_config.retriever.min_docs_before_bm25

    def __init__(self, **data):
        super().__init__(**data)

        """
        Args:
            vector_store: ChromaDB vector store for dense retrieval
            bm25_index: Pre-built BM25 index (optional)
            bm25_chunks: Documents corresponding to BM25 index (optional)
            vector_k: Number of docs to return from vector search
            vector_fetch_k: Number of docs to fetch before MMR filtering
            bm25_top_k: Number of additional docs to fetch from BM25
            lambda_mult: MMR diversity parameter (0=diversity, 1=relevance)
            min_docs_before_bm25: Only use BM25 if vector search returns fewer than this
        """

        if (self.bm25_index is None) != (not self.bm25_chunks):
            raise ValueError("bm25_index and bm25_chunks must both be provided or both be None")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid approach.

        Returns:
            List of documents (may contain duplicates if reranking is disabled)
        """
        # Step 1: Dense vector search with MMR
        vector_docs = self._vector_search(query)

        # Step 2: Supplement with BM25 if needed
        if self.bm25_index and len(vector_docs) < self.min_docs_before_bm25:
            bm25_docs = self._bm25_search(query, vector_docs)
            return vector_docs + bm25_docs

        return vector_docs

    def _vector_search(self, query: str) -> List[Document]:
        """Perform MMR-based vector search."""
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.vector_k,
                "fetch_k": self.vector_fetch_k,
                "lambda_mult": self.lambda_mult,
            },
        )
        return retriever.invoke(query)

    def _bm25_search(self, query: str, existing_docs: List[Document]) -> List[Document]:
        """
        Perform BM25 search and return top-k documents not already retrieved.

        Args:
            query: Search query
            existing_docs: Documents already retrieved (for deduplication)

        Returns:
            List of additional documents from BM25
        """
        if not self.bm25_index or not self.bm25_chunks:
            return []

        # Tokenize query (BM25 expects pre-tokenized input)
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)

        # Find top-k documents
        k = min(self.bm25_top_k, len(scores))
        if k == 0:
            return []

        # Use argpartition for efficient top-k selection
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort the top-k by score (descending)
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Deduplicate against existing documents
        existing_content = {doc.page_content for doc in existing_docs}
        bm25_docs = []

        for idx in top_indices:
            candidate = self.bm25_chunks[idx]
            if candidate.page_content not in existing_content:
                bm25_docs.append(candidate)
                existing_content.add(candidate.page_content)

        return bm25_docs


class RerankerCompressor:
    """
    Document compressor that uses a cross-encoder to rerank retrieved documents.

    Separated from retrieval logic for better testability and flexibility.
    """

    def __init__(
        self,
        model_name: str = app_config.reranker.model_name,
        top_k: int = app_config.reranker.top_k,
        batch_size: int = app_config.reranker.batch_size,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model name
            top_k: Number of documents to return after reranking
            batch_size: Batch size for cross-encoder inference
        """
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        self.batch_size = batch_size

    def compress(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents using cross-encoder.

        Args:
            documents: List of candidate documents
            query: Original search query

        Returns:
            Top-k reranked documents
        """
        if not documents:
            return []

        if len(documents) == 1:
            return documents

        # Prepare input pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        # Select top-k documents
        k = min(self.top_k, len(documents))
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort by score descending
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [documents[i] for i in top_indices]


_RERANKER_CACHE = {}

def get_reranker(
    model_name: str = app_config.reranker.model_name,
    top_k: int = 2,
) -> RerankerCompressor:
    """
    Get or create a cached reranker instance.

    Note: This uses a module-level cache rather than lru_cache to allow
    configuration parameters while still benefiting from caching.
    """
    cache_key = (model_name, top_k)
    if cache_key not in _RERANKER_CACHE:
        _RERANKER_CACHE[cache_key] = RerankerCompressor(
            model_name=model_name,
            top_k=top_k
        )
    return _RERANKER_CACHE[cache_key]


def create_hybrid_retrieval_pipeline(
    vector_store: Chroma,
    bm25_index: Optional[BM25Okapi] = None,
    bm25_chunks: Optional[List[Document]] = None,
    use_reranking: bool = True,
) -> BaseRetriever:
    """
    Factory function to create a complete retrieval pipeline.

    Args:
        vector_store: ChromaDB instance
        bm25_index: Optional BM25 index
        bm25_chunks: Optional documents for BM25
        use_reranking: Whether to apply cross-encoder reranking

    Returns:
        Configured retriever (either hybrid or hybrid+reranked)
    """
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        bm25_chunks=bm25_chunks,
    )

    if not use_reranking:
        return hybrid_retriever

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from pydantic import ConfigDict

    class LangChainRerankerAdapter(BaseDocumentCompressor):
        """Adapter to make RerankerCompressor work with LangChain's interface."""

        reranker: RerankerCompressor

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def compress_documents(
                self,
                documents: List[Document],
                query: str,
                callbacks: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            return self.reranker.compress(documents, query)

    reranker = get_reranker()
    compressor = LangChainRerankerAdapter(reranker=reranker)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )