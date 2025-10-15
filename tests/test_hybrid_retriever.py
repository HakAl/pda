"""
Tests for hybrid_retriever.py

Covers:
- Hybrid retrieval (vector + BM25)
- Deduplication logic
- MMR search
- Reranking with cross-encoder
- Factory functions and caching
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from hybrid_retriever import (
    HybridRetriever,
    RerankerCompressor,
    get_reranker,
    create_hybrid_retrieval_pipeline,
    _RERANKER_CACHE
)


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(page_content="Machine learning is a subset of AI", metadata={"source": "doc1.pdf"}),
        Document(page_content="Deep learning uses neural networks", metadata={"source": "doc2.pdf"}),
        Document(page_content="Natural language processing is important", metadata={"source": "doc3.pdf"}),
        Document(page_content="Computer vision helps machines see", metadata={"source": "doc4.pdf"}),
    ]


@pytest.fixture
def mock_vector_store(sample_documents):
    """Mock ChromaDB vector store"""
    store = Mock()
    retriever = Mock()
    retriever.invoke.return_value = sample_documents[:2]  # Return first 2 docs
    store.as_retriever.return_value = retriever
    return store


@pytest.fixture
def mock_bm25_index():
    """Mock BM25 index"""
    index = Mock()
    # Return scores for 4 documents
    index.get_scores.return_value = np.array([0.5, 0.8, 0.3, 0.9])
    return index


@pytest.fixture
def hybrid_retriever_with_bm25(mock_vector_store, mock_bm25_index, sample_documents):
    """Create HybridRetriever with BM25 enabled"""
    return HybridRetriever(
        vector_store=mock_vector_store,
        bm25_index=mock_bm25_index,
        bm25_chunks=sample_documents,
        vector_k=2,
        bm25_top_k=2,
        min_docs_before_bm25=3
    )


@pytest.fixture
def hybrid_retriever_no_bm25(mock_vector_store):
    """Create HybridRetriever without BM25"""
    return HybridRetriever(
        vector_store=mock_vector_store,
        bm25_index=None,
        bm25_chunks=[],
        vector_k=2
    )


class TestRerankerCaching:
    """Test reranker caching functionality"""

    def teardown_method(self):
        """Clear cache after each test"""
        _RERANKER_CACHE.clear()

    def test_get_reranker_creates_instance(self):
        """Test that get_reranker creates a new instance"""
        with patch('hybrid_retriever.RerankerCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_compressor.return_value = mock_instance

            reranker = get_reranker(model_name="test-model", top_k=3)

            assert reranker == mock_instance
            mock_compressor.assert_called_once_with(model_name="test-model", top_k=3)

    def test_get_reranker_caches_instance(self):
        """Test that get_reranker caches instances"""
        with patch('hybrid_retriever.RerankerCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_compressor.return_value = mock_instance

            reranker1 = get_reranker(model_name="test-model", top_k=3)
            reranker2 = get_reranker(model_name="test-model", top_k=3)

            # Should return the same cached instance
            assert reranker1 is reranker2
            # Constructor should only be called once
            assert mock_compressor.call_count == 1

    def test_get_reranker_different_params_different_cache(self):
        """Test that different parameters create different cache entries"""
        with patch('hybrid_retriever.RerankerCompressor') as mock_compressor:
            mock_compressor.side_effect = [Mock(), Mock()]

            reranker1 = get_reranker(model_name="model1", top_k=3)
            reranker2 = get_reranker(model_name="model2", top_k=3)

            # Should be different instances
            assert reranker1 is not reranker2
            assert mock_compressor.call_count == 2
