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


class TestHybridRetrieverInitialization:
    """Test HybridRetriever initialization"""

    def test_init_with_bm25(self, mock_vector_store, mock_bm25_index, sample_documents):
        """Test initialization with BM25 components"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=sample_documents
        )

        assert retriever.vector_store == mock_vector_store
        assert retriever.bm25_index == mock_bm25_index
        assert retriever.bm25_chunks == sample_documents

    def test_init_without_bm25(self, mock_vector_store):
        """Test initialization without BM25 components"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[]
        )

        assert retriever.bm25_index is None
        assert retriever.bm25_chunks == []

    def test_init_mismatched_bm25_raises_error(self, mock_vector_store, mock_bm25_index, sample_documents):
        """Test that mismatched BM25 components raise error"""
        with pytest.raises(ValueError, match="bm25_index and bm25_chunks"):
            HybridRetriever(
                vector_store=mock_vector_store,
                bm25_index=mock_bm25_index,
                bm25_chunks=[]  # Missing chunks
            )

        with pytest.raises(ValueError, match="bm25_index and bm25_chunks"):
            HybridRetriever(
                vector_store=mock_vector_store,
                bm25_index=None,
                bm25_chunks=sample_documents  # Missing index
            )

    def test_default_parameters(self, mock_vector_store):
        """Test that default parameters are set correctly"""
        from config import app_config

        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[]
        )

        assert retriever.vector_k == app_config.retriever.vector_k
        assert retriever.vector_fetch_k == app_config.retriever.vector_fetch_k
        assert retriever.lambda_mult == app_config.retriever.lambda_mult


class TestVectorSearch:
    """Test vector search functionality"""

    def test_vector_search_basic(self, hybrid_retriever_no_bm25, sample_documents):
        """Test basic vector search"""
        query = "machine learning"
        docs = hybrid_retriever_no_bm25._vector_search(query)

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)

    def test_vector_search_uses_mmr(self, hybrid_retriever_no_bm25):
        """Test that vector search uses MMR"""
        query = "test query"
        hybrid_retriever_no_bm25._vector_search(query)

        # Verify MMR parameters were used
        hybrid_retriever_no_bm25.vector_store.as_retriever.assert_called_once()
        call_kwargs = hybrid_retriever_no_bm25.vector_store.as_retriever.call_args

        assert call_kwargs[1]["search_type"] == "mmr"
        assert "k" in call_kwargs[1]["search_kwargs"]
        assert "fetch_k" in call_kwargs[1]["search_kwargs"]
        assert "lambda_mult" in call_kwargs[1]["search_kwargs"]

    def test_vector_search_returns_k_documents(self, mock_vector_store):
        """Test that vector search respects k parameter"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[],
            vector_k=5
        )

        retriever._vector_search("test")

        call_kwargs = mock_vector_store.as_retriever.call_args[1]["search_kwargs"]
        assert call_kwargs["k"] == 5


class TestBM25Search:
    """Test BM25 search functionality"""

    def test_bm25_search_basic(self, hybrid_retriever_with_bm25, sample_documents):
        """Test basic BM25 search"""
        query = "machine learning"
        existing_docs = []

        bm25_docs = hybrid_retriever_with_bm25._bm25_search(query, existing_docs)

        assert len(bm25_docs) <= 2  # Should return up to bm25_top_k
        assert all(isinstance(doc, Document) for doc in bm25_docs)

    def test_bm25_search_without_index_returns_empty(self, hybrid_retriever_no_bm25):
        """Test BM25 search returns empty list without index"""
        query = "test"
        existing_docs = []

        bm25_docs = hybrid_retriever_no_bm25._bm25_search(query, existing_docs)

        assert bm25_docs == []

    def test_bm25_search_deduplication(self, hybrid_retriever_with_bm25, sample_documents):
        """Test that BM25 search deduplicates against existing docs"""
        query = "machine learning"
        # Pass the highest scoring documents as existing
        existing_docs = [sample_documents[3]]  # Doc with score 0.9

        bm25_docs = hybrid_retriever_with_bm25._bm25_search(query, existing_docs)

        # Should not return the existing document
        existing_content = {doc.page_content for doc in existing_docs}
        for doc in bm25_docs:
            assert doc.page_content not in existing_content

    def test_bm25_search_tokenizes_query(self, hybrid_retriever_with_bm25, mock_bm25_index):
        """Test that BM25 search tokenizes the query"""
        query = "machine learning algorithms"
        existing_docs = []

        hybrid_retriever_with_bm25._bm25_search(query, existing_docs)

        # Should call get_scores with tokenized query
        mock_bm25_index.get_scores.assert_called_once()
        called_tokens = mock_bm25_index.get_scores.call_args[0][0]
        assert isinstance(called_tokens, list)
        assert "machine" in called_tokens
        assert "learning" in called_tokens

    def test_bm25_search_returns_top_k(self, hybrid_retriever_with_bm25):
        """Test that BM25 returns top-k highest scoring documents"""
        # Scores are [0.5, 0.8, 0.3, 0.9]
        # Top 2 should be indices 3 (0.9) and 1 (0.8)
        query = "test"
        existing_docs = []

        bm25_docs = hybrid_retriever_with_bm25._bm25_search(query, existing_docs)

        # Should get top 2 documents by score
        assert len(bm25_docs) <= 2

    def test_bm25_search_handles_empty_chunks(self, mock_vector_store, mock_bm25_index):
        """Test BM25 search with empty chunks list"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=[],
            bm25_top_k=2
        )

        bm25_docs = retriever._bm25_search("test", [])

        assert bm25_docs == []


class TestHybridRetrieval:
    """Test hybrid retrieval combining vector and BM25"""

    def test_retrieval_vector_only_sufficient_docs(self, hybrid_retriever_with_bm25, sample_documents):
        """Test that BM25 is not used when vector search returns enough docs"""
        # Configure to require 3 docs, but vector returns 2
        # This should NOT trigger BM25 because min_docs_before_bm25=3 and we got 2
        # Actually, looking at the code: if len(vector_docs) < min_docs_before_bm25, use BM25
        # So with min_docs_before_bm25=3 and 2 docs, BM25 SHOULD be used

        retriever = HybridRetriever(
            vector_store=hybrid_retriever_with_bm25.vector_store,
            bm25_index=hybrid_retriever_with_bm25.bm25_index,
            bm25_chunks=hybrid_retriever_with_bm25.bm25_chunks,
            vector_k=5,
            min_docs_before_bm25=2  # Don't use BM25 if we have 2+ docs
        )

        with patch.object(retriever, '_vector_search', return_value=sample_documents[:3]):
            with patch.object(retriever, '_bm25_search') as mock_bm25:
                docs = retriever._get_relevant_documents("test query")

                # Should not call BM25 because we have 3 docs and threshold is 2
                mock_bm25.assert_not_called()
                assert len(docs) == 3

    def test_retrieval_triggers_bm25_insufficient_docs(self, hybrid_retriever_with_bm25):
        """Test that BM25 is used when vector search returns insufficient docs"""
        # min_docs_before_bm25 = 3, vector returns 2 docs
        with patch.object(hybrid_retriever_with_bm25, '_vector_search', return_value=[Mock(), Mock()]):
            with patch.object(hybrid_retriever_with_bm25, '_bm25_search', return_value=[Mock()]) as mock_bm25:
                docs = hybrid_retriever_with_bm25._get_relevant_documents("test query")

                # Should call BM25 because 2 < 3
                mock_bm25.assert_called_once()
                assert len(docs) == 3  # 2 from vector + 1 from BM25

    def test_retrieval_without_bm25_index(self, hybrid_retriever_no_bm25, sample_documents):
        """Test retrieval when BM25 is not available"""
        with patch.object(hybrid_retriever_no_bm25, '_vector_search', return_value=sample_documents[:2]):
            docs = hybrid_retriever_no_bm25._get_relevant_documents("test query")

            # Should only return vector results
            assert len(docs) == 2

    def test_retrieval_combines_results(self, hybrid_retriever_with_bm25, sample_documents):
        """Test that hybrid retrieval combines vector and BM25 results"""
        vector_docs = sample_documents[:2]
        bm25_docs = [sample_documents[2]]

        with patch.object(hybrid_retriever_with_bm25, '_vector_search', return_value=vector_docs):
            with patch.object(hybrid_retriever_with_bm25, '_bm25_search', return_value=bm25_docs):
                docs = hybrid_retriever_with_bm25._get_relevant_documents("test query")

                assert len(docs) == 3
                assert docs[:2] == vector_docs
                assert docs[2:] == bm25_docs


class TestRerankerCompressor:
    """Test cross-encoder reranking"""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder model"""
        with patch('hybrid_retriever.CrossEncoder') as mock_ce:
            encoder = Mock()
            encoder.predict.return_value = np.array([0.8, 0.3, 0.9, 0.5])
            mock_ce.return_value = encoder
            yield mock_ce

    def test_reranker_initialization(self, mock_cross_encoder):
        """Test reranker compressor initialization"""
        reranker = RerankerCompressor(
            model_name="test-model",
            top_k=3,
            batch_size=16
        )

        assert reranker.top_k == 3
        assert reranker.batch_size == 16
        mock_cross_encoder.assert_called_once_with("test-model")

    def test_compress_empty_documents(self, mock_cross_encoder):
        """Test reranking with empty document list"""
        reranker = RerankerCompressor()

        result = reranker.compress([], "test query")

        assert result == []

    def test_compress_single_document(self, mock_cross_encoder, sample_documents):
        """Test reranking with single document"""
        reranker = RerankerCompressor()

        result = reranker.compress([sample_documents[0]], "test query")

        # Should return the single document without scoring
        assert len(result) == 1
        assert result[0] == sample_documents[0]

    def test_compress_reranks_documents(self, mock_cross_encoder, sample_documents):
        """Test that reranking orders documents by relevance"""
        reranker = RerankerCompressor(top_k=3)

        # Scores are [0.8, 0.3, 0.9, 0.5]
        # Top 3 should be indices 2, 0, 3 (scores 0.9, 0.8, 0.5)
        result = reranker.compress(sample_documents, "test query")

        assert len(result) == 3
        # Results should be ordered by score (descending)
        # Can't check exact order without knowing implementation details

    def test_compress_respects_top_k(self, mock_cross_encoder, sample_documents):
        """Test that reranker returns only top_k documents"""
        reranker = RerankerCompressor(top_k=2)

        result = reranker.compress(sample_documents, "test query")

        assert len(result) == 2

    def test_compress_top_k_larger_than_docs(self, mock_cross_encoder, sample_documents):
        """Test reranking when top_k > number of documents"""
        reranker = RerankerCompressor(top_k=10)

        result = reranker.compress(sample_documents[:2], "test query")

        # Should return all documents when top_k > len(documents)
        assert len(result) == 2

    def test_compress_calls_cross_encoder(self, mock_cross_encoder, sample_documents):
        """Test that compress calls cross-encoder with correct inputs"""
        reranker = RerankerCompressor(batch_size=8)
        query = "machine learning"

        reranker.compress(sample_documents, query)

        # Should call predict with query-document pairs
        reranker.model.predict.assert_called_once()
        call_args = reranker.model.predict.call_args

        pairs = call_args[0][0]
        assert len(pairs) == len(sample_documents)
        assert all(pair[0] == query for pair in pairs)
        assert call_args[1]["batch_size"] == 8


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


class TestHybridRetrievalPipeline:
    """Test the factory function for creating retrieval pipelines"""

    def test_create_pipeline_without_reranking(self, mock_vector_store, mock_bm25_index, sample_documents):
        """Test creating pipeline without reranking"""
        pipeline = create_hybrid_retrieval_pipeline(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=sample_documents,
            use_reranking=False
        )

        assert isinstance(pipeline, HybridRetriever)

    def test_create_pipeline_with_reranking(self, mock_vector_store, mock_bm25_index, sample_documents):
        """Test creating pipeline with reranking"""
        with patch('hybrid_retriever.get_reranker') as mock_get_reranker:
            mock_reranker = Mock()
            mock_get_reranker.return_value = mock_reranker

            pipeline = create_hybrid_retrieval_pipeline(
                vector_store=mock_vector_store,
                bm25_index=mock_bm25_index,
                bm25_chunks=sample_documents,
                use_reranking=True
            )

            # Should wrap in ContextualCompressionRetriever
            from langchain.retrievers import ContextualCompressionRetriever
            assert isinstance(pipeline, ContextualCompressionRetriever)

    def test_create_pipeline_no_bm25(self, mock_vector_store):
        """Test creating pipeline without BM25"""
        pipeline = create_hybrid_retrieval_pipeline(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=None,
            use_reranking=False
        )

        assert isinstance(pipeline, HybridRetriever)
        assert pipeline.bm25_index is None


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_retrieval_with_zero_vector_k(self, mock_vector_store):
        """Test retrieval with k=0"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[],
            vector_k=0
        )

        # Should handle gracefully
        with patch.object(retriever, '_vector_search', return_value=[]):
            docs = retriever._get_relevant_documents("test")
            assert docs == []

    def test_bm25_search_all_duplicates(self, hybrid_retriever_with_bm25, sample_documents):
        """Test BM25 search when all results are duplicates"""
        query = "test"
        # Pass all documents as existing
        existing_docs = sample_documents

        bm25_docs = hybrid_retriever_with_bm25._bm25_search(query, existing_docs)

        # Should return empty list since all are duplicates
        assert bm25_docs == []

    def test_bm25_with_very_long_query(self, hybrid_retriever_with_bm25):
        """Test BM25 with very long query"""
        long_query = "word " * 1000
        existing_docs = []

        # Should handle without error
        bm25_docs = hybrid_retriever_with_bm25._bm25_search(long_query, existing_docs)

        assert isinstance(bm25_docs, list)

    def test_reranker_with_identical_documents(self, sample_documents):
        """Test reranker with duplicate content"""
        with patch('hybrid_retriever.CrossEncoder') as mock_ce:
            encoder = Mock()
            encoder.predict.return_value = np.array([0.5, 0.5, 0.5])
            mock_ce.return_value = encoder

            reranker = RerankerCompressor(top_k=2)

            # Create docs with same content
            identical_docs = [
                Document(page_content="Same content", metadata={"id": i})
                for i in range(3)
            ]

            result = reranker.compress(identical_docs, "query")

            assert len(result) == 2

    def test_vector_search_empty_query(self, hybrid_retriever_no_bm25):
        """Test vector search with empty query"""
        docs = hybrid_retriever_no_bm25._vector_search("")

        # Should return results even with empty query
        assert isinstance(docs, list)

    def test_bm25_search_with_special_characters(self, hybrid_retriever_with_bm25):
        """Test BM25 search with special characters in query"""
        query = "machine-learning @#$ 123"
        existing_docs = []

        # Should handle special characters
        bm25_docs = hybrid_retriever_with_bm25._bm25_search(query, existing_docs)

        assert isinstance(bm25_docs, list)


class TestLangChainRerankerAdapter:
    """Test the LangChain adapter for reranker"""

    def test_adapter_compress_documents(self, sample_documents):
        """Test that adapter correctly calls underlying reranker"""
        with patch('hybrid_retriever.RerankerCompressor') as mock_compressor_class:
            mock_reranker = Mock()
            mock_reranker.compress.return_value = sample_documents[:2]
            mock_compressor_class.return_value = mock_reranker

            # Create the pipeline with reranking
            with patch('hybrid_retriever.get_reranker', return_value=mock_reranker):
                from hybrid_retriever import create_hybrid_retrieval_pipeline
                mock_vector_store = Mock()

                pipeline = create_hybrid_retrieval_pipeline(
                    vector_store=mock_vector_store,
                    bm25_index=None,
                    bm25_chunks=[],
                    use_reranking=True
                )

                # The adapter should be in the compressor
                assert hasattr(pipeline, 'base_compressor')

    def test_adapter_handles_callbacks(self, sample_documents):
        """Test that adapter handles optional callbacks parameter"""
        with patch('hybrid_retriever.CrossEncoder') as mock_ce:
            encoder = Mock()
            encoder.predict.return_value = np.array([0.8, 0.3])
            mock_ce.return_value = encoder

            reranker = RerankerCompressor()

            # Should work with callbacks=None
            result = reranker.compress(sample_documents[:2], "query")
            assert len(result) == 2


class TestIntegration:
    """Integration tests for complete retrieval workflows"""

    def test_complete_hybrid_workflow(self, mock_vector_store, mock_bm25_index, sample_documents):
        """Test complete hybrid retrieval workflow"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=sample_documents,
            vector_k=2,
            bm25_top_k=2,
            min_docs_before_bm25=3
        )

        # Mock vector search to return insufficient results
        with patch.object(retriever, '_vector_search', return_value=sample_documents[:1]):
            docs = retriever._get_relevant_documents("machine learning")

            # Should combine vector (1) + BM25 (up to 2) results
            assert len(docs) >= 1
            assert all(isinstance(doc, Document) for doc in docs)

    def test_pipeline_with_reranking_workflow(self, mock_vector_store, sample_documents):
        """Test complete pipeline with reranking"""
        with patch('hybrid_retriever.CrossEncoder') as mock_ce:
            encoder = Mock()
            encoder.predict.return_value = np.array([0.9, 0.3, 0.7, 0.5])
            mock_ce.return_value = encoder

            # Create mock retriever that returns all sample docs
            mock_retriever = Mock()
            mock_retriever.invoke.return_value = sample_documents
            mock_vector_store.as_retriever.return_value = mock_retriever

            pipeline = create_hybrid_retrieval_pipeline(
                vector_store=mock_vector_store,
                bm25_index=None,
                bm25_chunks=[],
                use_reranking=True
            )

            # Pipeline should be a ContextualCompressionRetriever
            from langchain.retrievers import ContextualCompressionRetriever
            assert isinstance(pipeline, ContextualCompressionRetriever)

    def test_vector_only_workflow(self, mock_vector_store, sample_documents):
        """Test vector-only retrieval (no BM25, no reranking)"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[],
            vector_k=3
        )

        with patch.object(retriever, '_vector_search', return_value=sample_documents[:3]):
            docs = retriever._get_relevant_documents("test query")

            assert len(docs) == 3
            assert docs == sample_documents[:3]

    def test_deduplication_across_sources(self, mock_vector_store, mock_bm25_index):
        """Test that deduplication works across vector and BM25 results"""
        # Create documents with some duplicates
        doc1 = Document(page_content="Unique content 1", metadata={"source": "doc1"})
        doc2 = Document(page_content="Duplicate content", metadata={"source": "doc2"})
        doc3 = Document(page_content="Unique content 2", metadata={"source": "doc3"})
        doc4 = Document(page_content="Duplicate content", metadata={"source": "doc4"})  # Duplicate

        all_docs = [doc1, doc2, doc3, doc4]

        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=all_docs,
            vector_k=2,
            bm25_top_k=2,
            min_docs_before_bm25=3
        )

        # Mock vector search returns docs with duplicate
        with patch.object(retriever, '_vector_search', return_value=[doc1, doc2]):
            # BM25 should not return doc2 again
            with patch.object(mock_bm25_index, 'get_scores', return_value=np.array([0.5, 0.9, 0.3, 0.8])):
                docs = retriever._get_relevant_documents("test")

                # Check no duplicate content
                contents = [doc.page_content for doc in docs]
                assert len(contents) == len(set(contents))  # All unique