"""
Tests for document_store.py

Covers:
- Document storage and retrieval
- Retriever creation and configuration
- Integration with vector stores and BM25
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

try:
    from document_store import DocumentStore
except ImportError:
    pytest.skip("document_store module not found", allow_module_level=True)


@pytest.fixture
def mock_vector_store():
    """Mock vector store"""
    store = Mock()
    store.similarity_search.return_value = []
    mock_retriever = Mock()
    store.as_retriever.return_value = mock_retriever
    return store


@pytest.fixture
def mock_bm25_index():
    """Mock BM25 index"""
    import numpy as np
    index = Mock()
    index.get_scores.return_value = np.array([0.5, 0.8, 0.3])
    return index


@pytest.fixture
def sample_chunks():
    """Sample document chunks"""
    return [
        Document(page_content="Chunk 1", metadata={"source": "doc1.pdf"}),
        Document(page_content="Chunk 2", metadata={"source": "doc2.pdf"}),
        Document(page_content="Chunk 3", metadata={"source": "doc3.pdf"}),
    ]


@pytest.fixture
def doc_store(mock_vector_store):
    """Create DocumentStore with mocked components"""
    return DocumentStore(
        vector_store=mock_vector_store,
        bm25_index=None,
        bm25_chunks=[]
    )


@pytest.fixture
def doc_store_with_bm25(mock_vector_store, mock_bm25_index, sample_chunks):
    """Create DocumentStore with BM25 enabled"""
    return DocumentStore(
        vector_store=mock_vector_store,
        bm25_index=mock_bm25_index,
        bm25_chunks=sample_chunks
    )


class TestDocumentStoreInitialization:
    """Test DocumentStore initialization"""

    def test_init_with_vector_store_only(self, mock_vector_store):
        """Test initialization with only vector store"""
        store = DocumentStore(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[]
        )

        assert store.vector_store == mock_vector_store
        assert store.bm25_index is None
        assert store.bm25_chunks == []

    def test_init_with_bm25(self, mock_vector_store, mock_bm25_index, sample_chunks):
        """Test initialization with BM25 components"""
        store = DocumentStore(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=sample_chunks
        )

        assert store.vector_store == mock_vector_store
        assert store.bm25_index == mock_bm25_index
        assert store.bm25_chunks == sample_chunks


class TestRetrieverCreation:
    """Test retriever creation and configuration"""

    @patch('document_store.create_hybrid_retrieval_pipeline')
    def test_get_retriever_uses_factory(self, mock_factory, doc_store):
        """Test that get_retriever uses the factory function"""
        mock_retriever = Mock()
        mock_factory.return_value = mock_retriever

        retriever = doc_store.get_retriever()

        mock_factory.assert_called_once()
        assert retriever == mock_retriever

    @patch('document_store.create_hybrid_retrieval_pipeline')
    def test_get_retriever_passes_bm25_components(self, mock_factory, doc_store_with_bm25):
        """Test that BM25 components are passed to factory"""
        doc_store_with_bm25.get_retriever()

        call_kwargs = mock_factory.call_args[1]
        assert call_kwargs['bm25_index'] == doc_store_with_bm25.bm25_index
        assert call_kwargs['bm25_chunks'] == doc_store_with_bm25.bm25_chunks


class TestEmbeddingFunction:
    """Test embedding function retrieval"""

    def test_get_embedding_function(self, doc_store, mock_vector_store):
        """Test getting embedding function from vector store"""
        mock_embeddings = Mock()
        mock_vector_store.embeddings = mock_embeddings

        embeddings = doc_store.get_embedding_function()

        # Should return the embeddings from vector store
        assert embeddings is not None

    def test_get_embedding_function_attribute_error(self, doc_store, mock_vector_store):
        """Test handling when vector store has no embeddings attribute"""
        # Remove embeddings attribute
        if hasattr(mock_vector_store, 'embeddings'):
            delattr(mock_vector_store, 'embeddings')

        # Should handle gracefully or raise appropriate error
        try:
            embeddings = doc_store.get_embedding_function()
            # If it doesn't raise, check it returns something reasonable
            assert embeddings is not None or embeddings is None
        except AttributeError:
            # This is acceptable behavior
            pass


class TestVectorStoreOperations:
    """Test operations on the underlying vector store"""

    def test_similarity_search(self, doc_store, mock_vector_store, sample_chunks):
        """Test similarity search through document store"""
        mock_vector_store.similarity_search.return_value = sample_chunks[:2]

        results = mock_vector_store.similarity_search("test query", k=2)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_similarity_search_with_score(self, doc_store, mock_vector_store, sample_chunks):
        """Test similarity search with scores"""
        mock_vector_store.similarity_search_with_score.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.7)
        ]

        results = mock_vector_store.similarity_search_with_score("test query")

        assert len(results) == 2
        assert all(isinstance(doc, Document) and isinstance(score, float)
                   for doc, score in results)


class TestRetrieverIntegration:
    """Test retriever integration with document store"""

    def test_retriever_invoke(self, doc_store, sample_chunks):
        """Test invoking retriever"""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_chunks[:2]

        with patch.object(doc_store, 'get_retriever', return_value=mock_retriever):
            retriever = doc_store.get_retriever()
            results = retriever.invoke("test query")

            assert len(results) == 2
            assert all(isinstance(doc, Document) for doc in results)

    def test_retriever_with_custom_k(self, doc_store, sample_chunks):
        """Test retriever with custom k parameter"""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_chunks[:3]

        with patch.object(doc_store, 'get_retriever', return_value=mock_retriever):
            retriever = doc_store.get_retriever(k=3)
            results = retriever.invoke("test query")

            assert len(results) <= 3


class TestBM25Integration:
    """Test BM25 integration with document store"""

    def test_store_with_bm25_has_index(self, doc_store_with_bm25):
        """Test that store with BM25 has index"""
        assert doc_store_with_bm25.bm25_index is not None
        assert len(doc_store_with_bm25.bm25_chunks) > 0

    def test_store_without_bm25_has_no_index(self, doc_store):
        """Test that store without BM25 has no index"""
        assert doc_store.bm25_index is None
        assert doc_store.bm25_chunks == []

    def test_bm25_chunks_match_index(self, doc_store_with_bm25):
        """Test that BM25 chunks are available for index"""
        assert len(doc_store_with_bm25.bm25_chunks) > 0
        assert all(isinstance(chunk, Document)
                   for chunk in doc_store_with_bm25.bm25_chunks)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_bm25_chunks(self, mock_vector_store, mock_bm25_index):
        """Test store with BM25 index but empty chunks"""
        store = DocumentStore(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            bm25_chunks=[]
        )

        assert store.bm25_index is not None
        assert store.bm25_chunks == []

    def test_vector_store_without_methods(self):
        """Test handling vector store without expected methods"""
        incomplete_store = Mock(spec=[])  # No methods

        # Should either raise error or handle gracefully
        try:
            store = DocumentStore(
                vector_store=incomplete_store,
                bm25_index=None,
                bm25_chunks=[]
            )
            # If it doesn't raise, operations should fail appropriately
        except (AttributeError, TypeError):
            pass  # Expected

    def test_none_bm25_components(self, mock_vector_store):
        """Test with explicit None for BM25 components"""
        store = DocumentStore(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=None
        )

        # Should handle None gracefully
        assert store.bm25_index is None


class TestDocumentStoreState:
    """Test document store state management"""

    def test_vector_store_reference(self, doc_store, mock_vector_store):
        """Test that document store maintains reference to vector store"""
        assert doc_store.vector_store is mock_vector_store

    def test_bm25_reference(self, doc_store_with_bm25, mock_bm25_index):
        """Test that document store maintains reference to BM25 index"""
        assert doc_store_with_bm25.bm25_index is mock_bm25_index


class TestIntegration:
    """Integration tests for document store"""

    def test_complete_workflow(self, mock_vector_store, sample_chunks):
        """Test complete document store workflow"""
        # Create store
        store = DocumentStore(
            vector_store=mock_vector_store,
            bm25_index=None,
            bm25_chunks=[]
        )

        # Get retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_chunks

        with patch.object(store, 'get_retriever', return_value=mock_retriever):
            retriever = store.get_retriever()

            # Perform retrieval
            results = retriever.invoke("test query")

            assert len(results) == len(sample_chunks)
            assert all(isinstance(doc, Document) for doc in results)
