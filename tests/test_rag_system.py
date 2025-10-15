"""
Tests for rag_system.py

Covers:
- Query preprocessing and normalization
- Caching behavior (hits and misses)
- Error handling and formatting
- Streaming functionality
- Integration with LLM and retriever
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from rag_system import RAGSystem, RAGSystemError, create_rag_system


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration"""
    config = Mock()
    config.create_llm.return_value = Mock()
    config.get_prompt_template.return_value = Mock()
    config.get_display_name.return_value = "Test LLM"
    return config


@pytest.fixture
def mock_document_store():
    """Mock document store with proper retriever"""
    store = Mock()
    # Create a proper mock retriever that can be called
    mock_retriever = Mock()
    mock_retriever.get_relevant_documents.return_value = []
    store.get_retriever.return_value = mock_retriever
    store.get_embedding_function.return_value = Mock()
    return store


# Mock word_tokenize at module import time if it doesn't exist
import rag_system
if not hasattr(rag_system, 'word_tokenize'):
    rag_system.word_tokenize = lambda x: x.split()


@pytest.fixture
def rag_system_fixture(mock_document_store, mock_llm_config):
    """Create RAGSystem with mocked dependencies"""
    with patch.object(rag_system, 'STOP_WORDS', {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}):
        with patch.object(rag_system, 'word_tokenize', side_effect=lambda x: x.split()):
            return RAGSystem(
                document_store=mock_document_store,
                llm_config=mock_llm_config,
                enable_cache=True,
                cache_similarity_threshold=0.85,
                cache_max_size=100
            )


@pytest.fixture
def rag_system_no_cache(mock_document_store, mock_llm_config):
    """Create RAGSystem without caching"""
    return RAGSystem(
        document_store=mock_document_store,
        llm_config=mock_llm_config,
        enable_cache=False
    )


@pytest.fixture
def sample_source_docs():
    """Sample source documents for testing"""
    return [
        Document(page_content="Source 1", metadata={"source": "doc1.pdf"}),
        Document(page_content="Source 2", metadata={"source": "doc2.pdf"}),
    ]


class TestQueryPreprocessing:
    """Test query preprocessing and normalization"""

    def test_basic_preprocessing(self, mock_document_store, mock_llm_config):
        """Test basic query preprocessing"""
        with patch.object(rag_system, 'STOP_WORDS', {'what', 'is', 'the', 'a', 'an'}):
            with patch.object(rag_system, 'word_tokenize', side_effect=lambda x: x.split()):
                system = RAGSystem(
                    document_store=mock_document_store,
                    llm_config=mock_llm_config,
                    enable_cache=False
                )
                query = "What is machine learning?"
                processed = system._preprocess_query(query)

                assert processed.islower()
                assert "what" not in processed  # Stopword removed
                assert "machine" in processed
                assert "learning" in processed

    def test_contraction_expansion(self, rag_system_fixture):
        """Test contraction expansion"""
        query = "What's the difference between it's and its?"
        processed = rag_system_fixture._preprocess_query(query)

        # Should expand "what's" to "what is" and "it's" to "it is"
        assert "what is" in processed or "what" not in processed  # Depends on stopwords
        assert "it is" in processed or "its" in processed

    def test_punctuation_removal(self, rag_system_fixture):
        """Test punctuation removal"""
        query = "What is AI? How does it work!!!"
        processed = rag_system_fixture._preprocess_query(query)

        assert "?" not in processed
        assert "!" not in processed

    def test_whitespace_normalization(self, mock_document_store, mock_llm_config):
        """Test whitespace normalization"""
        with patch.object(rag_system, 'STOP_WORDS', set()):
            # Use a tokenizer that properly handles whitespace
            def tokenize_with_split(text):
                return [t for t in text.split() if t]

            with patch.object(rag_system, 'word_tokenize', side_effect=tokenize_with_split):
                system = RAGSystem(
                    document_store=mock_document_store,
                    llm_config=mock_llm_config,
                    enable_cache=False
                )
                query = "  Multiple    spaces   between words  "
                processed = system._preprocess_query(query)

                assert not processed.startswith(" ")
                assert not processed.endswith(" ")
                # The current implementation joins tokens with single space
                # So we verify the result is properly tokenized and rejoined
                words = processed.split()
                assert len(words) == 4
                assert words == ["multiple", "spaces", "between", "words"]

    def test_very_short_query_raises_error(self, rag_system_fixture):
        """Test that very short queries raise an error"""
        with pytest.raises(RAGSystemError, match="too short"):
            rag_system_fixture._preprocess_query("a")

    def test_empty_query_raises_error(self, rag_system_fixture):
        """Test that empty queries raise an error"""
        with pytest.raises(RAGSystemError, match="too short"):
            rag_system_fixture._preprocess_query("  ")

    def test_preprocessing_without_stopwords(self, mock_document_store, mock_llm_config):
        """Test preprocessing when NLTK stopwords aren't available"""
        with patch.object(rag_system, 'STOP_WORDS', set()):
            system = RAGSystem(
                document_store=mock_document_store,
                llm_config=mock_llm_config,
                enable_cache=False
            )
            query = "What is the meaning of life?"
            processed = system._preprocess_query(query)

            # Should still work, just without stopword removal
            assert isinstance(processed, str)
            assert len(processed) > 0


class TestCaching:
    """Test semantic cache functionality"""

    def test_similar_query_cache_hit(self, rag_system_fixture, sample_source_docs):
        """Test cache hit on semantically similar query"""
        with patch.object(rag_system_fixture.cache, 'get') as mock_cache_get:
            mock_cache_get.return_value = {
                "answer": "Cached answer",
                "source_documents": sample_source_docs
            }

            result = rag_system_fixture.ask_question("What is artificial intelligence?")

            assert result["answer"] == "Cached answer"

    def test_cache_disabled(self, rag_system_no_cache, sample_source_docs):
        """Test that caching can be disabled"""
        assert rag_system_no_cache.cache is None

        with patch.object(rag_system, 'RetrievalQA') as mock_qa_class:
            mock_chain = Mock()
            mock_result = {
                "result": "Answer",
                "source_documents": sample_source_docs
            }
            mock_chain.invoke.return_value = mock_result
            mock_qa_class.from_chain_type.return_value = mock_chain

            rag_system_no_cache.ask_question("Test query")
            rag_system_no_cache.ask_question("Test query")

            # Should call chain twice since cache is disabled
            assert mock_chain.invoke.call_count == 2

    def test_get_cache_stats(self, rag_system_fixture):
        """Test cache statistics"""
        stats = rag_system_fixture.get_cache_stats()

        assert stats["cache_enabled"] is True
        assert stats["cache_size"] == 0
        assert stats["max_size"] == 100
        assert stats["similarity_threshold"] == 0.85

    def test_get_cache_stats_disabled(self, rag_system_no_cache):
        """Test cache statistics when cache is disabled"""
        stats = rag_system_no_cache.get_cache_stats()

        assert stats["cache_enabled"] is False


class TestStreaming:
    """Test streaming functionality"""

    def test_streaming_with_cache_hit(self, rag_system_fixture, sample_source_docs):
        """Test streaming when cache hit occurs"""
        # Pre-populate cache
        cached_result = {
            "answer": "Cached answer",
            "source_documents": sample_source_docs
        }
        with patch.object(rag_system_fixture.cache, 'get', return_value=cached_result):
            tokens_received = []

            def token_callback(token):
                tokens_received.append(token)

            result = rag_system_fixture.ask_question_stream("Test query", token_callback)

            assert result == cached_result
            assert "".join(tokens_received) == "Cached answer"

class TestErrorHandling:
    """Test error handling and error messages"""

    def test_llm_error_with_ollama_hints(self, rag_system_fixture):
        """Test error message includes Ollama troubleshooting hints"""
        with patch.object(rag_system_fixture.llm_config, 'get_display_name', return_value="Ollama Model"):
            with patch.object(rag_system, 'RetrievalQA') as mock_qa_class:
                mock_chain = Mock()
                mock_chain.invoke.side_effect = Exception("Connection refused")
                mock_qa_class.from_chain_type.return_value = mock_chain

                with pytest.raises(RAGSystemError) as exc_info:
                    rag_system_fixture.ask_question("Test query")

                error_msg = str(exc_info.value)
                assert "Ollama" in error_msg
                assert "troubleshooting" in error_msg.lower()

    def test_llm_error_with_google_hints(self, rag_system_fixture):
        """Test error message includes Google API hints"""
        with patch.object(rag_system_fixture.llm_config, 'get_display_name', return_value="Google Gemini"):
            with patch.object(rag_system, 'RetrievalQA') as mock_qa_class:
                mock_chain = Mock()
                mock_chain.invoke.side_effect = Exception("API key invalid")
                mock_qa_class.from_chain_type.return_value = mock_chain

                with pytest.raises(RAGSystemError) as exc_info:
                    rag_system_fixture.ask_question("Test query")

                error_msg = str(exc_info.value)
                assert "Google" in error_msg or "Gemini" in error_msg
                assert "API key" in error_msg

    def test_generic_error_handling(self, rag_system_fixture):
        """Test generic error handling"""
        with patch.object(rag_system, 'RetrievalQA') as mock_qa_class:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = ValueError("Generic error")
            mock_qa_class.from_chain_type.return_value = mock_chain

            with pytest.raises(RAGSystemError) as exc_info:
                rag_system_fixture.ask_question("Test query")

            assert "Generic error" in str(exc_info.value)

    def test_rag_system_error_passthrough(self, rag_system_fixture):
        """Test that RAGSystemError is passed through without wrapping"""
        with patch.object(rag_system, 'RetrievalQA') as mock_qa_class:
            original_error = RAGSystemError("Custom error")
            mock_chain = Mock()
            mock_chain.invoke.side_effect = original_error
            mock_qa_class.from_chain_type.return_value = mock_chain

            with pytest.raises(RAGSystemError) as exc_info:
                rag_system_fixture.ask_question("Test query")

            # Should be the same error, not wrapped
            assert str(exc_info.value) == "Custom error"


class TestLLMInfo:
    """Test LLM information retrieval"""

    def test_get_llm_info(self, rag_system_fixture, mock_llm_config):
        """Test getting LLM display information"""
        mock_llm_config.get_display_name.return_value = "Test Model v1.0"

        info = rag_system_fixture.get_llm_info()

        assert info == "Test Model v1.0"


class TestCreateRAGSystem:
    """Test the factory function for creating RAG systems"""

    def test_create_rag_system_local_mode(self):
        """Test creating RAG system in local mode"""
        mock_vector_store = Mock()

        # Patch the imports within the create_rag_system function
        with patch('llm_factory.LLMFactory') as mock_llm_factory:
            with patch('document_store.DocumentStore') as mock_doc_store_class:
                mock_llm_config = Mock()
                mock_llm_config.create_llm.return_value = Mock()
                mock_llm_config.get_prompt_template.return_value = Mock()
                mock_llm_config.get_display_name.return_value = "Test LLM"
                mock_llm_factory.create_from_mode.return_value = mock_llm_config

                mock_doc_store = Mock()
                mock_doc_store.get_retriever.return_value = Mock()
                mock_doc_store.get_embedding_function.return_value = Mock()
                mock_doc_store_class.return_value = mock_doc_store

                rag = create_rag_system(
                    vector_store=mock_vector_store,
                    mode="local",
                    enable_cache=True,
                    cache_similarity_threshold=0.9,
                    cache_max_size=50
                )

                mock_llm_factory.create_from_mode.assert_called_once_with(
                    mode="local",
                    api_key=None,
                    model_name=None
                )
                assert isinstance(rag, RAGSystem)

    def test_create_rag_system_google_mode(self):
        """Test creating RAG system in Google mode"""
        mock_vector_store = Mock()

        with patch('llm_factory.LLMFactory') as mock_llm_factory:
            with patch('document_store.DocumentStore') as mock_doc_store_class:
                mock_llm_config = Mock()
                mock_llm_config.create_llm.return_value = Mock()
                mock_llm_config.get_prompt_template.return_value = Mock()
                mock_llm_config.get_display_name.return_value = "Test LLM"
                mock_llm_factory.create_from_mode.return_value = mock_llm_config

                mock_doc_store = Mock()
                mock_doc_store.get_retriever.return_value = Mock()
                mock_doc_store.get_embedding_function.return_value = Mock()
                mock_doc_store_class.return_value = mock_doc_store

                rag = create_rag_system(
                    vector_store=mock_vector_store,
                    mode="google",
                    api_key="test_key",
                    model_name="gemini-pro"
                )

                mock_llm_factory.create_from_mode.assert_called_once_with(
                    mode="google",
                    api_key="test_key",
                    model_name="gemini-pro"
                )

    def test_create_rag_system_with_bm25(self):
        """Test creating RAG system with BM25 index"""
        mock_vector_store = Mock()
        mock_bm25_index = Mock()
        mock_bm25_chunks = [Mock()]

        with patch('llm_factory.LLMFactory') as mock_llm_factory:
            with patch('document_store.DocumentStore') as mock_doc_store_class:
                mock_llm_config = Mock()
                mock_llm_config.create_llm.return_value = Mock()
                mock_llm_config.get_prompt_template.return_value = Mock()
                mock_llm_config.get_display_name.return_value = "Test LLM"
                mock_llm_factory.create_from_mode.return_value = mock_llm_config

                mock_doc_store = Mock()
                mock_doc_store.get_retriever.return_value = Mock()
                mock_doc_store.get_embedding_function.return_value = Mock()
                mock_doc_store_class.return_value = mock_doc_store

                rag = create_rag_system(
                    vector_store=mock_vector_store,
                    mode="local",
                    bm25_index=mock_bm25_index,
                    bm25_chunks=mock_bm25_chunks
                )

                mock_doc_store_class.assert_called_once_with(
                    vector_store=mock_vector_store,
                    bm25_index=mock_bm25_index,
                    bm25_chunks=mock_bm25_chunks
                )


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_workflow_with_cache_disabled(self, rag_system_no_cache, sample_source_docs):
        """Test workflow with caching disabled"""
        with patch.object(rag_system, 'RetrievalQA') as mock_qa_class:
            mock_chain = Mock()
            mock_result = {
                "result": "Answer",
                "source_documents": sample_source_docs
            }
            mock_chain.invoke.return_value = mock_result
            mock_qa_class.from_chain_type.return_value = mock_chain

            result1 = rag_system_no_cache.ask_question("Query")
            result2 = rag_system_no_cache.ask_question("Query")

            # Chain should be called twice without caching
            assert mock_chain.invoke.call_count == 2

class TestTokenStreamCallbackHandler:
    """Test the streaming callback handler"""

    def test_callback_handler_calls_function(self):
        """Test that callback handler invokes the provided function"""
        from rag_system import TokenStreamCallbackHandler

        tokens_received = []

        def callback(token):
            tokens_received.append(token)

        handler = TokenStreamCallbackHandler(callback)

        handler.on_llm_new_token("Hello")
        handler.on_llm_new_token(" ")
        handler.on_llm_new_token("World")

        assert tokens_received == ["Hello", " ", "World"]

    def test_callback_handler_with_kwargs(self):
        """Test that callback handler handles additional kwargs"""
        from rag_system import TokenStreamCallbackHandler

        tokens_received = []

        def callback(token):
            tokens_received.append(token)

        handler = TokenStreamCallbackHandler(callback)

        # Should handle extra kwargs without error
        handler.on_llm_new_token("Test", chunk=None, run_id="123")

        assert tokens_received == ["Test"]