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
    """Mock document store"""
    store = Mock()
    store.get_retriever.return_value = Mock()
    store.get_embedding_function.return_value = Mock()
    return store


@pytest.fixture
def rag_system(mock_document_store, mock_llm_config):
    """Create RAGSystem with mocked dependencies"""
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

    def test_basic_preprocessing(self, rag_system):
        """Test basic query preprocessing"""
        query = "What is machine learning?"
        processed = rag_system._preprocess_query(query)

        assert processed.islower()
        assert "what" not in processed  # Stopword removed
        assert "machine" in processed
        assert "learning" in processed

    def test_contraction_expansion(self, rag_system):
        """Test contraction expansion"""
        query = "What's the difference between it's and its?"
        processed = rag_system._preprocess_query(query)

        assert "what is" in processed or "what" not in processed  # Depends on stopwords
        assert "it is" in processed or "its" in processed

    def test_punctuation_removal(self, rag_system):
        """Test punctuation removal"""
        query = "What is AI? How does it work!!!"
        processed = rag_system._preprocess_query(query)

        assert "?" not in processed
        assert "!" not in processed

    def test_whitespace_normalization(self, rag_system):
        """Test whitespace normalization"""
        query = "  Multiple    spaces   between words  "
        processed = rag_system._preprocess_query(query)

        assert not processed.startswith(" ")
        assert not processed.endswith(" ")
        assert "  " not in processed

    def test_very_short_query_raises_error(self, rag_system):
        """Test that very short queries raise an error"""
        with pytest.raises(RAGSystemError, match="too short"):
            rag_system._preprocess_query("a")

    def test_empty_query_raises_error(self, rag_system):
        """Test that empty queries raise an error"""
        with pytest.raises(RAGSystemError, match="too short"):
            rag_system._preprocess_query("  ")

    @patch('rag_system.STOP_WORDS', set())
    def test_preprocessing_without_stopwords(self, rag_system):
        """Test preprocessing when NLTK stopwords aren't available"""
        query = "What is the meaning of life?"
        processed = rag_system._preprocess_query(query)

        # Should still work, just without stopword removal
        assert isinstance(processed, str)
        assert len(processed) > 0


class TestCaching:
    """Test semantic cache functionality"""

    def test_cache_miss_and_set(self, rag_system, sample_source_docs):
        """Test cache miss and subsequent cache set"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            mock_result = {
                "result": "Test answer",
                "source_documents": sample_source_docs
            }
            mock_chain.return_value.invoke.return_value = mock_result

            result = rag_system.ask_question("What is AI?")

            assert result["answer"] == "Test answer"
            assert result["source_documents"] == sample_source_docs
            assert len(rag_system.cache.cache) == 1

    def test_cache_hit(self, rag_system, sample_source_docs):
        """Test cache hit on repeated query"""
        # First call - cache miss
        with patch.object(rag_system, '_get_chain') as mock_chain:
            mock_result = {
                "result": "Cached answer",
                "source_documents": sample_source_docs
            }
            mock_chain.return_value.invoke.return_value = mock_result

            result1 = rag_system.ask_question("What is AI?")
            call_count_after_first = mock_chain.call_count

            # Second call - should hit cache
            result2 = rag_system.ask_question("What is AI?")

            assert result1 == result2
            assert mock_chain.call_count == call_count_after_first  # Chain not called again

    def test_similar_query_cache_hit(self, rag_system, sample_source_docs):
        """Test cache hit on semantically similar query"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            with patch.object(rag_system.cache, 'get') as mock_cache_get:
                mock_cache_get.return_value = {
                    "answer": "Cached answer",
                    "source_documents": sample_source_docs
                }

                result = rag_system.ask_question("What is artificial intelligence?")

                assert result["answer"] == "Cached answer"
                mock_chain.assert_not_called()

    def test_cache_disabled(self, rag_system_no_cache, sample_source_docs):
        """Test that caching can be disabled"""
        assert rag_system_no_cache.cache is None

        with patch.object(rag_system_no_cache, '_get_chain') as mock_chain:
            mock_result = {
                "result": "Answer",
                "source_documents": sample_source_docs
            }
            mock_chain.return_value.invoke.return_value = mock_result

            rag_system_no_cache.ask_question("Test query")
            rag_system_no_cache.ask_question("Test query")

            # Should call chain twice since cache is disabled
            assert mock_chain.call_count == 2

    def test_clear_cache(self, rag_system, sample_source_docs):
        """Test cache clearing"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            mock_result = {
                "result": "Answer",
                "source_documents": sample_source_docs
            }
            mock_chain.return_value.invoke.return_value = mock_result

            rag_system.ask_question("Query 1")
            rag_system.ask_question("Query 2")

            assert len(rag_system.cache.cache) == 2

            rag_system.clear_cache()

            assert len(rag_system.cache.cache) == 0

    def test_get_cache_stats(self, rag_system):
        """Test cache statistics"""
        stats = rag_system.get_cache_stats()

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

    def test_streaming_without_cache(self, rag_system, sample_source_docs):
        """Test streaming when no cache hit"""
        rag_system.clear_cache()  # Ensure cache is empty

        tokens_received = []

        def token_callback(token):
            tokens_received.append(token)

        with patch('rag_system.RetrievalQA') as mock_qa:
            mock_chain = Mock()
            mock_chain.invoke.return_value = {
                "result": "Test answer",
                "source_documents": sample_source_docs
            }
            mock_qa.from_chain_type.return_value = mock_chain

            # Mock the streaming LLM to call the callback
            with patch.object(rag_system.llm_config, 'create_llm') as mock_create_llm:
                mock_streaming_llm = Mock()
                mock_create_llm.return_value = mock_streaming_llm

                result = rag_system.ask_question_stream("Test query", token_callback)

        assert "answer" in result
        assert "source_documents" in result

    def test_streaming_with_cache_hit(self, rag_system, sample_source_docs):
        """Test streaming when cache hit occurs"""
        # Pre-populate cache
        cached_result = {
            "answer": "Cached answer",
            "source_documents": sample_source_docs
        }
        with patch.object(rag_system.cache, 'get', return_value=cached_result):
            tokens_received = []

            def token_callback(token):
                tokens_received.append(token)

            result = rag_system.ask_question_stream("Test query", token_callback)

            assert result == cached_result
            assert "".join(tokens_received) == "Cached answer"

    def test_streaming_token_collection(self, rag_system, sample_source_docs):
        """Test that streaming collects tokens for caching"""
        rag_system.clear_cache()

        collected_tokens = []

        def token_callback(token):
            collected_tokens.append(token)

        with patch('rag_system.RetrievalQA') as mock_qa:
            mock_chain = Mock()

            # Simulate the chain calling the token callback
            def invoke_with_callbacks(*args, **kwargs):
                # Simulate streaming tokens
                streaming_llm = rag_system.llm_config.create_llm()
                if streaming_llm.callbacks:
                    for char in "Streamed answer":
                        streaming_llm.callbacks[0].on_llm_new_token(char)

                return {
                    "result": "Streamed answer",
                    "source_documents": sample_source_docs
                }

            mock_chain.invoke = invoke_with_callbacks
            mock_qa.from_chain_type.return_value = mock_chain

            with patch.object(rag_system.llm_config, 'create_llm') as mock_create:
                mock_streaming_llm = Mock()
                mock_streaming_llm.callbacks = []
                mock_create.return_value = mock_streaming_llm

                result = rag_system.ask_question_stream("Test query", token_callback)

        # Tokens should have been collected
        assert len(collected_tokens) > 0


class TestErrorHandling:
    """Test error handling and error messages"""

    def test_llm_error_with_ollama_hints(self, rag_system):
        """Test error message includes Ollama troubleshooting hints"""
        with patch.object(rag_system.llm_config, 'get_display_name', return_value="Ollama Model"):
            with patch.object(rag_system, '_get_chain') as mock_chain:
                mock_chain.return_value.invoke.side_effect = Exception("Connection refused")

                with pytest.raises(RAGSystemError) as exc_info:
                    rag_system.ask_question("Test query")

                error_msg = str(exc_info.value)
                assert "Ollama" in error_msg
                assert "troubleshooting" in error_msg.lower()

    def test_llm_error_with_google_hints(self, rag_system):
        """Test error message includes Google API hints"""
        with patch.object(rag_system.llm_config, 'get_display_name', return_value="Google Gemini"):
            with patch.object(rag_system, '_get_chain') as mock_chain:
                mock_chain.return_value.invoke.side_effect = Exception("API key invalid")

                with pytest.raises(RAGSystemError) as exc_info:
                    rag_system.ask_question("Test query")

                error_msg = str(exc_info.value)
                assert "Google" in error_msg or "Gemini" in error_msg
                assert "API key" in error_msg

    def test_generic_error_handling(self, rag_system):
        """Test generic error handling"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            mock_chain.return_value.invoke.side_effect = ValueError("Generic error")

            with pytest.raises(RAGSystemError) as exc_info:
                rag_system.ask_question("Test query")

            assert "Generic error" in str(exc_info.value)

    def test_rag_system_error_passthrough(self, rag_system):
        """Test that RAGSystemError is passed through without wrapping"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            original_error = RAGSystemError("Custom error")
            mock_chain.return_value.invoke.side_effect = original_error

            with pytest.raises(RAGSystemError) as exc_info:
                rag_system.ask_question("Test query")

            # Should be the same error, not wrapped
            assert str(exc_info.value) == "Custom error"


class TestLLMInfo:
    """Test LLM information retrieval"""

    def test_get_llm_info(self, rag_system, mock_llm_config):
        """Test getting LLM display information"""
        mock_llm_config.get_display_name.return_value = "Test Model v1.0"

        info = rag_system.get_llm_info()

        assert info == "Test Model v1.0"


class TestCreateRAGSystem:
    """Test the factory function for creating RAG systems"""

    @patch('rag_system.LLMFactory')
    @patch('rag_system.DocumentStore')
    def test_create_rag_system_local_mode(self, mock_doc_store_class, mock_llm_factory):
        """Test creating RAG system in local mode"""
        mock_vector_store = Mock()
        mock_llm_config = Mock()
        mock_llm_factory.create_from_mode.return_value = mock_llm_config
        mock_doc_store = Mock()
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

    @patch('rag_system.LLMFactory')
    @patch('rag_system.DocumentStore')
    def test_create_rag_system_google_mode(self, mock_doc_store_class, mock_llm_factory):
        """Test creating RAG system in Google mode"""
        mock_vector_store = Mock()
        mock_llm_config = Mock()
        mock_llm_factory.create_from_mode.return_value = mock_llm_config
        mock_doc_store = Mock()
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

    @patch('rag_system.LLMFactory')
    @patch('rag_system.DocumentStore')
    def test_create_rag_system_with_bm25(self, mock_doc_store_class, mock_llm_factory):
        """Test creating RAG system with BM25 index"""
        mock_vector_store = Mock()
        mock_bm25_index = Mock()
        mock_bm25_chunks = [Mock()]
        mock_llm_config = Mock()
        mock_llm_factory.create_from_mode.return_value = mock_llm_config
        mock_doc_store = Mock()
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

    def test_complete_qa_workflow(self, rag_system, sample_source_docs):
        """Test complete question-answering workflow"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            mock_result = {
                "result": "Machine learning is a subset of AI.",
                "source_documents": sample_source_docs
            }
            mock_chain.return_value.invoke.return_value = mock_result

            # First query
            result1 = rag_system.ask_question("What is machine learning?")

            assert result1["answer"] == "Machine learning is a subset of AI."
            assert len(result1["source_documents"]) == 2

            # Second identical query should hit cache
            result2 = rag_system.ask_question("What is machine learning?")

            assert result1 == result2
            # Chain should only be called once due to cache
            assert mock_chain.call_count == 1

    def test_workflow_with_cache_disabled(self, rag_system_no_cache, sample_source_docs):
        """Test workflow with caching disabled"""
        with patch.object(rag_system_no_cache, '_get_chain') as mock_chain:
            mock_result = {
                "result": "Answer",
                "source_documents": sample_source_docs
            }
            mock_chain.return_value.invoke.return_value = mock_result

            result1 = rag_system_no_cache.ask_question("Query")
            result2 = rag_system_no_cache.ask_question("Query")

            # Chain should be called twice without caching
            assert mock_chain.call_count == 2

    def test_multiple_queries_different_questions(self, rag_system, sample_source_docs):
        """Test handling multiple different queries"""
        with patch.object(rag_system, '_get_chain') as mock_chain:
            def side_effect(query_dict):
                query = query_dict["query"]
                return {
                    "result": f"Answer to: {query}",
                    "source_documents": sample_source_docs
                }

            mock_chain.return_value.invoke.side_effect = side_effect

            questions = [
                "What is AI?",
                "How does machine learning work?",
                "What is deep learning?"
            ]

            results = [rag_system.ask_question(q) for q in questions]

            assert len(results) == 3
            assert all("answer" in r for r in results)
            # All should be cached
            assert len(rag_system.cache.cache) == 3


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
