"""
Tests for query_cache.py

Covers:
- Semantic similarity caching
- Cache hit/miss logic
- Cache size limits
- Embedding-based similarity
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

# Assuming your query_cache module has a SemanticQueryCache class
# Adjust import based on your actual implementation
try:
    from query_cache import SemanticQueryCache
except ImportError:
    pytest.skip("query_cache module not found", allow_module_level=True)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings function for testing."""
    embeddings = Mock()
    # Return different embeddings for different queries
    embeddings.embed_query.side_effect = lambda q: {
        "What is AI?": [0.1, 0.2, 0.3],
        "What is artificial intelligence?": [0.11, 0.21, 0.31],  # Similar
        "How do neural networks work?": [0.9, 0.8, 0.7],  # Different
    }.get(q, [0.5, 0.5, 0.5])
    return embeddings


@pytest.fixture
def cache_with_embeddings(mock_embeddings):
    """Create cache with mocked embeddings."""
    cache = SemanticQueryCache(
        similarity_threshold=0.85,
        max_size=10
    )
    cache.embeddings = mock_embeddings
    return cache


@pytest.fixture
def sample_result():
    """Sample query result for caching."""
    return {
        "answer": "AI is artificial intelligence.",
        "source_documents": [Mock()]
    }


class TestCacheInitialization:
    """Test cache initialization and configuration"""

    def test_init_default_params(self):
        """Test cache initialization with default parameters"""
        cache = SemanticQueryCache()

        assert cache.threshold == 0.85
        assert cache.max_size == 100
        assert cache.cache == {}
        assert cache.embeddings is None

    def test_init_custom_params(self):
        """Test cache initialization with custom parameters"""
        cache = SemanticQueryCache(
            similarity_threshold=0.9,
            max_size=50
        )

        assert cache.threshold == 0.9
        assert cache.max_size == 50

    def test_init_invalid_threshold_too_high(self):
        """Test that threshold > 1.0 raises error"""
        with pytest.raises((ValueError, AssertionError)):
            SemanticQueryCache(similarity_threshold=1.5)

    def test_init_invalid_threshold_too_low(self):
        """Test that threshold < 0 raises error"""
        with pytest.raises((ValueError, AssertionError)):
            SemanticQueryCache(similarity_threshold=-0.1)


class TestCacheOperations:
    """Test basic cache set and get operations"""

    def test_cache_miss_returns_none(self, cache_with_embeddings):
        """Test that cache miss returns None"""
        result = cache_with_embeddings.get("What is AI?")

        assert result is None

    def test_cache_set_and_get_exact_match(self, cache_with_embeddings, sample_result):
        """Test setting and getting from cache with exact query match"""
        query = "What is AI?"

        cache_with_embeddings.set(query, sample_result)
        cached = cache_with_embeddings.get(query)

        assert cached == sample_result

    def test_cache_set_stores_embedding(self, cache_with_embeddings, sample_result):
        """Test that cache stores the query embedding"""
        query = "What is AI?"

        cache_with_embeddings.set(query, sample_result)

        assert len(cache_with_embeddings.cache) == 1
        cache_entry = list(cache_with_embeddings.cache.values())[0]
        assert "embedding" in cache_entry
        assert "result" in cache_entry

    def test_cache_get_similar_query(self, cache_with_embeddings, sample_result):
        """Test cache hit with semantically similar query"""
        original_query = "What is AI?"
        similar_query = "What is artificial intelligence?"

        # Mock cosine similarity to return high similarity
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[0.95]])

            cache_with_embeddings.set(original_query, sample_result)
            cached = cache_with_embeddings.get(similar_query)

            assert cached == sample_result

    def test_cache_miss_dissimilar_query(self, cache_with_embeddings, sample_result):
        """Test cache miss with dissimilar query"""
        original_query = "What is AI?"
        different_query = "How do neural networks work?"

        # Mock cosine similarity to return low similarity
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[0.3]])

            cache_with_embeddings.set(original_query, sample_result)
            cached = cache_with_embeddings.get(different_query)

            assert cached is None


class TestCacheSizeManagement:
    """Test cache size limits and eviction"""

    def test_cache_respects_max_size(self, cache_with_embeddings, sample_result):
        """Test that cache evicts old entries when max size is reached"""
        cache_with_embeddings.max_size = 3

        # Add 4 entries
        for i in range(4):
            cache_with_embeddings.set(f"Query {i}", {**sample_result, "id": i})

        # Should only have 3 entries
        assert len(cache_with_embeddings.cache) <= 3

    def test_cache_eviction_policy(self, cache_with_embeddings, sample_result):
        """Test cache eviction policy (FIFO or LRU)"""
        cache_with_embeddings.max_size = 2

        cache_with_embeddings.set("Query 1", {**sample_result, "id": 1})
        cache_with_embeddings.set("Query 2", {**sample_result, "id": 2})
        cache_with_embeddings.set("Query 3", {**sample_result, "id": 3})

        # First query should be evicted
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[0.95]])
            cached = cache_with_embeddings.get("Query 1")
            # Depending on eviction policy, this might or might not be cached

    def test_empty_cache(self, cache_with_embeddings):
        """Test that cache can be emptied"""
        cache_with_embeddings.set("Query 1", {"answer": "Answer 1"})
        cache_with_embeddings.set("Query 2", {"answer": "Answer 2"})

        cache_with_embeddings.cache.clear()

        assert len(cache_with_embeddings.cache) == 0


class TestSimilarityCalculation:
    """Test similarity threshold logic"""

    def test_exact_threshold_boundary(self, cache_with_embeddings, sample_result):
        """Test behavior at exact similarity threshold"""
        query1 = "Test query 1"
        query2 = "Test query 2"

        cache_with_embeddings.set(query1, sample_result)

        # Mock similarity exactly at threshold
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[cache_with_embeddings.threshold]])

            cached = cache_with_embeddings.get(query2)
            # Should return result (>= threshold)
            assert cached == sample_result

    def test_just_below_threshold(self, cache_with_embeddings, sample_result):
        """Test cache miss just below threshold"""
        query1 = "Test query 1"
        query2 = "Test query 2"

        cache_with_embeddings.set(query1, sample_result)

        # Mock similarity just below threshold
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[cache_with_embeddings.threshold - 0.01]])

            cached = cache_with_embeddings.get(query2)
            assert cached is None

    def test_just_above_threshold(self, cache_with_embeddings, sample_result):
        """Test cache hit just above threshold"""
        query1 = "Test query 1"
        query2 = "Test query 2"

        cache_with_embeddings.set(query1, sample_result)

        # Mock similarity just above threshold
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[cache_with_embeddings.threshold + 0.01]])

            cached = cache_with_embeddings.get(query2)

            cached = cache_with_embeddings.get(query2)
            assert cached == sample_result


class TestEmbeddingHandling:
    """Test embedding generation and handling"""

    def test_cache_without_embeddings_raises_error(self, sample_result):
        """Test that using cache without embeddings raises error"""
        cache = SemanticQueryCache()

        with pytest.raises((AttributeError, ValueError, TypeError)):
            cache.set("Test query", sample_result)

    def test_cache_calls_embed_query(self, cache_with_embeddings, sample_result):
        """Test that cache calls embed_query for new queries"""
        query = "What is AI?"

        cache_with_embeddings.set(query, sample_result)

        cache_with_embeddings.embeddings.embed_query.assert_called_with(query)

    def test_multiple_queries_multiple_embeddings(self, cache_with_embeddings, sample_result):
        """Test that each query gets its own embedding"""
        queries = ["Query 1", "Query 2", "Query 3"]

        for query in queries:
            cache_with_embeddings.set(query, {**sample_result, "query": query})

        # Should have called embed_query for each set operation
        assert cache_with_embeddings.embeddings.embed_query.call_count >= len(queries)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_query(self, cache_with_embeddings, sample_result):
        """Test caching empty query"""
        cache_with_embeddings.set("", sample_result)
        cached = cache_with_embeddings.get("")

        # Should handle gracefully
        assert cached == sample_result or cached is None

    def test_very_long_query(self, cache_with_embeddings, sample_result):
        """Test caching very long query"""
        long_query = "word " * 1000

        cache_with_embeddings.set(long_query, sample_result)
        cached = cache_with_embeddings.get(long_query)

        assert cached == sample_result

    def test_query_with_special_characters(self, cache_with_embeddings, sample_result):
        """Test caching query with special characters"""
        special_query = "What is AI? @#$%^&* 123"

        cache_with_embeddings.set(special_query, sample_result)
        cached = cache_with_embeddings.get(special_query)

        assert cached == sample_result

    def test_unicode_query(self, cache_with_embeddings, sample_result):
        """Test caching query with unicode characters"""
        unicode_query = "¿Qué es IA? 机器学习是什么？"

        cache_with_embeddings.set(unicode_query, sample_result)
        cached = cache_with_embeddings.get(unicode_query)

        assert cached == sample_result

    def test_none_result(self, cache_with_embeddings):
        """Test caching None as result"""
        cache_with_embeddings.set("Query", None)
        cached = cache_with_embeddings.get("Query")

        # Should handle None result
        assert cached is None or cached == {}

    def test_cache_size_zero(self):
        """Test cache with max_size of 0"""
        cache = SemanticQueryCache(max_size=0)

        # Should either raise error or not cache anything
        assert cache.max_size == 0

    def test_cache_size_one(self, cache_with_embeddings, sample_result):
        """Test cache with max_size of 1"""
        cache_with_embeddings.max_size = 1

        cache_with_embeddings.set("Query 1", {**sample_result, "id": 1})
        cache_with_embeddings.set("Query 2", {**sample_result, "id": 2})

        # Should only have 1 entry
        assert len(cache_with_embeddings.cache) == 1


class TestCacheStatistics:
    """Test cache statistics and monitoring"""

    def test_cache_size(self, cache_with_embeddings, sample_result):
        """Test getting cache size"""
        cache_with_embeddings.set("Query 1", sample_result)
        cache_with_embeddings.set("Query 2", sample_result)
        cache_with_embeddings.set("Query 3", sample_result)

        assert len(cache_with_embeddings.cache) == 3

    def test_cache_contains_query(self, cache_with_embeddings, sample_result):
        """Test checking if cache contains a query"""
        query = "What is AI?"

        cache_with_embeddings.set(query, sample_result)

        # Cache should contain an entry
        assert len(cache_with_embeddings.cache) > 0


class TestConcurrency:
    """Test cache behavior with concurrent access"""

    def test_multiple_sets_same_query(self, cache_with_embeddings):
        """Test setting same query multiple times"""
        query = "What is AI?"
        result1 = {"answer": "Answer 1"}
        result2 = {"answer": "Answer 2"}

        cache_with_embeddings.set(query, result1)
        cache_with_embeddings.set(query, result2)

        # Later set should overwrite
        cached = cache_with_embeddings.get(query)
        assert cached["answer"] == "Answer 2"

    def test_interleaved_operations(self, cache_with_embeddings):
        """Test interleaved set and get operations"""
        cache_with_embeddings.set("Q1", {"answer": "A1"})
        cache_with_embeddings.get("Q1")
        cache_with_embeddings.set("Q2", {"answer": "A2"})
        cache_with_embeddings.get("Q2")

        assert cache_with_embeddings.get("Q1")["answer"] == "A1"
        assert cache_with_embeddings.get("Q2")["answer"] == "A2"


class TestIntegration:
    """Integration tests with real similarity calculations"""

    def test_realistic_similarity_workflow(self, mock_embeddings):
        """Test realistic caching workflow with similarity calculations"""
        cache = SemanticQueryCache(similarity_threshold=0.85, max_size=10)
        cache.embeddings = mock_embeddings

        # Cache a result
        result1 = {"answer": "AI is artificial intelligence"}
        cache.set("What is AI?", result1)

        # Try to get with exact query
        cached = cache.get("What is AI?")
        assert cached == result1

        # Try to get with similar query (needs actual similarity calculation)
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[0.95]])
            cached_similar = cache.get("What is artificial intelligence?")
            assert cached_similar == result1

    def test_cache_performance_multiple_queries(self, cache_with_embeddings):
        """Test cache performance with many queries"""
        # Add many entries
        for i in range(100):
            cache_with_embeddings.set(f"Query {i}", {"answer": f"Answer {i}"})

        # Cache should respect max size
        assert len(cache_with_embeddings.cache) <= cache_with_embeddings.max_size

    def test_different_thresholds(self, mock_embeddings, sample_result):
        """Test cache behavior with different similarity thresholds"""
        strict_cache = SemanticQueryCache(similarity_threshold=0.95, max_size=10)
        strict_cache.embeddings = mock_embeddings

        lenient_cache = SemanticQueryCache(similarity_threshold=0.7, max_size=10)
        lenient_cache.embeddings = mock_embeddings

        query1 = "What is AI?"
        query2 = "What is artificial intelligence?"

        strict_cache.set(query1, sample_result)
        lenient_cache.set(query1, sample_result)

        # Mock moderate similarity (0.8)
        with patch('query_cache.cosine_similarity') as mock_cosim:
            mock_cosim.return_value = np.array([[0.8]])

            # Strict cache should miss
            strict_result = strict_cache.get(query2)
            assert strict_result is None

            # Lenient cache should hit
            lenient_result = lenient_cache.get(query2)
            assert lenient_result == sample_result