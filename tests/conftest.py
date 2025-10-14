"""
Shared pytest fixtures and configuration for the PDA test suite.

This file provides common test fixtures, mock objects, and utilities
used across multiple test modules.
"""
import sys
from pathlib import Path

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_documents():
    """Create a list of sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml_intro.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "dl_basics.pdf", "page": 1}
        ),
        Document(
            page_content="Natural language processing enables computers to understand text.",
            metadata={"source": "nlp_guide.pdf", "page": 1}
        ),
        Document(
            page_content="Computer vision allows machines to interpret visual information.",
            metadata={"source": "cv_intro.pdf", "page": 1}
        ),
        Document(
            page_content="Reinforcement learning involves agents learning through interaction.",
            metadata={"source": "rl_basics.pdf", "page": 1}
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing retrieval."""
    return [
        Document(
            page_content="Chunk 1: Introduction to AI",
            metadata={"source": "doc1.pdf", "chunk_id": 0}
        ),
        Document(
            page_content="Chunk 2: Machine learning fundamentals",
            metadata={"source": "doc1.pdf", "chunk_id": 1}
        ),
        Document(
            page_content="Chunk 3: Deep learning architectures",
            metadata={"source": "doc2.pdf", "chunk_id": 0}
        ),
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain natural language processing",
        "What are neural networks?",
    ]


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def documents_folder(temp_dir):
    """Create a documents folder with sample files."""
    docs_folder = temp_dir / "documents"
    docs_folder.mkdir()

    # Create sample text file
    (docs_folder / "sample.txt").write_text("This is a sample text document.")

    # Create sample CSV
    (docs_folder / "data.csv").write_text("column1,column2\nvalue1,value2\n")

    return docs_folder


@pytest.fixture
def empty_documents_folder(temp_dir):
    """Create an empty documents folder."""
    docs_folder = temp_dir / "empty_documents"
    docs_folder.mkdir()
    return docs_folder


# ============================================================================
# Mock LLM and Embeddings Fixtures
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Mock embeddings function."""
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 10
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return embeddings


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.invoke.return_value = "This is a mocked LLM response."
    llm.predict.return_value = "This is a mocked LLM response."
    return llm


@pytest.fixture
def mock_llm_config(mock_llm):
    """Mock LLM configuration."""
    config = Mock()
    config.create_llm.return_value = mock_llm
    config.get_prompt_template.return_value = Mock()
    config.get_display_name.return_value = "Mock LLM"
    return config


# ============================================================================
# Vector Store Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store(sample_chunks):
    """Mock ChromaDB vector store."""
    store = Mock(spec=Chroma)

    # Mock similarity search
    store.similarity_search.return_value = sample_chunks[:3]
    store.similarity_search_with_score.return_value = [
        (chunk, 0.9 - i * 0.1) for i, chunk in enumerate(sample_chunks[:3])
    ]

    # Mock as_retriever
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = sample_chunks[:3]
    store.as_retriever.return_value = mock_retriever

    # Mock get for loading existing store
    store.get.return_value = {
        "documents": [chunk.page_content for chunk in sample_chunks],
        "metadatas": [chunk.metadata for chunk in sample_chunks],
        "ids": [f"id_{i}" for i in range(len(sample_chunks))],
    }

    return store


@pytest.fixture
def mock_chroma_db(temp_dir):
    """Create a mock Chroma database directory."""
    chroma_dir = temp_dir / "chroma_db"
    chroma_dir.mkdir()
    return chroma_dir


# ============================================================================
# BM25 Fixtures
# ============================================================================

@pytest.fixture
def mock_bm25_index():
    """Mock BM25 index."""
    import numpy as np
    index = Mock()
    index.get_scores.return_value = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
    return index


@pytest.fixture
def mock_bm25_chunks(sample_documents):
    """Sample documents for BM25 index."""
    return sample_documents


# ============================================================================
# Document Store Fixtures
# ============================================================================

@pytest.fixture
def mock_document_store(mock_vector_store, mock_embeddings):
    """Mock document store."""
    store = Mock()
    store.vector_store = mock_vector_store
    store.get_retriever.return_value = mock_vector_store.as_retriever()
    store.get_embedding_function.return_value = mock_embeddings
    return store


# ============================================================================
# Query Cache Fixtures
# ============================================================================

@pytest.fixture
def mock_cache():
    """Mock semantic query cache."""
    cache = Mock()
    cache.cache = {}
    cache.get.return_value = None
    cache.set.return_value = None
    cache.threshold = 0.85
    cache.max_size = 100
    return cache


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_app_config():
    """Mock application configuration."""
    config = Mock()

    # Chunking config
    config.chunking.chunk_size = 1000
    config.chunking.chunk_overlap = 200

    # Retriever config
    config.retriever.vector_k = 5
    config.retriever.vector_fetch_k = 20
    config.retriever.bm25_top_k = 3
    config.retriever.lambda_mult = 0.5
    config.retriever.min_docs_before_bm25 = 3

    # Reranker config
    config.reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    config.reranker.top_k = 2
    config.reranker.batch_size = 32

    return config


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_documents_equal():
    """Helper function to compare documents."""

    def _assert_equal(doc1: Document, doc2: Document):
        assert doc1.page_content == doc2.page_content
        assert doc1.metadata == doc2.metadata

    return _assert_equal


@pytest.fixture
def create_test_document():
    """Factory fixture to create test documents."""

    def _create(content: str, source: str = "test.pdf", **metadata):
        return Document(
            page_content=content,
            metadata={"source": source, **metadata}
        )

    return _create


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_nltk: marks tests that require NLTK data"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API keys"
    )


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_nltk_available():
    """Mock NLTK availability."""
    import sys
    from unittest.mock import MagicMock

    # Mock nltk modules
    sys.modules['nltk'] = MagicMock()
    sys.modules['nltk.corpus'] = MagicMock()
    sys.modules['nltk.tokenize'] = MagicMock()

    yield

    # Clean up
    if 'nltk' in sys.modules:
        del sys.modules['nltk']
    if 'nltk.corpus' in sys.modules:
        del sys.modules['nltk.corpus']
    if 'nltk.tokenize' in sys.modules:
        del sys.modules['nltk.tokenize']


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup happens automatically via temp_dir fixture