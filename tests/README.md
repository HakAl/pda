# PDA Test Suite

Comprehensive test suite for the PDA (Personal Document Assistant) RAG application.

## Test Coverage

### Core Modules

- **`test_document_processor.py`** - Document loading, processing, chunking, and vector store creation
- **`test_rag_system.py`** - RAG pipeline, query processing, caching, and error handling
- **`test_hybrid_retriever.py`** - Hybrid retrieval (vector + BM25), reranking, and deduplication
- **`test_query_cache.py`** - Semantic caching and similarity matching
- **`test_document_store.py`** - Document storage and retriever configuration
- **`test_llm_factory.py`** - LLM configuration and factory functions (existing)
- **`test_retrieval_performance.py`** - Performance benchmarking (existing)

### Test Organization

```
tests/
├── __init__.py
├── conftest.py                      # Shared fixtures and configuration
├── test_document_processor.py        
├── test_rag_system.py                
├── test_hybrid_retriever.py         # todo
├── test_query_cache.py              # todo
├── test_document_store.py            
├── test_llm_factory.py              # todo
├── test_retrieval_performance.py    # monkey test
└── generate_test_data.py            # make bad data
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_document_processor.py
```

### Run Specific Test Class
```bash
pytest tests/test_rag_system.py::TestCaching
```

### Run Specific Test
```bash
pytest tests/test_hybrid_retriever.py::TestHybridRetrieval::test_retrieval_triggers_bm25_insufficient_docs
```

### Run with Coverage
```bash
pytest --cov=pda --cov-report=html
```

### Run with Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests requiring NLTK
pytest -m requires_nltk
```

### Run with Verbose Output
```bash
pytest -v
```

### Run and Show Print Statements
```bash
pytest -s
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests involving multiple components
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.requires_nltk` - Tests requiring NLTK data
- `@pytest.mark.requires_api` - Tests requiring API keys

## Fixtures

### Shared Fixtures (conftest.py)

- **`sample_documents`** - Standard set of test documents
- **`sample_chunks`** - Pre-chunked document segments
- **`sample_queries`** - Common test queries
- **`temp_dir`** - Temporary directory for file operations
- **`documents_folder`** - Folder with sample test files
- **`mock_embeddings`** - Mocked embedding function
- **`mock_llm`** - Mocked language model
- **`mock_vector_store`** - Mocked ChromaDB instance
- **`mock_bm25_index`** - Mocked BM25 index
- **`mock_document_store`** - Complete mocked document store
- **`mock_cache`** - Mocked semantic cache

## Writing New Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestYourFeature:
    """Test suite for specific feature"""

    def test_basic_functionality(self, fixture1, fixture2):
        """Test basic feature behavior"""
        # Arrange
        component = YourComponent(fixture1)
        
        # Act
        result = component.method(fixture2)
        
        # Assert
        assert result == expected_value

    def test_error_handling(self):
        """Test error conditions"""
        with pytest.raises(YourException):
            component.method_that_fails()

    @pytest.mark.slow
    def test_performance(self):
        """Test performance-critical code"""
        # Performance test code
        pass
```

### Mocking Guidelines

1. **Mock external dependencies** (APIs, file I/O, network calls)
2. **Don't mock the code you're testing**
3. **Use fixtures for complex mocks**
4. **Verify mock calls when testing interactions**

```python
from unittest.mock import Mock, patch, call

def test_with_mocks():
    mock_service = Mock()
    mock_service.method.return_value = "result"
    
    # Use the mock
    result = component.use_service(mock_service)
    
    # Verify interactions
    mock_service.method.assert_called_once_with("expected_arg")
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

## Test Data

### Generating Test Data

Use the `generate_test_data.py` script to create test documents:

```python
from tests.generate_test_data import create_test_pdf, create_test_txt

# Create test files
create_test_pdf("test.pdf", "Test content")
create_test_txt("test.txt", "Test content")
```

### Using Fixtures for Test Data

```python
@pytest.fixture
def my_test_data():
    """Custom test data fixture"""
    return {
        "documents": [...],
        "expected_output": ...
    }

def test_with_custom_data(my_test_data):
    result = process(my_test_data["documents"])
    assert result == my_test_data["expected_output"]
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=pda --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Coverage Goals

Target coverage by module:

- **document_processor.py**: 85%+
- **rag_system.py**: 90%+
- **hybrid_retriever.py**: 85%+
- **query_cache.py**: 90%+
- **document_store.py**: 80%+
- **llm_factory.py**: 85%+

### Viewing Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=pda --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Troubleshooting

### Common Issues

**Tests fail with import errors:**
```bash
# Make sure you're in the project root
cd /path/to/pda

# Install in development mode
pip install -e .
```

**NLTK data missing:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**ChromaDB issues:**
```bash
# Clear test database
rm -rf ./chroma_db_test
```

**Mock not working:**
- Ensure you're patching the correct import path
- Use `patch.object()` for class methods
- Check mock is applied before code execution

### Debug Mode

Run tests with Python debugger:

```bash
pytest --pdb  # Drop into debugger on failure
pytest -x     # Stop on first failure
pytest --lf   # Run last failed tests only
```

## Best Practices

1. **Test one thing at a time** - Each test should verify a single behavior
2. **Use descriptive test names** - Name should explain what is being tested
3. **Follow AAA pattern** - Arrange, Act, Assert
4. **Keep tests independent** - Tests shouldn't depend on each other
5. **Use fixtures for setup** - Avoid repetitive setup code
6. **Mock external dependencies** - Tests should be fast and reliable
7. **Test edge cases** - Empty inputs, None values, boundary conditions
8. **Write tests first (TDD)** - When fixing bugs, write failing test first
9. **Keep tests maintainable** - Refactor tests along with code
10. **Document complex tests** - Add comments for non-obvious test logic

## Performance Testing

### Benchmarking

```python
import time

def test_retrieval_performance(benchmark_timer):
    benchmark_timer.start()
    
    # Code to benchmark
    result = retriever.get_relevant_documents(query)
    
    benchmark_timer.stop()
    
    assert benchmark_timer.elapsed < 1.0  # Should complete in < 1 second
```

### Load Testing

```python
@pytest.mark.slow
def test_high_volume_queries():
    """Test system under load"""
    queries = [f"Query {i}" for i in range(1000)]
    
    start = time.time()
    results = [rag_system.ask_question(q) for q in queries]
    elapsed = time.time() - start
    
    assert len(results) == 1000
    assert elapsed < 60  # Should complete in reasonable time
```

## Contributing

When adding new tests:

1. Place tests in appropriate test file based on module
2. Use existing fixtures when possible
3. Add new fixtures to `conftest.py` if reusable
4. Mark tests appropriately (`@pytest.mark.unit`, etc.)
5. Ensure tests pass locally before committing
6. Update this README if adding new test categories

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)