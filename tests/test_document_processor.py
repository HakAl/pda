"""
Tests for document_processor.py

Covers:
- Document loading from various file types
- Retry logic and error handling
- BM25 index building
- Vector store creation
- Progress tracking
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from langchain_core.documents import Document

from document_processor import (
    DocumentProcessor,
    LoadResult,
    ProgressEmbeddings,
    create_progress_embeddings
)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings function"""
    embeddings = Mock()
    embeddings.embed_documents.return_value = [[0.1] * 384] * 5
    embeddings.embed_query.return_value = [0.1] * 384
    return embeddings


@pytest.fixture
def doc_processor(tmp_path, mock_embeddings):
    """Create DocumentProcessor with mocked embeddings"""
    with patch.object(DocumentProcessor, '_setup_embeddings', return_value=mock_embeddings):
        processor = DocumentProcessor(persist_directory=str(tmp_path / "chroma_db"))
    return processor


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(page_content="Test content 1", metadata={"source": "test1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "test2.pdf"}),
        Document(page_content="Test content 3", metadata={"source": "test3.txt"}),
    ]


class TestProgressEmbeddings:
    """Test progress tracking wrapper"""

    def test_embed_documents_updates_progress(self, mock_embeddings):
        """Test that embedding documents updates progress bar"""
        wrapper = ProgressEmbeddings(mock_embeddings, total_chunks=10)

        texts = ["text1", "text2", "text3"]
        result = wrapper.embed_documents(texts)

        assert wrapper.embedded_count == 3
        assert result == [[0.1] * 384] * 5
        mock_embeddings.embed_documents.assert_called_once_with(texts)
        wrapper.close()

    def test_embed_query_passthrough(self, mock_embeddings):
        """Test that query embedding passes through without tracking"""
        wrapper = ProgressEmbeddings(mock_embeddings, total_chunks=10)

        result = wrapper.embed_query("test query")

        assert result == [0.1] * 384
        assert wrapper.embedded_count == 0  # Query embedding shouldn't increment
        mock_embeddings.embed_query.assert_called_once_with("test query")
        wrapper.close()

    def test_context_manager(self, mock_embeddings):
        """Test progress embeddings context manager"""
        with create_progress_embeddings(mock_embeddings, 5) as wrapper:
            assert isinstance(wrapper, ProgressEmbeddings)
            wrapper.embed_documents(["test"])

        # Progress bar should be closed after context exits
        assert wrapper.pbar.n == 1  # One update was made


class TestDocumentLoading:
    """Test document loading functionality"""

    def test_load_documents_empty_folder(self, doc_processor, tmp_path):
        """Test loading from non-existent folder"""
        result = doc_processor.load_documents(str(tmp_path / "nonexistent"))

        assert isinstance(result, LoadResult)
        assert result.loaded_documents == []
        assert result.failed_files == []

    def test_load_documents_no_supported_files(self, doc_processor, tmp_path):
        """Test loading from folder with no supported files"""
        docs_folder = tmp_path / "documents"
        docs_folder.mkdir()

        # Create unsupported file
        (docs_folder / "test.xyz").write_text("unsupported")

        result = doc_processor.load_documents(str(docs_folder))

        assert result.loaded_documents == []
        assert result.failed_files == []

    @patch('document_processor.TextLoader')
    def test_load_single_txt_file(self, mock_loader_class, doc_processor, tmp_path):
        """Test loading a single text file"""
        docs_folder = tmp_path / "documents"
        docs_folder.mkdir()
        txt_file = docs_folder / "test.txt"
        txt_file.write_text("Test content")

        # Mock the loader
        mock_loader = Mock()
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={})
        ]
        mock_loader_class.return_value = mock_loader

        result = doc_processor.load_documents(str(docs_folder))

        assert len(result.loaded_documents) == 1
        assert result.loaded_documents[0].metadata['source'] == 'test.txt'
        assert result.loaded_documents[0].metadata['file_type'] == '.txt'
        assert len(result.failed_files) == 0

    @patch('document_processor.PyPDFLoader')
    def test_load_single_pdf_file(self, mock_loader_class, doc_processor, tmp_path):
        """Test loading a single PDF file"""
        docs_folder = tmp_path / "documents"
        docs_folder.mkdir()
        pdf_file = docs_folder / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_loader = Mock()
        mock_loader.load.return_value = [
            Document(page_content="PDF content", metadata={})
        ]
        mock_loader_class.return_value = mock_loader

        result = doc_processor.load_documents(str(docs_folder))

        assert len(result.loaded_documents) == 1
        assert result.loaded_documents[0].metadata['file_type'] == '.pdf'
        assert len(result.failed_files) == 0

    def test_retry_logic_on_failure(self, doc_processor, tmp_path):
        """Test that retry logic works on transient failures"""
        docs_folder = tmp_path / "documents"
        docs_folder.mkdir()
        txt_file = docs_folder / "test.txt"
        txt_file.write_text("Test content")

        with patch('document_processor.TextLoader') as mock_loader_class:
            mock_loader = Mock()
            # Fail twice, then succeed
            mock_loader.load.side_effect = [
                Exception("Transient error 1"),
                Exception("Transient error 2"),
                [Document(page_content="Success", metadata={})]
            ]
            mock_loader_class.return_value = mock_loader

            result = doc_processor.load_documents(str(docs_folder), retries=2)

            assert len(result.loaded_documents) == 1
            assert len(result.failed_files) == 0
            assert mock_loader.load.call_count == 3

    def test_retry_exhaustion(self, doc_processor, tmp_path):
        """Test that files are marked as failed after all retries"""
        docs_folder = tmp_path / "documents"
        docs_folder.mkdir()
        txt_file = docs_folder / "test.txt"
        txt_file.write_text("Test content")

        with patch('document_processor.TextLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load.side_effect = Exception("Persistent error")
            mock_loader_class.return_value = mock_loader

            result = doc_processor.load_documents(str(docs_folder), retries=2)

            assert len(result.loaded_documents) == 0
            assert len(result.failed_files) == 1
            assert "test.txt" in result.failed_files[0]['path']
            assert "Persistent error" in result.failed_files[0]['error']

    @patch('document_processor.PyPDFLoader')
    @patch('document_processor.TextLoader')
    def test_parallel_loading_multiple_files(self, mock_txt_loader, mock_pdf_loader,
                                             doc_processor, tmp_path):
        """Test parallel loading of multiple files"""
        docs_folder = tmp_path / "documents"
        docs_folder.mkdir()

        # Create multiple files
        (docs_folder / "test1.txt").write_text("Content 1")
        (docs_folder / "test2.txt").write_text("Content 2")
        (docs_folder / "test3.pdf").write_bytes(b"PDF content")

        # Mock loaders
        mock_txt = Mock()
        mock_txt.load.return_value = [Document(page_content="Text", metadata={})]
        mock_txt_loader.return_value = mock_txt

        mock_pdf = Mock()
        mock_pdf.load.return_value = [Document(page_content="PDF", metadata={})]
        mock_pdf_loader.return_value = mock_pdf

        result = doc_processor.load_documents(str(docs_folder))

        assert len(result.loaded_documents) == 3
        assert len(result.failed_files) == 0


class TestDocumentProcessing:
    """Test document processing pipeline"""

    def test_split_documents(self, doc_processor, sample_documents):
        """Test document splitting"""
        # Create long document
        long_doc = Document(
            page_content="word " * 1000,  # Long enough to be split
            metadata={"source": "long.txt"}
        )

        chunks = doc_processor._split_documents_batch([long_doc])

        assert len(chunks) > 1
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_build_bm25_index(self, doc_processor, sample_documents):
        """Test BM25 index building"""
        bm25_index, bm25_chunks = doc_processor._build_bm25_index(sample_documents)

        assert bm25_index is not None
        assert len(bm25_chunks) == len(sample_documents)
        assert bm25_chunks == sample_documents

    @patch.object(DocumentProcessor, 'load_documents')
    @patch.object(DocumentProcessor, '_create_vectorstore_batched')
    def test_process_documents_success(self, mock_create_vs, mock_load,
                                       doc_processor, sample_documents):
        """Test successful document processing pipeline"""
        mock_load.return_value = LoadResult(
            loaded_documents=sample_documents,
            failed_files=[]
        )
        mock_vs = Mock()
        mock_create_vs.return_value = mock_vs

        result = doc_processor.process_documents("./documents")

        assert result is not None
        vector_store, bm25_index, bm25_chunks = result
        assert vector_store == mock_vs
        assert bm25_index is not None
        assert len(bm25_chunks) > 0

    @patch.object(DocumentProcessor, 'load_documents')
    def test_process_documents_no_documents_loaded(self, mock_load, doc_processor):
        """Test processing when no documents are loaded"""
        mock_load.return_value = LoadResult(
            loaded_documents=[],
            failed_files=[{"path": "test.pdf", "error": "Failed to load"}]
        )

        result = doc_processor.process_documents("./documents")

        assert result is None

    @patch.object(DocumentProcessor, 'load_documents')
    @patch.object(DocumentProcessor, '_create_vectorstore_batched')
    def test_process_documents_with_failed_files(self, mock_create_vs, mock_load,
                                                 doc_processor, sample_documents):
        """Test processing with some failed files"""
        mock_load.return_value = LoadResult(
            loaded_documents=sample_documents,
            failed_files=[{"path": "bad.pdf", "error": "Corrupted"}]
        )
        mock_vs = Mock()
        mock_create_vs.return_value = mock_vs

        result = doc_processor.process_documents("./documents")

        assert result is not None  # Should still succeed with partial documents

    def test_process_documents_with_pre_chunked(self, doc_processor):
        """Test processing with pre-chunked documents (e.g., CSV data)"""
        pre_chunked = Document(
            page_content="Pre-chunked content",
            metadata={"source": "data.csv", "doc_type": "data_chunk"}
        )
        regular = Document(
            page_content="Regular content",
            metadata={"source": "doc.txt"}
        )

        with patch.object(DocumentProcessor, 'load_documents') as mock_load:
            mock_load.return_value = LoadResult(
                loaded_documents=[pre_chunked, regular],
                failed_files=[]
            )

            with patch.object(DocumentProcessor, '_create_vectorstore_batched') as mock_vs:
                mock_vs.return_value = Mock()
                result = doc_processor.process_documents()

                assert result is not None


class TestVectorStore:
    """Test vector store operations"""

    @patch('document_processor.Chroma')
    def test_create_vectorstore(self, mock_chroma, doc_processor, sample_documents):
        """Test vector store creation"""
        mock_vs = Mock()
        mock_chroma.from_documents.return_value = mock_vs

        vs = doc_processor._create_vectorstore_batched(sample_documents)

        assert vs == mock_vs
        mock_chroma.from_documents.assert_called_once()

    @patch('document_processor.Chroma')
    def test_load_existing_vectorstore_success(self, mock_chroma, doc_processor):
        """Test loading existing vector store"""
        mock_vs = Mock()
        mock_vs.get.return_value = {
            "documents": ["doc1", "doc2"],
            "metadatas": [{"source": "1"}, {"source": "2"}]
        }
        mock_chroma.return_value = mock_vs

        # Create the persist directory
        os.makedirs(doc_processor.persist_directory, exist_ok=True)

        vs, bm25_index, bm25_chunks = doc_processor.load_existing_vectorstore()

        assert vs == mock_vs
        assert bm25_index is not None
        assert len(bm25_chunks) == 2

    def test_load_existing_vectorstore_not_found(self, doc_processor):
        """Test loading when vector store doesn't exist"""
        vs, bm25_index, bm25_chunks = doc_processor.load_existing_vectorstore()

        assert vs is None
        assert bm25_index is None
        assert bm25_chunks is None

    @patch('document_processor.Chroma')
    def test_load_existing_vectorstore_empty(self, mock_chroma, doc_processor):
        """Test loading empty vector store"""
        mock_vs = Mock()
        mock_vs.get.return_value = {
            "documents": [],
            "metadatas": []
        }
        mock_chroma.return_value = mock_vs

        os.makedirs(doc_processor.persist_directory, exist_ok=True)

        vs, bm25_index, bm25_chunks = doc_processor.load_existing_vectorstore()

        assert vs == mock_vs
        assert bm25_index is None
        assert bm25_chunks is None


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_supported_extensions(self, doc_processor):
        """Test that supported extensions are properly defined"""
        assert '.pdf' in doc_processor.supported_extensions
        assert '.txt' in doc_processor.supported_extensions
        assert '.docx' in doc_processor.supported_extensions
        assert '.csv' in doc_processor.supported_extensions

    def test_empty_document_content(self, doc_processor):
        """Test handling of empty document content"""
        empty_docs = [Document(page_content="", metadata={"source": "empty.txt"})]

        chunks = doc_processor._split_documents_batch(empty_docs)

        # Should handle empty content gracefully
        assert isinstance(chunks, list)

    def test_very_small_chunks(self, doc_processor):
        """Test processing very small documents"""
        small_doc = Document(
            page_content="Small",
            metadata={"source": "small.txt"}
        )

        chunks = doc_processor._split_documents_batch([small_doc])

        assert len(chunks) >= 1
        assert chunks[0].page_content == "Small"
