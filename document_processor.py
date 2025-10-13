import os
from typing import List, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


class DocumentProcessor:
    def __init__(self, persist_directory="./chroma_db", api_key=None):
        self.persist_directory = persist_directory
        self.api_key = api_key
        self.embeddings = self._setup_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
        self.bm25_index = None
        self.bm25_chunks = None

    def _setup_embeddings(self):
        """Setup embeddings with caching enabled"""
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
                cache_folder="./model_cache"  # Cache model locally
            )
        except ImportError:
            from langchain.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384)

    def load_documents(self, documents_folder="./documents") -> List[Document]:
        """Load documents with optimized parallel processing"""
        if not os.path.exists(documents_folder):
            print(f"⚠️  Folder '{documents_folder}' not found")
            return []

        # Get all files upfront
        file_list = [
            os.path.join(documents_folder, f)
            for f in os.listdir(documents_folder)
            if f.lower().endswith(('.pdf', '.txt'))
        ]

        if not file_list:
            return []

        documents = []
        # Use optimal worker count
        max_workers = min(len(file_list), os.cpu_count() or 4, 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and track with progress bar
            future_to_file = {
                executor.submit(self._load_single_document, fp): fp
                for fp in file_list
            }

            with tqdm(total=len(file_list), desc="📄 Loading files") as pbar:
                for future in as_completed(future_to_file):
                    try:
                        docs = future.result()
                        if docs:
                            documents.extend(docs)
                    except Exception as e:
                        file_path = future_to_file[future]
                        print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
                    finally:
                        pbar.update(1)

        return documents

    def _load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document with error handling"""
        try:
            filename = os.path.basename(file_path)

            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                return []

            loaded_docs = loader.load()

            # Add filename to metadata
            for doc in loaded_docs:
                doc.metadata['source'] = filename
                doc.metadata['file_path'] = file_path

            return loaded_docs

        except Exception as e:
            # Re-raise to be caught by executor
            raise Exception(f"Failed to load {os.path.basename(file_path)}: {str(e)}")

    def _build_bm25_index(self, chunks: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
        """Build BM25 index with pre-tokenized corpus"""
        print("🔍 Building BM25 index...")
        # Tokenize once and cache
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        return BM25Okapi(tokenized), chunks

    def _split_documents_batch(self, documents: List[Document]) -> List[Document]:
        """Split documents efficiently without redundant progress bar"""
        return self.text_splitter.split_documents(documents)

    def process_documents(self, documents_folder="./documents") -> Optional[Tuple]:
        """Process documents and create vector store with optimizations"""
        print("📂 Loading documents...")
        documents = self.load_documents(documents_folder)

        if not documents:
            print("❌ No documents found!")
            print(f"   Please add PDF or text files to '{documents_folder}'")
            return None

        print(f"✅ Loaded {len(documents)} document pages")
        print("✂️  Splitting into chunks...")

        # Split without nested progress bar
        chunks = self._split_documents_batch(documents)
        print(f"✅ Created {len(chunks)} chunks")

        # Build BM25 index
        self.bm25_index, self.bm25_chunks = self._build_bm25_index(chunks)

        # Create vector store with batched embeddings
        print("🗄️  Creating vector database...")
        vector_store = self._create_vectorstore_batched(chunks)

        print("✅ Vector database created successfully!")
        return vector_store, self.bm25_index, self.bm25_chunks

    def _create_vectorstore_batched(self, chunks: List[Document]) -> Chroma:
        """Create vector store with progress tracking via wrapper"""
        from langchain_core.embeddings import Embeddings

        # Create a wrapper class that tracks progress
        class ProgressEmbeddings(Embeddings):
            def __init__(self, base_embeddings, total_chunks):
                self.base = base_embeddings
                self.pbar = tqdm(total=total_chunks, desc="🔢 Embedding chunks")
                self.embedded_count = 0

            def embed_documents(self, texts):
                result = self.base.embed_documents(texts)
                self.pbar.update(len(texts))
                self.embedded_count += len(texts)
                return result

            def embed_query(self, text):
                return self.base.embed_query(text)

            def close(self):
                self.pbar.close()

        # Wrap embeddings with progress tracker
        progress_embeddings = ProgressEmbeddings(self.embeddings, len(chunks))

        try:
            # Single batch creation - let Chroma handle optimization
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=progress_embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
        finally:
            progress_embeddings.close()

        return vector_store

    def load_existing_vectorstore(self) -> Tuple[Optional[Chroma], Optional[BM25Okapi], Optional[List[Document]]]:
        """Load existing vector store and rebuild BM25 index"""
        if not os.path.exists(self.persist_directory):
            print(f"ℹ️  No existing vector store found at '{self.persist_directory}'")
            return None, None, None

        try:
            print("📥 Loading existing vector store...")
            vs = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            # Rebuild BM25 from stored chunks
            print("🔍 Rebuilding BM25 index...")
            raw = vs.get()

            if not raw["documents"]:
                print("⚠️  Vector store is empty")
                return vs, None, None

            chunks = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(raw["documents"], raw["metadatas"])
            ]

            self.bm25_index, self.bm25_chunks = self._build_bm25_index(chunks)
            print(f"✅ Loaded {len(chunks)} chunks from existing store")

            return vs, self.bm25_index, self.bm25_chunks

        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return None, None, None

    def _create_hyde_embeddings(self):
        """Create HyDE embeddings wrapper"""
        llm = OllamaLLM(model="phi3:mini", temperature=0.1)
        return HypotheticalDocumentEmbedder.from_llm(
            llm=llm,
            base_embeddings=self.embeddings,
            prompt_key="web_search"
        )