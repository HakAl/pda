import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi
from tqdm import tqdm as tqdm_bar

class DocumentProcessor:
    def __init__(self, persist_directory="./chroma_db", api_key=None):
        self.persist_directory = persist_directory
        self.api_key = api_key
        
        # Choose embeddings based on availability
        self.embeddings = self._setup_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
    
    def _setup_embeddings(self):
        """Setup embeddings - always use local for compatibility"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        except ImportError:
            # Fallback to simpler embeddings if needed
            from langchain.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384)
    
    def load_documents(self, documents_folder="./documents"):
        """Load all documents from the specified folder"""
        documents = []
        
        if not os.path.exists(documents_folder):
            return documents
        
        for filename in tqdm_bar(os.listdir(documents_folder), desc="📄 Processing files"):
            file_path = os.path.join(documents_folder, filename)
            
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    # Add filename to metadata
                    for doc in loaded_docs:
                        doc.metadata['source'] = filename
                    documents.extend(loaded_docs)
                elif filename.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata['source'] = filename
                    documents.extend(loaded_docs)
                print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        
        return documents

    def _build_bm25_index(self, chunks):
        """Return BM25 index for the *already-split* chunks."""
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        return BM25Okapi(tokenized), chunks
    
    def process_documents(self, documents_folder="./documents"):
        """Process documents and create vector store"""
        print("📂 Loading documents...")
        documents = self.load_documents(documents_folder)
        
        if not documents:
            print("❌ No documents found in './documents' folder!")
            print("   Please add PDF or text files to the 'documents' folder.")
            return None
        
        print(f"📄 Loaded {len(documents)} document chunks")
        print("✂️ Splitting documents into chunks...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(tqdm_bar(documents, desc="✂️ Splitting"))
        self.bm25_index, self.bm25_chunks = self._build_bm25_index(chunks)
        print(f"🔢 Created {len(chunks)} chunks")
        
        # Create vector store
        print("🗄️ Creating vector database...")

        # progress bar on the *embeddings* (batches automatically)
        from tqdm import tqdm
        class TqdmHFEmbeddings(HuggingFaceEmbeddings):
            def embed_documents(self, texts):
                return super().embed_documents(tqdm_bar(texts, desc="🔢 Embedding chunks"))

        tqdm_emb = TqdmHFEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        vector_store = Chroma.from_documents(
            documents=chunks,  # plain list
            embedding=tqdm_emb,  # wrapped embeddings
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        print("✅ Vector database created successfully!")
        return vector_store, self.bm25_index, self.bm25_chunks

    def load_existing_vectorstore(self):
        """Return (chroma_vs, bm25_index, bm25_chunks) or (None, None, None)"""
        if not os.path.exists(self.persist_directory):
            return None, None, None

        try:
            vs = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # rebuild BM25 from the chunks already in Chroma
            raw = vs.get()
            chunks = [Document(page_content=d, metadata=m)
                      for d, m in zip(raw["documents"], raw["metadatas"])]
            self.bm25_index, self.bm25_chunks = self._build_bm25_index(chunks)
            return vs, self.bm25_index, self.bm25_chunks
        except Exception as e:
            print(f"❌ Error loading existing vector store: {e}")
            return None, None, None

    def _create_hyde_embeddings(self):
        """Return HyDE wrapper around the local phi3:mini."""
        base = self.embeddings   # your existing MiniLM model
        llm = OllamaLLM(model="phi3:mini", temperature=0.1)
        return HypotheticalDocumentEmbedder.from_llm(llm=llm, base_embeddings=base,
                                                     prompt_key="web_search")