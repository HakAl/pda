import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

class DocumentProcessor:
    def __init__(self, persist_directory="./chroma_db", api_key=None):
        self.persist_directory = persist_directory
        self.api_key = api_key
        
        # Choose embeddings based on availability
        self.embeddings = self._setup_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller for better performance
            chunk_overlap=100,
            length_function=len,
        )
    
    def _setup_embeddings(self):
        """Setup embeddings - always use local for compatibility"""
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
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
        
        for filename in os.listdir(documents_folder):
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
        chunks = self.text_splitter.split_documents(documents)
        print(f"🔢 Created {len(chunks)} chunks")
        
        # Create vector store
        print("🗄️ Creating vector database...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vector_store.persist()
        
        print("✅ Vector database created successfully!")
        return vector_store
    
    def load_existing_vectorstore(self):
        """Load existing vector store if available"""
        if os.path.exists(self.persist_directory):
            try:
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"❌ Error loading existing vector store: {e}")
                print("   Creating new vector store...")
        return None