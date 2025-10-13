import os
import sys
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentQAAssistant:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.processor = None
        self.rag_system = None
        self.current_mode = None
        self.setup()
    
    def setup(self):
        """Setup the RAG system with mode selection"""
        print("🚀 Setting up Document Q&A Assistant...")
        print("=" * 50)
        
        # Let user choose mode
        self.choose_mode()
        
        # Initialize document processor
        self.processor = DocumentProcessor(
            api_key=self.api_key if self.current_mode == "google" else None
        )
        
        # Try to load existing vector store
        # Try to load existing vector store + BM25
        vector_store, bm25_index, bm25_chunks = self.processor.load_existing_vectorstore()

        if vector_store is None:  # nothing on disk
            print("\nNo existing database found. Processing documents...")
            if not os.path.exists("./documents"):
                os.makedirs("./documents")
                print("📁 Created 'documents' folder. Please add files there and restart.")
                return

            # ***NEW*** returns triple
            vector_store, bm25_index, bm25_chunks = self.processor.process_documents("./documents")
            if vector_store is None:
                return

        # Initialize RAG system with BM25 objects
        self.rag_system = RAGSystem(
            vector_store=vector_store,
            mode=self.current_mode,
            api_key=self.api_key if self.current_mode == "google" else None,
            bm25_index=bm25_index,
            bm25_chunks=bm25_chunks
        )
        
        print(f"\n✅ {self.current_mode.upper()} Mode RAG System ready!")
        print("You can now ask questions about your documents.")
    
    def choose_mode(self):
        """Let user choose between local and Google Gen AI mode"""
        print("\n🔧 Choose Processing Mode:")
        print("1. Local Mode (Private, Offline)")
        print("   - Uses Ollama with local models")
        print("   - 100% private - no data leaves your computer")
        print("   - Requires adequate RAM and Ollama installation")
        print("   - Slower but completely offline")
        
        print("\n2. Google Gen AI Mode (Fast, Cloud)")
        print("   - Uses Google's Gemini models")
        print("   - Very fast responses")
        print("   - Requires internet connection and API key")
        print("   - Data is sent to Google's servers")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                self.current_mode = "local"
                # Verify Ollama is available
                if not self.check_ollama_available():
                    print("❌ Ollama not found. Please install Ollama from https://ollama.ai/")
                    print("   Then run: ollama pull phi3:mini")
                    sys.exit(1)
                break
            elif choice == "2":
                self.current_mode = "google"
                if not self.api_key:
                    print("❌ GOOGLE_API_KEY not found in .env file")
                    print("   Please add your Google API key to the .env file")
                    sys.exit(1)
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def check_ollama_available(self):
        """Check if Ollama is installed and accessible"""
        try:
            import ollama
            # Try to list models to verify Ollama is running
            models = ollama.list()
            return True
        except:
            return False
    
    def chat_loop(self):
        """Main chat loop"""
        if self.rag_system is None:
            print("System not ready. Please check if documents are available.")
            return
        
        print(f"\n💬 Document Q&A Assistant ({self.current_mode.upper()} Mode)")
        print("Type 'quit' to exit, 'help' for commands, 'mode' to switch modes\n")
        
        while True:
            try:
                question = input("\n📝 Your question: ").strip()
                
                if question.lower() == 'quit':
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif question.lower() == 'mode':
                    self.switch_mode()
                    continue
                elif question.lower() == 'reload':
                    self.reload_documents()
                    continue
                elif not question:
                    continue
                
                print(f"\n🤔 Thinking ({self.current_mode} mode)...")
                result = self.rag_system.ask_question(question)
                
                print(f"\n📚 Answer: {result['answer']}")
                
                # Show sources
                if result['source_documents']:
                    print(f"\n📖 Sources:")
                    for i, doc in enumerate(result['source_documents'][:2], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        print(f"  {i}. {os.path.basename(source)} (Page {page})")
                        print(f"     Preview: {content_preview}")
                        
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def switch_mode(self):
        """Switch between local and Google Gen AI mode"""
        print(f"\n🔄 Current mode: {self.current_mode.upper()}")
        confirm = input("Do you want to switch modes? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            print("Restarting with mode selection...")
            self.current_mode = None
            self.rag_system = None
            self.setup()
            if self.rag_system:
                self.chat_loop()

    def reload_documents(self):
        """Reprocess docs and rebuild both stores."""
        print("🔄 Reloading documents...")
        vs, bm25_idx, bm25_docs = self.processor.process_documents("./documents")
        if vs:
            self.rag_system = RAGSystem(
                vector_store=vs,
                mode=self.current_mode,
                api_key=self.api_key if self.current_mode == "google" else None,
                bm25_index=bm25_idx,
                bm25_chunks=bm25_docs
            )
            print("✅ Documents reloaded successfully!")
    
    def show_help(self):
        """Show available commands"""
        print("\n📋 Available commands:")
        print("  - Ask any question about your documents")
        print("  - 'quit' - Exit the application")
        print("  - 'mode' - Switch between local and Google Gen AI modes")
        print("  - 'reload' - Reload and reprocess all documents")
        print("  - 'help' - Show this help message")
        
        print(f"\n📊 Current Mode: {self.current_mode.upper()}")
        if self.current_mode == "local":
            print("   Using: Ollama with local models")
        else:
            print("   Using: Google Gemini models")

def main():
    assistant = DocumentQAAssistant()
    if assistant.rag_system:
        assistant.chat_loop()

if __name__ == "__main__":
    main()