import os
import sys
from document_processor import DocumentProcessor
from config import app_config
from llm_factory import (
    LLMFactory,
    LLMConfig,
    check_ollama_available,
    GoogleGenAIConfig,
    OllamaConfig
)
from rag_system import RAGSystem, RAGSystemError


class DocumentQAAssistant:
    def __init__(self):
        self.processor = None
        self.rag_system = None
        self.llm_config: LLMConfig = None
        self.setup()

    def setup(self):
        print("🚀 Setting up Document Q&A Assistant...")
        print("=" * 50)

        # Choose LLM configuration
        self.llm_config = self.choose_llm_config()

        # Setup document processor
        self.processor = DocumentProcessor()

        # Try to load existing vector store + BM25
        vector_store, bm25_index, bm25_chunks = self.processor.load_existing_vectorstore()

        if vector_store is None:
            print("\nNo existing database found. Processing documents...")
            if not os.path.exists("./documents"):
                os.makedirs("./documents")
                print("📁 Created 'documents' folder. Please add files there and restart.")
                return

            vector_store, bm25_index, bm25_chunks = self.processor.process_documents("./documents")
            if vector_store is None:
                return

        # Initialize RAG system with injected LLM config
        self.rag_system = RAGSystem(
            vector_store=vector_store,
            llm_config=self.llm_config,
            bm25_index=bm25_index,
            bm25_chunks=bm25_chunks
        )

        print(f"\n✅ RAG System ready with {self.llm_config.get_display_name()}!")
        print("You can now ask questions about your documents.")

    def choose_llm_config(self) -> LLMConfig:
        print("\n🔧 Choose Your LLM:")
        print("1. Local Mode (Ollama)")
        print("   - Uses Ollama with local models")
        print("   - 100% private - no data leaves your computer")
        print("   - Requires adequate RAM and Ollama installation")
        print("   - Slower but completely offline")

        print("\n2. Google Gemini (Cloud)")
        print("   - Uses Google's Gemini models")
        print("   - Very fast responses")
        print("   - Requires internet connection and API key")
        print("   - Data is sent to Google's servers")

        print("\n3. OpenAI GPT (Cloud)")
        print("   - Uses OpenAI's GPT models")
        print("   - Fast and high-quality responses")
        print("   - Requires internet connection and API key")
        print("   - Data is sent to OpenAI's servers")

        while True:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()

            if choice == "1":
                return self._configure_ollama()
            elif choice == "2":
                return self._configure_google()
            elif choice == "3":
                return self._configure_openai()
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

    def _configure_ollama(self) -> OllamaConfig:
        if not check_ollama_available():
            print("❌ Ollama not found. Please install Ollama from https://ollama.ai/")
            print("   Then run: ollama pull llama3.1:8b-instruct-q4_K_M")
            sys.exit(1)

        # Let user choose model
        print("\nAvailable Ollama models:")
        print("1. llama3.1:8b-instruct-q4_K_M (recommended, balanced)")
        print("2. phi3:mini (faster, smaller)")

        model_choice = input("\nChoose model (1-2, default=1): ").strip() or "1"

        model_map = {
            "1": "llama3.1:8b-instruct-q4_K_M",
            "2": "phi3:mini",
        }

        if model_choice in model_map:
            model_name = model_map[model_choice]
        elif model_choice == "4":
            model_name = input("Enter model name: ").strip()
        else:
            model_name = "llama3.1:8b-instruct-q4_K_M"

        print(f"✅ Using Ollama model: {model_name}")
        return LLMFactory.create_ollama(model_name=model_name)

    def _configure_google(self) -> GoogleGenAIConfig:
        api_key = app_config.config.google_api_key

        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .env file")
            api_key = input("Enter your Google API key (or press Enter to exit): ").strip()
            if not api_key:
                sys.exit(1)

        # Let user choose model
        print("\nAvailable Google models:")
        print("1. gemini-2.0-flash-lite (recommended, fastest)")
        print("2. gemini-2.0-flash-exp (experimental)")
        print("3. gemini-1.5-pro (highest quality)")

        model_choice = input("\nChoose model (1-3, default=1): ").strip() or "1"

        model_map = {
            "1": "gemini-2.0-flash-lite",
            "2": "gemini-2.0-flash-exp",
            "3": "gemini-1.5-pro",
        }

        if model_choice in model_map:
            model_name = model_map[model_choice]
        elif model_choice == "4":
            model_name = input("Enter model name: ").strip()
        else:
            model_name = "gemini-2.0-flash-lite"

        print(f"✅ Using Google model: {model_name}")
        return LLMFactory.create_google(api_key=api_key, model_name=model_name)

    def _configure_openai(self) -> 'OpenAIConfig':
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print("❌ OPENAI_API_KEY not found in .env file")
            api_key = input("Enter your OpenAI API key (or press Enter to exit): ").strip()
            if not api_key:
                sys.exit(1)

        # Let user choose model
        print("\nAvailable OpenAI models:")
        print("1. gpt-4o-mini (recommended, fast and cheap)")
        print("2. gpt-4o (highest quality)")
        print("3. gpt-4-turbo (balanced)")

        model_choice = input("\nChoose model (1-3, default=1): ").strip() or "1"

        model_map = {
            "1": "gpt-4o-mini",
            "2": "gpt-4o",
            "3": "gpt-4-turbo",
        }

        if model_choice in model_map:
            model_name = model_map[model_choice]
        elif model_choice == "4":
            model_name = input("Enter model name: ").strip()
        else:
            model_name = "gpt-4o-mini"

        print(f"✅ Using OpenAI model: {model_name}")
        return LLMFactory.create_openai(api_key=api_key, model_name=model_name)

    def chat_loop(self):
        """
        Main loop for the command-line chat interface
        """
        if self.rag_system is None:
            print("System not ready. Please check if documents are available.")
            return

        print(f"\n💬 Document Q&A Assistant ({self.llm_config.get_display_name()})")
        print("Type 'quit' to exit, 'help' for commands, 'switch' to change LLM\n")

        while True:
            try:
                question = input("\n📝 Your question: ").strip()

                if question.lower() == 'quit':
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif question.lower() == 'switch':
                    self.switch_llm()
                    continue
                elif question.lower() == 'reload':
                    self.reload_documents()
                    continue
                elif not question:
                    continue

                try:
                    # Define a simple callback handler for printing tokens
                    def stream_handler(token: str):
                        print(token, end="", flush=True)

                    # Print the header, then start the stream
                    print(f"\n📚 Answer: ", end="", flush=True)
                    result = self.rag_system.ask_question_stream(question, stream_handler)

                    # The answer is streamed above. We add a newline here for clean formatting
                    # before printing the sources.
                    print()

                    if result['source_documents']:
                        print(f"\n📖 Sources:")
                        for i, doc in enumerate(result['source_documents'][:2], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            content_preview = doc.page_content[:100] + "..." if len(
                                doc.page_content) > 100 else doc.page_content
                            print(f"  {i}. {os.path.basename(source)} (Page {page})")
                            print(f"     Preview: {content_preview}")

                except RAGSystemError as e:
                    print(f"\n❌ {str(e)}")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")

    def switch_llm(self):
        print(f"\n🔄 Current LLM: {self.llm_config.get_display_name()}")
        confirm = input("Do you want to switch to a different LLM? (yes/no): ").strip().lower()

        if confirm in ['yes', 'y']:
            print("Selecting new LLM configuration...")
            new_config = self.choose_llm_config()

            # Rebuild RAG system with new LLM
            if self.rag_system:
                self.rag_system = RAGSystem(
                    vector_store=self.rag_system.vector_store,
                    llm_config=new_config,
                    bm25_index=self.rag_system.bm25_index,
                    bm25_chunks=self.rag_system.bm25_chunks
                )
                self.llm_config = new_config
                print(f"✅ Switched to {new_config.get_display_name()}")

    def reload_documents(self):
        print("🔄 Reloading documents...")
        vs, bm25_idx, bm25_docs = self.processor.process_documents("./documents")
        if vs:
            self.rag_system = RAGSystem(
                vector_store=vs,
                llm_config=self.llm_config,
                bm25_index=bm25_idx,
                bm25_chunks=bm25_docs
            )
            print("✅ Documents reloaded successfully!")

    def show_help(self):
        print("\n📋 Available commands:")
        print("  - Ask any question about your documents")
        print("  - 'quit' - Exit the application")
        print("  - 'switch' - Switch to a different LLM")
        print("  - 'reload' - Reload and reprocess all documents")
        print("  - 'help' - Show this help message")

        print(f"\n📊 Current LLM: {self.llm_config.get_display_name()}")


def main():
    assistant = DocumentQAAssistant()
    if assistant.rag_system:
        assistant.chat_loop()


if __name__ == "__main__":
    main()