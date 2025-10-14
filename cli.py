import os
import sys
from typing import Optional

# Dependency Injections
from domain import QAService
from document_processor import DocumentProcessor
from llm_factory import LLMFactory, LLMConfig, check_ollama_available, GoogleGenAIConfig, OllamaConfig
from config import app_config
from rag_system import RAGSystem  # Needed for re-instantiation


class CLI:
    """
    Handles all command-line interface interactions.
    """

    def __init__(
            self,
            qa_service: QAService,
            processor: DocumentProcessor,
            initial_vector_store,
            initial_bm25_index,
            initial_bm25_chunks
    ):
        self.qa_service = qa_service
        self.processor = processor

        # Keep state of the document stores for rebuilding services
        self.vector_store = initial_vector_store
        self.bm25_index = initial_bm25_index
        self.bm25_chunks = initial_bm25_chunks

        self.llm_config = qa_service.rag_system.llm_config

    def run(self):
        """
        Main loop for the command-line chat interface.
        """
        print(f"\n💬 Document Q&A Assistant ({self.qa_service.get_llm_display_name()})")
        print("Type 'quit' to exit, 'help' for commands, 'switch' to change LLM\n")

        while True:
            try:
                question = input("\n📝 Your question: ").strip()

                if not question:
                    continue

                command = question.lower()
                if command == 'quit':
                    break
                elif command == 'help':
                    self._show_help()
                    continue
                elif command == 'switch':
                    self._handle_switch_llm()
                    continue
                elif command == 'reload':
                    self._handle_reload()
                    continue

                def stream_handler(token: str):
                    print(token, end="", flush=True)

                result = self.qa_service.answer_question_stream(question, stream_handler)

                if result and result.get('source_documents'):
                    self._display_sources(result['source_documents'])

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred in the chat loop: {e}")

        print("\n\nGoodbye! 👋")

    def _display_sources(self, source_documents):
        print(f"\n📖 Sources:")
        for i, doc in enumerate(source_documents[:2], 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content_preview = doc.page_content[:100] + "..."
            print(f"  {i}. {os.path.basename(source)} (Page {page})")
            print(f"     Preview: {content_preview}")

    def _handle_switch_llm(self):
        print(f"\n🔄 Current LLM: {self.qa_service.get_llm_display_name()}")
        confirm = input("Do you want to switch to a different LLM? (yes/no): ").strip().lower()

        if confirm in ['yes', 'y']:
            new_config = self._choose_llm_config()
            if new_config:
                new_rag_system = RAGSystem(
                    vector_store=self.vector_store,
                    llm_config=new_config,
                    bm25_index=self.bm25_index,
                    bm25_chunks=self.bm25_chunks
                )
                self.qa_service = QAService(new_rag_system)
                self.llm_config = new_config
                print(f"\n✅ Switched to {new_config.get_display_name()}.")
                print("Type 'help' for commands.")

    def _handle_reload(self):
        print("\n🔄 Reloading documents...")
        vs, bm25_idx, bm25_docs = self.processor.process_documents("./documents")
        if vs:
            self.vector_store = vs
            self.bm25_index = bm25_idx
            self.bm25_chunks = bm25_docs

            # Rebuild RAG system with new docs and existing LLM
            new_rag_system = RAGSystem(
                vector_store=self.vector_store,
                llm_config=self.llm_config,
                bm25_index=self.bm25_index,
                bm25_chunks=self.bm25_chunks
            )
            self.qa_service = QAService(new_rag_system)
            print("✅ Documents reloaded successfully!")
        else:
            print("❌ Failed to reload documents.")

    def _show_help(self):
        print("\n📋 Available commands:")
        print("  - Ask any question about your documents")
        print("  - 'quit'   - Exit the application")
        print("  - 'switch' - Switch to a different LLM")
        print("  - 'reload' - Reload and reprocess all documents")
        print("  - 'help'   - Show this help message")
        print(f"\n📊 Current LLM: {self.qa_service.get_llm_display_name()}")

    @staticmethod
    def _choose_llm_config() -> Optional[LLMConfig]:
        print("\n🔧 Choose Your LLM:")
        # ... (print options for Ollama, Google, OpenAI)
        print("1. Local (Ollama)\n2. Google Gemini\n3. OpenAI GPT")

        while True:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice == "1":
                return CLI._configure_ollama()
            elif choice == "2":
                return CLI._configure_google()
            elif choice == "3":
                return CLI._configure_openai()
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

    @staticmethod
    def _configure_ollama() -> Optional[OllamaConfig]:
        if not check_ollama_available():
            print("Ollama not found...")
            return None
        model_name = "llama3.1:8b-instruct-q4_K_M"  # TODO
        print(f"✅ Using Ollama model: {model_name}")
        return LLMFactory.create_ollama(model_name=model_name)

    @staticmethod
    def _configure_google() -> Optional[GoogleGenAIConfig]:
        api_key = app_config.config.google_api_key or input("Enter Google API Key: ").strip()
        if not api_key: return None
        model_name = "gemini-1.5-flash-latest"  # Simplified
        print(f"✅ Using Google model: {model_name}")
        return LLMFactory.create_google(api_key=api_key, model_name=model_name)

    @staticmethod
    def _configure_openai() -> Optional['OpenAIConfig']:
        api_key = os.getenv("OPENAI_API_KEY") or input("Enter OpenAI API Key: ").strip()
        if not api_key: return None
        model_name = "gpt-4o-mini"  # Simplified
        print(f"✅ Using OpenAI model: {model_name}")
        return LLMFactory.create_openai(api_key=api_key, model_name=model_name)