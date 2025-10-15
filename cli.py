import logging
import os
import sys
from typing import Optional
from domain import QAService
from document_processor import DocumentProcessor
from llm_factory import (
    LLMFactory, LLMConfig, check_ollama_available,
    GoogleGenAIConfig, OllamaConfig, get_available_ollama_models
)
from config import app_config
from rag_system import RAGSystem
from document_store import DocumentStore

try:
    import google.generativeai as genai
    from google.api_core import exceptions
except ImportError:
    # todo
    genai = None
    exceptions = None


def setup_logging():
    """
    Configures the application's logging to reduce verbosity from libraries.
    """
    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


class CLI:
    """
    Handles all command-line interface interactions.
    """

    def __init__(
            self,
            qa_service: QAService,
            processor: DocumentProcessor,
            document_store: DocumentStore
    ):
        self.qa_service = qa_service
        self.processor = processor
        self.document_store = document_store
        self.llm_config = qa_service.rag_system.llm_config

    def run(self):
        """
        Main loop for the command-line chat interface.
        """
        setup_logging()

        print(f"\n💬 Document Q&A Assistant ({self.qa_service.get_llm_display_name()})")
        print("Type 'quit' to exit, 'help' for commands, 'switch' to change LLM\n")

        while True:
            try:
                question = input("\n🔍 Your question: ").strip()

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

        print("\n\n👋 Goodbye!")

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
        try:
            confirm = input("Do you want to switch to a different LLM? (yes/no): ").strip().lower()
        except KeyboardInterrupt:
            print("\n❌ Switch cancelled.")
            return

        if confirm in ['yes', 'y']:
            new_config = self._choose_llm_config()
            if new_config:
                try:
                    new_rag_system = RAGSystem(
                        document_store=self.document_store,
                        llm_config=new_config
                    )
                    self.qa_service = QAService(new_rag_system)
                    self.llm_config = new_config

                    print(f"\n✅ Switched to {new_config.get_display_name()}.")
                    print("You can now ask questions using the new model.")
                except Exception as e:
                    print(f"\n❌ Failed to switch LLM: {e}")
            else:
                print("\n❌ Model switch cancelled or failed.")

    def _handle_reload(self):
        print("\n🔄 Reloading documents...")
        vs, bm25_idx, bm25_docs = self.processor.process_documents("./documents")
        if vs:
            self.document_store.vector_store = vs
            self.document_store.bm25_index = bm25_idx
            self.document_store.bm25_chunks = bm25_docs

            try:
                new_rag_system = RAGSystem(
                    document_store=self.document_store,
                    llm_config=self.llm_config
                )
                self.qa_service = QAService(new_rag_system)
                print("✅ Documents reloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to rebuild RAG system: {e}")
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
        print("1. Local (Ollama)\n2. Google Gemini")

        while True:
            try:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice == "1":
                    return CLI._configure_ollama()
                elif choice == "2":
                    return CLI._configure_google()
                else:
                    print("❌ Invalid choice. Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\n❌ Configuration cancelled.")
                return None

    @staticmethod
    def _configure_ollama() -> Optional[OllamaConfig]:
        """
        Lists available Ollama models and prompts the user to select one.
        """
        if not check_ollama_available():
            print("Ollama not found...")
            return None
        available_models = get_available_ollama_models()

        if not available_models:
            print("\n❌ Could not find any Ollama models.")
            print("   Please ensure Ollama is running and you have pulled a model.")
            print("   You can pull a model with: 'ollama run llama3.1'")
            return None

        print("\n🤖 Select an available Ollama model:")
        for i, model_name in enumerate(available_models):
            print(f"  {i + 1}. {model_name}")

        while True:
            try:
                choice_str = input(f"\nEnter your choice (1-{len(available_models)}): ").strip()
                if not choice_str:
                    continue

                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(available_models):
                    selected_model = available_models[choice_idx]
                    print(f"✅ Using Ollama model: {selected_model}")
                    return LLMFactory.create_ollama(model_name=selected_model)
                else:
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(available_models)}.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled.")
                return None

    @staticmethod
    def _configure_google() -> Optional[GoogleGenAIConfig]:
        if not genai:
            print("❌ Google Generative AI library not found.")
            print("Please install it with: pip install google-generativeai")
            return None

        try:
            api_key = app_config.config.google_api_key or input("Enter Google API Key: ").strip()
        except KeyboardInterrupt:
            print("\n❌ Configuration cancelled.")
            return None

        if not api_key:
            return None

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            print(f"❌ Failed to configure Google API: {e}")
            return None

        try:
            print("Fetching available Google AI models...")
            available_chat_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_id = m.name.split('/')[-1]
                    available_chat_models.append((m.display_name, model_id))

        except exceptions.PermissionDenied:
            print("❌ Authentication failed. Please check your API key.")
            return None
        except Exception as e:
            print(f"❌ Could not fetch models from Google AI: {e}")
            return None

        if not available_chat_models:
            print("❌ No compatible chat models found for your account.")
            return None

        model_map = {
            str(i + 1): model_info for i, model_info in enumerate(available_chat_models)
        }

        print("\nPlease select a Google AI model:")
        for key, (display_name, model_id) in model_map.items():
            print(f"  [{key}] {display_name} ({model_id})")

        choice = ""
        while choice not in model_map:
            try:
                choice = input(f"Enter your choice (1-{len(model_map)}): ").strip()
                if choice not in model_map:
                    print("Invalid selection. Please try again.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled.")
                return None

        _, model_name = model_map[choice]

        print(f"✅ Using Google model: {model_name}")
        return LLMFactory.create_google(api_key=api_key, model_name=model_name)