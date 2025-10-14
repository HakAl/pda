import os
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from typing import List, Dict, Any, Callable
from rag_system import RAGSystem, RAGSystemError
from error_handler import parse_api_error


class QAService:
    """
    Service layer responsible for handling the question-answering logic.
    """
    def __init__(self, rag_system: RAGSystem):
        if rag_system is None:
            raise ValueError("RAGSystem cannot be None")
        self.rag_system = rag_system

    def answer_question_stream(self, question: str, stream_handler: Callable[[str], None]) -> Dict[str, Any]:
        """
        Answers a question using the configured RAG system and streams the response.
        """
        try:
            print(f"\n📚 Answer: ", end="", flush=True)
            result = self.rag_system.ask_question_stream(question, stream_handler)
            print()
            return result
        except (RAGSystemError, Exception) as e:
            user_friendly_message = parse_api_error(e)
            print(f"\n❌ Error: {user_friendly_message}")
            return {"answer": "", "source_documents": []}

    def get_llm_display_name(self) -> str:
        """Gets the display name of the current LLM."""
        return self.rag_system.llm_config.get_display_name()
