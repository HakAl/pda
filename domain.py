import os
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from typing import List, Dict, Any, Callable
from rag_system import RAGSystem, RAGSystemError

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
            print() # Newline after streaming is complete
            return result
        except RAGSystemError as e:
            print(f"\n❌ Error during answer generation: {str(e)}")
            return {"answer": "", "source_documents": []}
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {str(e)}")
            return {"answer": "", "source_documents": []}

    def get_llm_display_name(self) -> str:
        """Gets the display name of the current LLM."""
        return self.rag_system.llm_config.get_display_name()
