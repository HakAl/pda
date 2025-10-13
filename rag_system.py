"""
RAG System with dependency injection and clean separation of concerns.
"""

import time
from typing import Any, Dict, List, Optional
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable

from llm_factory import LLMConfig


class RAGSystem:
    """
    RAG system with dependency injection for LLM configuration.

    This class focuses purely on the QA pipeline, delegating:
    - LLM configuration to LLMConfig
    - Retrieval to HybridRetriever (injected)
    - Model loading to external factories
    """

    def __init__(
            self,
            vector_store: Chroma,
            llm_config: LLMConfig,
            bm25_index: Optional[Any] = None,
            bm25_chunks: Optional[List[Document]] = None,
    ):
        """
        Initialize RAG system with dependency injection.

        Args:
            vector_store: ChromaDB vector store
            llm_config: LLM configuration object (handles creation + prompts)
            bm25_index: Optional BM25 index for hybrid retrieval
            bm25_chunks: Optional documents for BM25 index
        """
        self.vector_store = vector_store
        self.llm_config = llm_config
        self.bm25_index = bm25_index
        self.bm25_chunks = bm25_chunks or []

        # Create LLM from config
        self.llm = llm_config.create_llm()
        self.prompt = llm_config.get_prompt_template()

        # Build retriever and chain
        self._retriever = self._build_retriever()
        self._chain: Optional[RunnableSerializable] = None

        print(f"🤖 Loaded LLM: {llm_config.get_display_name()}")

    def _build_retriever(self):
        """Build hybrid retriever with reranking."""
        from hybrid_retriever import create_hybrid_retrieval_pipeline

        return create_hybrid_retrieval_pipeline(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            bm25_chunks=self.bm25_chunks,
            use_reranking=True,
        )

    def _get_chain(self) -> RunnableSerializable:
        """Lazy-load and cache the QA chain."""
        if self._chain is None:
            self._chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self._retriever,
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True,
            )
        return self._chain

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.

        Args:
            question: User's question

        Returns:
            Dictionary containing:
                - answer: str (the LLM's response)
                - source_documents: List[Document] (supporting documents)

        Raises:
            RAGSystemError: If retrieval or generation fails
        """
        t0 = time.perf_counter()

        try:
            result = self._get_chain().invoke({"query": question})
            t1 = time.perf_counter()

            print(f"⏱  QA chain took {t1 - t0:.2f}s")

            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
            }

        except Exception as e:
            # Construct helpful error message based on LLM type
            error_msg = self._format_error_message(e)
            raise RAGSystemError(error_msg) from e

    def _format_error_message(self, error: Exception) -> str:
        """Format error message with helpful context."""
        base_msg = f"Error processing your question: {str(error)}"

        # Add LLM-specific troubleshooting hints
        llm_name = self.llm_config.get_display_name().lower()

        if "ollama" in llm_name:
            base_msg += (
                "\n\n💡 Troubleshooting tips for Ollama:"
                "\n  1. Ensure Ollama is running (check with: ollama list)"
                "\n  2. Verify your model is downloaded (e.g., ollama pull llama3.1:8b-instruct-q4_K_M)"
                "\n  3. Check if Ollama service is accessible"
            )
        elif "google" in llm_name or "gemini" in llm_name:
            base_msg += (
                "\n\n💡 Troubleshooting tips for Google Gemini:"
                "\n  1. Verify your API key is valid"
                "\n  2. Check your internet connection"
                "\n  3. Ensure you have API quota remaining"
            )
        elif "openai" in llm_name or "gpt" in llm_name:
            base_msg += (
                "\n\n💡 Troubleshooting tips for OpenAI:"
                "\n  1. Verify your API key is valid"
                "\n  2. Check your internet connection"
                "\n  3. Ensure you have API credits remaining"
            )

        return base_msg

    def get_llm_info(self) -> str:
        """Get human-readable LLM information."""
        return self.llm_config.get_display_name()


class RAGSystemError(Exception):
    """Custom exception for RAG system errors."""
    pass


# Factory function for backward compatibility
def create_rag_system(
        vector_store: Chroma,
        mode: str = "local",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        bm25_index: Optional[Any] = None,
        bm25_chunks: Optional[List[Document]] = None,
) -> RAGSystem:
    """
    Factory function to create RAGSystem with legacy mode parameter.

    This maintains backward compatibility while using the new architecture.
    New code should use RAGSystem(llm_config=...) directly.

    Args:
        vector_store: ChromaDB vector store
        mode: "local", "google", or "openai"
        api_key: API key for cloud providers
        model_name: Optional model name override
        bm25_index: Optional BM25 index
        bm25_chunks: Optional BM25 documents

    Returns:
        Configured RAGSystem instance
    """
    from llm_factory import LLMFactory

    llm_config = LLMFactory.create_from_mode(
        mode=mode,
        api_key=api_key,
        model_name=model_name,
    )

    return RAGSystem(
        vector_store=vector_store,
        llm_config=llm_config,
        bm25_index=bm25_index,
        bm25_chunks=bm25_chunks,
    )