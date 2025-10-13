from functools import lru_cache
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_chroma import Chroma
import time
from typing import Any, List, Optional
import numpy as np


class RAGSystem:
    def __init__(self, vector_store, mode="local", api_key=None, model_name=None,
                 bm25_index=None, bm25_chunks=None):
        self.vector_store = vector_store
        self.embeddings = vector_store._embedding_function
        self.mode = mode
        self.bm25_index = bm25_index
        self.bm25_chunks = bm25_chunks

        # Pre-warm components for faster first response
        self._pre_warm_components()

        # Setup LLM based on mode
        self.llm = self._setup_llm(mode, api_key, model_name)

        # Setup prompt based on mode
        self.prompt = self._setup_prompt(mode)

        self.qa_chain = self._setup_optimized_qa_chain()

        print(f"🤖 Loaded {mode.upper()} model: {self._get_model_info()}")

    def _pre_warm_components(self):
        """Pre-warm critical components to reduce first-call latency"""
        # Pre-warm BM25 with a simple query
        if self.bm25_index:
            try:
                _ = self.bm25_index.get_scores("test".split())
            except:
                pass

    def _setup_llm(self, mode, api_key, model_name):
        """Setup the language model based on mode"""
        if mode == "google":
            return self._setup_google_llm(api_key, model_name)
        else:  # local mode
            return self._setup_optimized_local_llm(model_name)

    def _setup_google_llm(self, api_key, model_name):
        """Setup Google Gemini LLM"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.generativeai as genai

            if api_key:
                genai.configure(api_key=api_key)

            model = model_name or "gemini-2.0-flash-lite"

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=1000
            )
        except ImportError:
            raise ImportError("Please install langchain-google-genai: pip install langchain-google-genai")

    def _setup_optimized_local_llm(self, model_name):
        """Setup optimized local Ollama LLM"""
        try:
            from langchain_ollama import OllamaLLM

            model = model_name or "phi3:mini"

            return OllamaLLM(
                model=model,
                temperature=0.1,
                num_predict=600,  # Reduced from 800 for faster responses
                num_thread=6,  # Increased for better CPU utilization
                num_gpu=1,  # Use GPU if available
                top_k=20,
                top_p=0.9,
                repeat_penalty=1.1,
            )
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        except Exception as e:
            raise Exception(
                f"Failed to connect to Ollama. Please ensure Ollama is running and model '{model_name}' is pulled.")

    def _setup_prompt(self, mode):
        """Setup prompt template based on mode"""
        if mode == "google":
            template = """You are a helpful document assistant. Use the provided context to answer the question accurately.

Context: {context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say "I cannot find this information in the provided documents."
- Keep answers clear and concise
- Cite source documents when relevant

Answer: """
        else:
            template = """You are an accurate assistant.
RULE: answer only from the context below.
REWARD: if the context lacks the answer, exactly say "I cannot find this information in the provided documents."

Context: {context}

Question: {question}

Answer: """

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _setup_optimized_qa_chain(self):
        """Fast retrieval pipeline that works for both small and large datasets"""
        if not hasattr(self, '_reranker'):
            self._reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

        bm25_index = self.bm25_index
        bm25_chunks = self.bm25_chunks
        reranker = self._reranker

        class FastCrossEncoderCompressor(BaseDocumentCompressor):
            def compress_documents(self, docs, query, *, callbacks=None):
                merged = list(docs)
                seen = {d.page_content for d in docs}

                # Fast BM25 augmentation - minimal overhead
                if bm25_index and len(merged) < 3:
                    bm25_scores = bm25_index.get_scores(query.lower().split())
                    num_docs = len(bm25_scores)
                    k_augment = min(2, num_docs)
                    if k_augment > 0:
                        top_indices = np.argpartition(bm25_scores, -k_augment)[-k_augment:][::-1]
                        for i in top_indices:
                            if len(merged) >= 3:
                                break
                            candidate = bm25_chunks[i]
                            if candidate.page_content not in seen:
                                merged.append(candidate)
                                seen.add(candidate.page_content)

                # Fast reranking
                if len(merged) > 1:
                    pairs = [(query, doc.page_content) for doc in merged]
                    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                    top_idx = np.argpartition(scores, -2)[-2:][::-1]
                    result = [merged[i] for i in top_idx]
                else:
                    result = merged

                return result

        # Use fast, consistent parameters that work at any scale
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 8, "lambda_mult": 0.5}
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=FastCrossEncoderCompressor(),
            base_retriever=base_retriever
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def _get_model_info(self):
        """Get model information for display"""
        if self.mode == "google":
            return getattr(self.llm, 'model', 'gemini-1.5-flash')
        else:
            return getattr(self.llm, 'model', 'phi3:mini')

    @lru_cache(maxsize=128)
    def ask_question(self, question):
        """Ask a question and get an answer"""
        t0 = time.perf_counter()
        try:
            result = self.qa_chain.invoke({"query": question})
            t1 = time.perf_counter()
            print(f"⏱  QA chain took {t1 - t0:.2f}s")

            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            if self.mode == "local":
                error_msg += "\n\n💡 Tip: Make sure Ollama is running and your model is downloaded."
            return {
                "answer": error_msg,
                "source_documents": []
            }