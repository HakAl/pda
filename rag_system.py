from functools import lru_cache
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableSequence
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
import time
from typing import Any, List
from langchain_chroma import Chroma


class RAGSystem:
    def __init__(self, vector_store, mode="local", api_key=None, model_name=None,
                 bm25_index=None, bm25_chunks=None):
        self.vector_store = vector_store
        self.embeddings = vector_store._embedding_function
        self.mode = mode
        self.bm25_index = bm25_index
        self.bm25_chunks = bm25_chunks

        # Setup LLM based on mode
        self.llm = self._setup_llm(mode, api_key, model_name)

        # Setup prompt based on mode
        self.prompt = self._setup_prompt(mode)

        self.qa_chain = self._setup_qa_chain()

        print(f"🤖 Loaded {mode.upper()} model: {self._get_model_info()}")

    def _setup_llm(self, mode, api_key, model_name):
        """Setup the language model based on mode"""
        if mode == "google":
            return self._setup_google_llm(api_key, model_name)
        else:  # local mode
            return self._setup_local_llm(model_name)

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

    def _setup_local_llm(self, model_name):
        """Setup local Ollama LLM"""
        try:
            from langchain_ollama import OllamaLLM

            model = model_name or "phi3:mini"

            return OllamaLLM(
                model=model,
                temperature=0.1,
                num_predict=800,  # Limit output length
                num_thread=4,  # Use fewer threads for low-power
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

    def _setup_qa_chain(self):
        """Fast path: BM25 pre-filter → MMR on 20 candidates → TinyBERT rerank"""
        import time

        # cheap keyword pre-filter: 6609 → 20 docs
        t0 = time.perf_counter()
        scores = self.bm25_index.get_scores("test query".lower().split())
        top_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:20]
        candidates = [self.bm25_chunks[i] for i in top_idx]
        print(f"⏱  BM25 filter {time.perf_counter() - t0:.3f}s  ({len(candidates)} docs)")

        # build tiny temp vector store for MMR only on those 20
        candidate_vs = Chroma.from_documents(candidates, embedding=self.embeddings)
        mmr_ret = candidate_vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 20, "lambda_mult": 0.5}
        )

        rerank_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

        bm25_index = self.bm25_index
        bm25_chunks = self.bm25_chunks

        class CrossEncoderCompressor(BaseDocumentCompressor):
            def compress_documents(self, docs, query, *, callbacks=None):
                # start with the *incoming* docs (already good MMR hits)
                merged = list(docs)  # keep order
                seen = {d.page_content for d in docs}  # fast lookup

                # add top-2 BM25 **only** if new
                bm25_scores = bm25_index.get_scores(query.lower().split())
                for i in sorted(range(len(bm25_scores)), key=bm25_scores.__getitem__, reverse=True)[:2]:
                    c = bm25_chunks[i]
                    if c.page_content not in seen:
                        merged.append(c)
                        seen.add(c.page_content)
                        if len(merged) >= 3:  # never > 4
                            break

                # rerank the merged set (≤ original + 2)
                pairs = [(query, d.page_content) for d in merged]
                scores = rerank_model.predict(pairs)
                top_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
                return [merged[i] for i in top_idx]

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=CrossEncoderCompressor(),
            base_retriever=mmr_ret
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