from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSerializable
from sentence_transformers import CrossEncoder


# ---------- singleton reranker ----------
@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")


class RAGSystem:
    def __init__(
        self,
        vector_store: Chroma,
        mode: str = "local",
        api_key: str | None = None,
        model_name: str | None = None,
        bm25_index: Any | None = None,
        bm25_chunks: list[Document] | None = None,
    ):
        self.vector_store = vector_store
        self.embeddings = vector_store._embedding_function
        self.mode = mode
        self.bm25_index = bm25_index
        self.bm25_chunks = bm25_chunks or []

        self._pre_warm_components()
        self.llm = self._setup_llm(mode, api_key, model_name)
        self.prompt = self._setup_prompt(mode)

        # ---------- important: keep retriever and chain separate ----------
        self._retriever = self._build_retriever()
        self._chain: RunnableSerializable | None = None

        print(f"🤖 Loaded {mode.upper()} model: {self._get_model_info()}")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _pre_warm_components(self) -> None:
        if self.bm25_index:
            try:
                _ = self.bm25_index.get_scores(["test"])
            except Exception:
                pass

    def _setup_llm(self, mode: str, api_key: str | None, model_name: str | None):
        if mode == "google":
            return self._setup_google_llm(api_key, model_name)
        return self._setup_optimized_local_llm(model_name)

    def _setup_google_llm(self, api_key: str | None, model_name: str | None):
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
                max_output_tokens=1000,
            )
        except ImportError as e:
            raise ImportError("pip install langchain-google-genai") from e

    def _setup_optimized_local_llm(self, model_name: str | None):
        try:
            from langchain_ollama import OllamaLLM

            model = model_name or "phi3:mini"
            return OllamaLLM(
                model=model,
                temperature=0.1,
                num_predict=600,
                num_thread=6,
                num_gpu=1,
                top_k=20,
                top_p=0.9,
                repeat_penalty=1.1,
            )
        except ImportError as e:
            raise ImportError("pip install ollama") from e

    def _setup_prompt(self, mode: str) -> PromptTemplate:
        template = (
            "Context: {context}\n\nQuestion: {question}\n\n"
            "Answer only from the context above.  "
            "If the context lacks the answer, say exactly: "
            "'I cannot find this information in the provided documents.'\n"
            "Answer: "
        )
        if mode == "google":  # longer prompt for cloud model
            template = (
                "You are a helpful document assistant.  "
                "Use only the provided context to answer accurately.\n\n"
                "Context: {context}\n\nQuestion: {question}\n\n"
                "Instructions:\n"
                "- Answer solely from the context\n"
                "- If the context does not contain the information, state: "
                "'I cannot find this information in the provided documents.'\n"
                "- Keep answers concise and cite sources when relevant.\n\n"
                "Answer: "
            )
        return PromptTemplate(template=template, input_variables=["context", "question"])

    # ------------------------------------------------------------------
    #  Retriever (built once)
    # ------------------------------------------------------------------
    def _build_retriever(self) -> ContextualCompressionRetriever:
        reranker = _get_reranker()
        bm25_index = self.bm25_index
        bm25_chunks = self.bm25_chunks

        class FastCompressor(BaseDocumentCompressor):
            def compress_documents(
                self,
                docs: List[Document],
                query: str,
                *,
                callbacks: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> List[Document]:
                merged, seen = list(docs), {d.page_content for d in docs}

                # quick BM25 top-up
                if bm25_index and len(merged) < 4:
                    scores = bm25_index.get_scores(query.lower().split())
                    k = min(2, len(scores))
                    if k:
                        top_idx = np.argpartition(scores, -k)[-k:][::-1]
                        for i in top_idx:
                            if len(merged) >= 4:
                                break
                            c = bm25_chunks[i]
                            if c.page_content not in seen:
                                merged.append(c)
                                seen.add(c.page_content)

                # rerank
                if len(merged) > 1:
                    pairs = [(query, d.page_content) for d in merged]
                    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                    best = np.argpartition(scores, -2)[-2:][::-1]
                    return [merged[i] for i in best]
                return merged

        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.5},
        )
        return ContextualCompressionRetriever(
            base_compressor=FastCompressor(),
            base_retriever=base_retriever,
            callbacks=[],  # disable debug overhead
        )

    # ------------------------------------------------------------------
    #  Chain (built once, cached forever)
    # ------------------------------------------------------------------
    def _get_chain(self) -> RunnableSerializable:
        if self._chain is None:
            self._chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self._retriever,  # ← BaseRetriever, not RetrievalQA
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True,
            )
        return self._chain

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def _get_model_info(self) -> str:
        if self.mode == "google":
            return getattr(self.llm, "model", "gemini-2.0-flash-lite")
        return getattr(self.llm, "model", "phi3:mini")

    @lru_cache(maxsize=128)
    def ask_question(self, question: str) -> dict[str, Any]:
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
            msg = f"Error processing your question: {e}"
            if self.mode == "local":
                msg += "\n\n💡 Tip: Make sure Ollama is running and your model is downloaded."
            return {"answer": msg, "source_documents": []}