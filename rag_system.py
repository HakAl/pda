from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGSystem:
    def __init__(self, vector_store, mode="local", api_key=None, model_name=None):
        self.vector_store = vector_store
        self.mode = mode
        
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
            
            model = model_name or "gemini-1.5-flash"
            
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
            from langchain.llms import Ollama
            
            model = model_name or "phi3:mini"
            
            return Ollama(
                model=model,
                temperature=0.1,
                num_predict=800,  # Limit output length
                num_thread=4,     # Use fewer threads for low-power
            )
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama. Please ensure Ollama is running and model '{model_name}' is pulled.")
    
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
        else:  # local mode - simpler prompt for smaller models
            template = """Answer the question based only on this context:

Context: {context}

Question: {question}

If the context doesn't contain the answer, say "I cannot find this information in the provided documents."

Answer: """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _setup_qa_chain(self):
        """Setup the QA chain with retrieval"""
        # Use fewer documents for local mode to improve performance
        k_value = 2 if self.mode == "local" else 3
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value}
            ),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
    
    def _get_model_info(self):
        """Get model information for display"""
        if self.mode == "google":
            return getattr(self.llm, 'model', 'gemini-1.5-flash')
        else:
            return getattr(self.llm, 'model', 'phi3:mini')
    
    def ask_question(self, question):
        """Ask a question and get an answer"""
        try:
            result = self.qa_chain({"query": question})
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