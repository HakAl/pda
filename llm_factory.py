from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate


@dataclass(kw_only=True)
class LLMConfig(ABC):
    """Base configuration for LLM providers."""

    temperature: float = field(default=0.1, init=False)
    max_tokens: int = field(default=1000, init=False)

    @abstractmethod
    def create_llm(self) -> BaseLanguageModel:
        """Create and return configured LLM instance."""
        pass

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        """Return appropriate prompt template for this LLM."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Return human-readable name for logging/UI."""
        pass


@dataclass(kw_only=True)
class GoogleGenAIConfig(LLMConfig):
    """Configuration for Google Generative AI models."""

    api_key: str
    model_name: str = "gemini-2.0-flash-lite"
    temperature: float = 0.1
    max_tokens: int = 1000

    def create_llm(self) -> BaseLanguageModel:
        """Create Google Gemini LLM instance."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        except ImportError as e:
            raise ImportError(
                "Google Generative AI not available. "
                "Install with: pip install langchain-google-genai"
            ) from e

    def get_prompt_template(self) -> PromptTemplate:
        """Return detailed prompt template optimized for Gemini."""
        template = """You are a helpful document assistant. Use only the provided context to answer accurately.

Context: {context}

Question: {question}

Instructions:
- Answer solely from the context provided above
- If the context does not contain the information, state: 'I cannot find this information in the provided documents.'
- Keep answers concise and cite sources when relevant
- Be direct and factual

Answer: """

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def get_display_name(self) -> str:
        return f"Google {self.model_name}"


@dataclass(kw_only=True)
class OllamaConfig(LLMConfig):
    """Configuration for local Ollama models."""

    model_name: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.1
    max_tokens: int = 600
    num_thread: int = 6
    num_gpu: int = 1
    top_k: int = 20
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    def create_llm(self) -> BaseLanguageModel:
        """Create Ollama LLM instance."""
        try:
            from langchain_ollama import OllamaLLM

            return OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                num_thread=self.num_thread,
                num_gpu=self.num_gpu,
                top_k=self.top_k,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
            )
        except ImportError as e:
            raise ImportError(
                "Ollama not available. "
                "Install with: pip install langchain-ollama"
            ) from e

    def get_prompt_template(self) -> PromptTemplate:
        """Return concise prompt template optimized for local models."""
        template = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "Answer ONLY using the context below.\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def get_display_name(self) -> str:
        return f"Ollama {self.model_name}"


@dataclass(kw_only=True)
class OpenAIConfig(LLMConfig):
    """Configuration for OpenAI models (example of extensibility)."""

    api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1000

    def create_llm(self) -> BaseLanguageModel:
        """Create OpenAI LLM instance."""
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except ImportError as e:
            raise ImportError(
                "OpenAI not available. "
                "Install with: pip install langchain-openai"
            ) from e

    def get_prompt_template(self) -> PromptTemplate:
        """Return prompt template optimized for OpenAI."""
        template = """You are a document Q&A assistant. Answer based solely on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Only use information from the context above
- If you cannot answer from the context, say so clearly
- Be concise and accurate

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def get_display_name(self) -> str:
        return f"OpenAI {self.model_name}"


class LLMFactory:
    """
    Factory for creating LLM configurations based on user preferences.

    This separates configuration logic from the RAG system itself.
    """

    @staticmethod
    def create_from_mode(
            mode: str,
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
    ) -> LLMConfig:
        """
        Create LLM config from legacy mode string (for backward compatibility).

        Args:
            mode: "local", "google", or "openai"
            api_key: API key for cloud providers
            model_name: Optional model override

        Returns:
            Configured LLMConfig instance

        Raises:
            ValueError: If mode is invalid or required parameters missing
        """
        mode = mode.lower().strip()

        if mode == "local":
            return OllamaConfig(
                model_name=model_name or "llama3.1:8b-instruct-q4_K_M"
            )

        elif mode == "google":
            if not api_key:
                raise ValueError("Google mode requires api_key parameter")
            return GoogleGenAIConfig(
                api_key=api_key,
                model_name=model_name or "gemini-2.0-flash-lite"
            )

        elif mode == "openai":
            if not api_key:
                raise ValueError("OpenAI mode requires api_key parameter")
            return OpenAIConfig(
                api_key=api_key,
                model_name=model_name or "gpt-4o-mini"
            )

        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                f"Valid options: 'local', 'google', 'openai'"
            )

    @staticmethod
    def create_google(
            api_key: str,
            model_name: str = "gemini-2.0-flash-lite",
            **kwargs
    ) -> GoogleGenAIConfig:
        """Convenience method for creating Google config."""
        return GoogleGenAIConfig(
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )

    @staticmethod
    def create_ollama(
            model_name: str = "llama3.1:8b-instruct-q4_K_M",
            **kwargs
    ) -> OllamaConfig:
        """Convenience method for creating Ollama config."""
        return OllamaConfig(
            model_name=model_name,
            **kwargs
        )

    @staticmethod
    def create_openai(
            api_key: str,
            model_name: str = "gpt-4o-mini",
            **kwargs
    ) -> OpenAIConfig:
        """Convenience method for creating OpenAI config."""
        return OpenAIConfig(
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )

def check_google_api_key(api_key: str) -> bool:
    """
    Validate Google API key by attempting to configure.

    Args:
        api_key: Google API key to validate

    Returns:
        True if key appears valid, False otherwise
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Could add a test API call here if needed
        return True
    except Exception:
        return False

def check_ollama_available() -> bool:
    """
    Check if Ollama is installed and running.

    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False

def get_available_ollama_models() -> List[str]:
    """
    Fetches the names of all locally available Ollama models.
    This version is highly defensive to handle variations in the ollama library's output.
    """
    try:
        import ollama
        response: Any = ollama.list()
        models_list: List[Any] = []

        if isinstance(response, dict) and 'models' in response:
            models_list = response['models']
        elif hasattr(response, 'models'):
            models_list = response.models
        else:
            print(
                f"\n[DEBUG] Could not find a 'models' list in the response from ollama.list(). Response: {response}\n")
            return []

        if not isinstance(models_list, list):
            print(f"\n[DEBUG] The 'models' field found was not a list. Found: {models_list}\n")
            return []

        model_names: List[str] = []
        for model_item in models_list:
            name = None
            if isinstance(model_item, dict):
                name = model_item.get('name') or model_item.get('model')
            elif hasattr(model_item, 'name'):
                name = model_item.name
            elif hasattr(model_item, 'model'):
                name = model_item.model

            if name:
                model_names.append(name)

        if not model_names and models_list:
            print("\n[DEBUG] Found a models list, but could not extract any model names from it.\n")

        return model_names

    except Exception as e:
        print(f"\n[DEBUG] An error occurred while executing ollama.list(): {e}\n")
        if "Connection refused" in str(e):
            print("[INFO] Could not connect to Ollama. Please ensure the Ollama application is running.")
        return []