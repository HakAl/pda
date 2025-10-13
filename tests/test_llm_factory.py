import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_factory import (
    LLMFactory,
    GoogleGenAIConfig,
    OllamaConfig,
    OpenAIConfig,
    check_ollama_available,
)


class TestLLMFactory:
    """Test LLM factory methods."""

    def test_create_from_mode_local(self):
        """Test creating Ollama config from mode string."""
        config = LLMFactory.create_from_mode(mode="local")

        assert isinstance(config, OllamaConfig)
        assert config.model_name == "llama3.1:8b"
        assert config.temperature == 0.1

    def test_create_from_mode_google(self):
        """Test creating Google config from mode string."""
        config = LLMFactory.create_from_mode(
            mode="google",
            api_key="test-key"
        )

        assert isinstance(config, GoogleGenAIConfig)
        assert config.api_key == "test-key"
        assert config.model_name == "gemini-2.0-flash-lite"

    def test_create_from_mode_invalid_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            LLMFactory.create_from_mode(mode="invalid")

    def test_create_from_mode_google_without_key_raises(self):
        """Test that Google mode without API key raises ValueError."""
        with pytest.raises(ValueError, match="requires api_key"):
            LLMFactory.create_from_mode(mode="google")

    def test_create_ollama_with_custom_params(self):
        """Test creating Ollama config with custom parameters."""
        config = LLMFactory.create_ollama(
            model_name="phi3:mini",
            temperature=0.3,
            num_thread=8,
        )

        assert config.model_name == "phi3:mini"
        assert config.temperature == 0.3
        assert config.num_thread == 8

    def test_create_google_with_custom_model(self):
        """Test creating Google config with custom model."""
        config = LLMFactory.create_google(
            api_key="test-key",
            model_name="gemini-1.5-pro"
        )

        assert config.model_name == "gemini-1.5-pro"


class TestGoogleGenAIConfig:
    """Test Google Gemini configuration."""

    def test_get_display_name(self):
        """Test display name formatting."""
        config = GoogleGenAIConfig(
            api_key="test",
            model_name="gemini-2.0-flash-lite"
        )
        assert config.get_display_name() == "Google gemini-2.0-flash-lite"

    def test_get_prompt_template(self):
        """Test prompt template generation."""
        config = GoogleGenAIConfig(api_key="test")
        template = config.get_prompt_template()

        assert "context" in template.input_variables
        assert "question" in template.input_variables
        assert "Context:" in template.template
        assert "Question:" in template.template

    @patch('llm_factory.ChatGoogleGenerativeAI')
    @patch('llm_factory.genai')
    def test_create_llm(self, mock_genai, mock_chat_class):
        """Test LLM creation with mocked dependencies."""
        config = GoogleGenAIConfig(
            api_key="test-key",
            model_name="gemini-2.0-flash-lite",
            temperature=0.2,
            max_tokens=500
        )

        mock_llm = Mock()
        mock_chat_class.return_value = mock_llm

        llm = config.create_llm()

        # Verify genai.configure was called
        mock_genai.configure.assert_called_once_with(api_key="test-key")

        # Verify ChatGoogleGenerativeAI was instantiated correctly
        mock_chat_class.assert_called_once_with(
            model="gemini-2.0-flash-lite",
            google_api_key="test-key",
            temperature=0.2,
            max_output_tokens=500,
        )

        assert llm == mock_llm


class TestOllamaConfig:
    """Test Ollama configuration."""

    def test_get_display_name(self):
        """Test display name formatting."""
        config = OllamaConfig(model_name="llama3.1:8b")
        assert config.get_display_name() == "Ollama llama3.1:8b"

    def test_get_prompt_template(self):
        """Test prompt template is more concise for local models."""
        config = OllamaConfig()
        template = config.get_prompt_template()

        # Local prompt should be shorter
        assert len(template.template) < 500
        assert "Context:" in template.template
        assert "Question:" in template.template

    @patch('llm_factory.OllamaLLM')
    def test_create_llm(self, mock_ollama_class):
        """Test Ollama LLM creation."""
        config = OllamaConfig(
            model_name="phi3:mini",
            temperature=0.2,
            num_thread=4,
        )

        mock_llm = Mock()
        mock_ollama_class.return_value = mock_llm

        llm = config.create_llm()

        mock_ollama_class.assert_called_once()
        call_kwargs = mock_ollama_class.call_args[1]

        assert call_kwargs['model'] == "phi3:mini"
        assert call_kwargs['temperature'] == 0.2
        assert call_kwargs['num_thread'] == 4
        assert llm == mock_llm

    def test_default_parameters(self):
        """Test default parameter values."""
        config = OllamaConfig()

        assert config.model_name == "llama3.1:8b"
        assert config.temperature == 0.1
        assert config.max_tokens == 600
        assert config.num_thread == 6
        assert config.num_gpu == 1


class TestOpenAIConfig:
    """Test OpenAI configuration."""

    def test_get_display_name(self):
        """Test display name formatting."""
        config = OpenAIConfig(api_key="test", model_name="gpt-4o-mini")
        assert config.get_display_name() == "OpenAI gpt-4o-mini"

    @patch('llm_factory.ChatOpenAI')
    def test_create_llm(self, mock_openai_class):
        """Test OpenAI LLM creation."""
        config = OpenAIConfig(
            api_key="test-key",
            model_name="gpt-4o-mini",
            temperature=0.3,
        )

        mock_llm = Mock()
        mock_openai_class.return_value = mock_llm

        llm = config.create_llm()

        mock_openai_class.assert_called_once_with(
            model="gpt-4o-mini",
            api_key="test-key",
            temperature=0.3,
            max_tokens=1000,
        )

        assert llm == mock_llm


class TestUtilityFunctions:
    """Test utility functions."""

    @patch('llm_factory.ollama')
    def test_check_ollama_available_success(self, mock_ollama):
        """Test Ollama availability check when available."""
        mock_ollama.list.return_value = []

        result = check_ollama_available()

        assert result is True
        mock_ollama.list.assert_called_once()

    @patch('llm_factory.ollama')
    def test_check_ollama_available_failure(self, mock_ollama):
        """Test Ollama availability check when not available."""
        mock_ollama.list.side_effect = Exception("Connection failed")

        result = check_ollama_available()

        assert result is False


# Integration-style test example
class TestConfigIntegration:
    """Integration tests showing how configs work together."""

    def test_switching_between_configs(self):
        """Test that different configs produce different display names."""
        ollama = LLMFactory.create_ollama(model_name="phi3:mini")
        google = LLMFactory.create_google(api_key="test", model_name="gemini-2.0-flash-lite")

        assert "Ollama" in ollama.get_display_name()
        assert "Google" in google.get_display_name()
        assert ollama.get_display_name() != google.get_display_name()

    def test_all_configs_have_required_methods(self):
        """Test that all configs implement the required interface."""
        configs = [
            OllamaConfig(),
            GoogleGenAIConfig(api_key="test"),
            OpenAIConfig(api_key="test"),
        ]

        for config in configs:
            # All should have these methods
            assert hasattr(config, 'create_llm')
            assert hasattr(config, 'get_prompt_template')
            assert hasattr(config, 'get_display_name')

            # Test they return correct types
            assert isinstance(config.get_display_name(), str)
            template = config.get_prompt_template()
            assert 'context' in template.input_variables
            assert 'question' in template.input_variables


# Example fixture for testing RAGSystem with dependency injection
@pytest.fixture
def mock_llm_config():
    """Fixture providing a mock LLM config for testing RAGSystem."""
    config = Mock(spec=GoogleGenAIConfig)

    # Mock the methods
    mock_llm = Mock()
    config.create_llm.return_value = mock_llm

    from langchain.prompts import PromptTemplate
    config.get_prompt_template.return_value = PromptTemplate(
        template="Context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

    config.get_display_name.return_value = "Mock LLM"

    return config


# Example test showing how easy it is to test RAGSystem now
class TestRAGSystemWithDI:
    """Example tests for RAGSystem using dependency injection."""

    def test_rag_system_initialization(self, mock_llm_config):
        """Test that RAGSystem initializes with injected config."""
        from rag_system import RAGSystem
        from unittest.mock import Mock

        mock_vector_store = Mock()

        # This is so much easier to test now!
        rag = RAGSystem(
            vector_store=mock_vector_store,
            llm_config=mock_llm_config,
        )

        # Verify LLM was created
        mock_llm_config.create_llm.assert_called_once()
        mock_llm_config.get_prompt_template.assert_called_once()

        assert rag.llm_config == mock_llm_config

    def test_rag_system_uses_injected_llm(self, mock_llm_config):
        """Test that RAGSystem uses the injected LLM."""
        from rag_system import RAGSystem
        from unittest.mock import Mock

        mock_vector_store = Mock()
        mock_llm = Mock()
        mock_llm_config.create_llm.return_value = mock_llm

        rag = RAGSystem(
            vector_store=mock_vector_store,
            llm_config=mock_llm_config,
        )

        # The injected LLM should be used
        assert rag.llm == mock_llm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])