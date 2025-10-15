import streamlit as st
import sys
from typing import Optional
from pathlib import Path

# Add your project root to path if needed
# sys.path.insert(0, str(Path(__file__).parent))

from rag_system import RAGSystem, RAGSystemError
from llm_factory import LLMFactory, get_available_ollama_models, check_ollama_available
from document_store import DocumentStore

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance with proper contrast
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-box {
        background-color: #e8f4f8;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border-left: 4px solid #4CAF50;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .cache-hit {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    /* Ensure expander content is visible */
    .streamlit-expanderContent {
        background-color: transparent;
    }
    /* Source header styling */
    .source-header {
        color: #dadada;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system(
        vector_store_path: str,
        mode: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        enable_cache: bool = True,
        cache_threshold: float = 0.85
) -> RAGSystem:
    """Initialize and cache the RAG system"""
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Load vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )

        # Create LLM config based on mode
        if mode == "local":
            llm_config = LLMFactory.create_ollama(model_name=model_name)
        elif mode == "google":
            # Only pass model_name if it's provided, otherwise use default
            if model_name:
                llm_config = LLMFactory.create_google(
                    api_key=api_key,
                    model_name=model_name
                )
            else:
                llm_config = LLMFactory.create_google(api_key=api_key)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create document store
        document_store = DocumentStore(
            vector_store=vector_store,
            bm25_index=None,  # Optional: can be added if available
            bm25_chunks=None
        )

        # Create RAG system directly
        rag_system = RAGSystem(
            document_store=document_store,
            llm_config=llm_config,
            enable_cache=enable_cache,
            cache_similarity_threshold=cache_threshold
        )

        return rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        st.stop()


def sidebar_config():
    """Render sidebar configuration"""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # Vector store path
        vector_store_path = st.text_input(
            "Vector Store Path",
            value="./chroma_db",
            help="Path to your ChromaDB vector store"
        )

        # LLM Mode
        mode = st.selectbox(
            "LLM Mode",
            options=["local", "google"],
            help="Choose between local (Ollama) or cloud (Google Gemini)"
        )

        # Model selection based on mode
        model_name = None
        if mode == "local":
            # Get available Ollama models
            if check_ollama_available():
                available_models = get_available_ollama_models()

                if available_models:
                    st.info(f"Found {len(available_models)} Ollama model(s)")
                    model_name = st.selectbox(
                        "Select Ollama Model",
                        options=available_models,
                        help="Choose from your locally installed Ollama models"
                    )
                else:
                    st.warning("⚠️ No Ollama models found. Please ensure Ollama is running and models are installed.")
                    model_name = st.text_input(
                        "Model Name",
                        value="",
                        help="Enter model name manually (e.g., llama3.1:8b-instruct-q4_K_M)"
                    )
                    st.info("💡 Install models with: `ollama pull <model-name>`")
            else:
                st.error("❌ Ollama is not available. Please ensure Ollama is installed and running.")
                model_name = st.text_input(
                    "Model Name",
                    value="",
                    help="Enter model name manually (e.g., llama3.1:8b-instruct-q4_K_M)"
                )
        else:
            # Google mode - manual input
            model_name = st.text_input(
                "Model Name (Optional)",
                value="",
                placeholder="gemini-2.0-flash-lite",
                help="Leave empty to use default model (gemini-2.0-flash-lite)"
            )
            # Convert empty string to None for proper default handling
            if not model_name or model_name.strip() == "":
                model_name = None

        # API Key for cloud mode
        api_key = None
        if mode == "google":
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Enter your Google Gemini API key"
            )
            if not api_key:
                st.warning("⚠️ Google API key is required for cloud mode")

        # Cache settings
        st.divider()
        st.subheader("🧠 Cache Settings")
        enable_cache = st.toggle("Enable Semantic Cache", value=True)
        cache_threshold = st.slider(
            "Cache Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
            disabled=not enable_cache
        )

        # Validation
        can_initialize = True
        if mode == "local" and not model_name:
            can_initialize = False
            st.warning("⚠️ Please select or enter an Ollama model")
        if mode == "google" and not api_key:
            can_initialize = False

        # Initialize button
        if st.button("🚀 Initialize System", type="primary", use_container_width=True, disabled=not can_initialize):
            st.session_state.rag_initialized = True
            st.session_state.vector_store_path = vector_store_path
            st.session_state.mode = mode
            st.session_state.api_key = api_key
            st.session_state.model_name = model_name
            st.session_state.enable_cache = enable_cache
            st.session_state.cache_threshold = cache_threshold
            st.rerun()

        # Show system info if initialized
        if st.session_state.get("rag_initialized"):
            st.divider()
            st.subheader("📊 System Info")

            try:
                rag_system = initialize_rag_system(
                    st.session_state.vector_store_path,
                    st.session_state.mode,
                    st.session_state.api_key,
                    st.session_state.model_name,
                    st.session_state.enable_cache,
                    st.session_state.cache_threshold
                )

                st.info(f"**LLM:** {rag_system.get_llm_info()}")

                # Cache stats
                if st.session_state.enable_cache:
                    stats = rag_system.get_cache_stats()
                    st.metric("Cache Size", f"{stats['cache_size']}/{stats['max_size']}")

                    if st.button("🧹 Clear Cache", use_container_width=True):
                        rag_system.clear_cache()
                        st.success("Cache cleared!")
                        st.rerun()
            except:
                pass


def main():
    """Main application logic"""

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False

    # Render sidebar
    sidebar_config()

    # Main content
    st.title("💬 RAG Q&A System")
    st.markdown("Ask questions about your documents and get AI-powered answers with sources.")

    # Check if system is initialized
    if not st.session_state.rag_initialized:
        st.warning("👈 Please configure and initialize the system using the sidebar.")
        st.info("""
        **Quick Start:**
        1. Set your vector store path
        2. Choose LLM mode (local/google)
        3. Configure your model and API key (if needed)
        4. Configure cache settings (optional)
        5. Click 'Initialize System'
        """)
        return

    # Load RAG system
    try:
        rag_system = initialize_rag_system(
            st.session_state.vector_store_path,
            st.session_state.mode,
            st.session_state.api_key,
            st.session_state.model_name,
            st.session_state.enable_cache,
            st.session_state.cache_threshold
        )
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f'<p class="source-header">Source {i}:</p>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="source-box">{source}</div>',
                                    unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            source_placeholder = st.empty()

            full_response = ""
            sources = []

            try:
                # Token callback for streaming
                def token_callback(token: str):
                    nonlocal full_response
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                # Get streaming response
                result = rag_system.ask_question_stream(question, token_callback)

                # Final response without cursor
                response_placeholder.markdown(full_response)

                # Extract and display sources
                if result.get("source_documents"):
                    sources = [
                        doc.page_content[:300] + "..." if len(doc.page_content) > 300
                        else doc.page_content
                        for doc in result["source_documents"]
                    ]

                    with source_placeholder.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f'<p class="source-header">Source {i}:</p>',
                                        unsafe_allow_html=True)
                            st.markdown(f'<div class="source-box">{source}</div>',
                                        unsafe_allow_html=True)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })

            except RAGSystemError as e:
                error_msg = f"❌ **Error:** {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"❌ **Unexpected Error:** {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Clear chat button in sidebar
    with st.sidebar:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()