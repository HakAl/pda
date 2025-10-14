# Personal Document Q&A Assistant

A flexible RAG-based document question-answering system that supports both **local processing** (via Ollama) and **cloud processing** (via Google Gen AI). Choose the option that best fits your privacy, performance, and hardware needs.

## Features

- **Dual-Mode Operation**: Switch between local processing (private, offline) and Google Gen AI (fast, powerful)
- **Multiple File Support**: PDFs and text files
- **Lightweight Design**: Optimized for low-power machines
- **Privacy-First**: Local mode keeps all data on your device
- **Fast Cloud Option**: Google Gen AI mode for superior performance when internet is available

## Architecture
```
pda/
├── __init__.py  
├── app.py # Main application with mode selection
├── document_processor.py # Document loading and processing
├── hybrid_retriever.py # Retriever pipeline
├── llm_factory.py # Factory functions to setup LLMs
├── rag_system.py # RAG implementation
├── requirements.txt # Project dependencies
├── .env # Environment variables (API keys)
├── documents/ # Your PDFs and text files go here
└── tests/
    ├── __init__.py
    └── test_llm_factory.py
```

## Setup

### Create project directory
```bash
mkdir pda
cd pda
```

### Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Requirements (`requirements.txt`)

```text
langchain>=0.1.0
langchain-community>=0.0.20
langchain-chroma>=0.1.0
langchain-core>=0.1.0
langchain-huggingface>=0.1.0
langchain-ollama>=0.1.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.1
chromadb>=0.4.0
tqdm>=4.65.0
python-dotenv>=1.0.0
numpy>=1.21.0
huggingface-hub>=0.16.0
torch>=1.9.0
transformers>=4.21.0
ollama>=0.1.0
```

---

## Usage

1. Add your documents:

   * Create a `documents` folder in your project
   * Add PDFs, Word, CSV, and text files you want to query

2. Run the application:

```bash
python app.py
```

3. Choose your preferred mode when prompted:

   * **Local**: Private, offline processing using Ollama
   * **Google Gen AI**: Cloud-based processing using Gemini models

---

## Configuration

### For Local Mode (Private)

#### Install Ollama:

* Download from [ollama.ai](https://ollama.ai)
* Pull a lightweight model:

```bash
# For low-power machines:
ollama pull phi3:mini

# For better quality (requires more RAM):
ollama pull llama3.1:8b-instruct-q4_K_M
```

---

### For Google Generative AI Mode (Cloud)

#### Get Google API Key:

1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Create a new API key
3. Add it to your `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

#### Google Gen AI Benefits:

* Optimized models: Gemini models are efficient and powerful
* Cost-effective: Generous free tier
* Fast inference: Optimized for production use
* Regular updates: Access to latest model improvements

---

### For OpenAI Mode (Cloud)

#### Get OpenAI API Key:

1. Go to [OpenAI Settings](https://platform.openai.com/settings/organization/api-keys)
2. Create a new API key
3. Add it to your `.env` file:

```env
OPENAI_API_KEY=your_google_api_key_here
```

---

## Mode Comparison

| Aspect       | Local Mode             | Cloud AI Modes             |
| ------------ | ---------------------- |----------------------------|
| **Privacy**  | ✅ 100% local           | ❌ Data sent to Google      |
| **Internet** | ✅ Works offline        | ❌ Requires connection      |
| **Speed**    | ⚠️ Depends on hardware | ✅ Very fast                |
| **Cost**     | ✅ Completely free      | ⚠️ Usage-based (free tier) |
| **Hardware** | ❌ Requires RAM         | ✅ Minimal requirements     |

---

## Model Options

### Google Gen AI Models (`rag_system.py`):
```python
model_options = {
    "fast": "gemini-2.0-flash",      # Fastest, most efficient
    "balanced": "gemini-2.0-flash-lite",    # Better quality, still efficient
    "high_quality": "gemini-2.5-flash" # Best quality
}
```

### Local Models (Recommended for Low-Power Systems):

* `phi3:mini (3.8B)` – Best for low-power systems
* `llama3.1:8b-instruct-q4_K_M (8B)` – Better quality (needs 8GB+ RAM)

---

## Next Steps

You now have a **hybrid document Q&A system**! You can extend it by:

* Implementing a web interface with Streamlit
* Adding document management (add/remove documents)
* Implementing conversation history
* JSON grammar constraint (Ollama feature)
* Strip images, headers, footers before chunking
* Add support for Excel

---

## Need Help?

* **Local mode issues**: Check [Ollama documentation](https://ollama.ai) and verify models are downloaded
* **Google Gen AI issues**: Check API key and internet connection
* **Performance issues**: Try smaller models or reduce chunk sizes

### Public Domain Repositories for RAG Test Data

- **[Project Gutenberg](https://www.gutenberg.org/)**  
  • 60,000+ public domain eBooks (literature, history)  
  • Format: Plain text (.txt)  

- **[Internet Archive](https://archive.org/)**  
  • Millions of books, texts, and documents  
  • Format: PDF/TXT (filter for "Texts" and public domain)  

- **[USA.gov](https://www.usa.gov/federal-agencies)**  
  • U.S. government publications (reports, laws, policies)  
  • Format: PDF/TXT (public domain)  
  • Examples: NASA, DOE, and federal agency archives  
------