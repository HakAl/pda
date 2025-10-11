# Personal Document Q&A Assistant

A flexible RAG-based document question-answering system that supports both **local processing** (via Ollama) and **cloud processing** (via Google ADK). Choose the option that best fits your privacy, performance, and hardware needs.

## Features

- **Dual-Mode Operation**: Switch between local processing (private, offline) and Google ADK (fast, powerful)
- **Multiple File Support**: PDFs and text files
- **Lightweight Design**: Optimized for low-power machines
- **Privacy-First**: Local mode keeps all data on your device
- **Fast Cloud Option**: Google ADK mode for superior performance when internet is available

## Architecture
```
pda/
├── app.py # Main application with mode selection
├── document_processor.py # Document loading and processing
├── rag_system.py # RAG implementation with dual-mode support
├── requirements.txt # Project dependencies
├── .env # Environment variables (API keys)
└── documents/ # Your PDFs and text files go here
```


## Setup

### Create project directory
```bash
mkdir pda
cd pda
```

### Create virtual environment (optional but recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Requirements (`requirements.txt`)

```text
langchain>=0.1.0  
langchain-community>=0.0.10  
langchain-google-genai>=0.0.4  
chromadb>=0.4.0  
pypdf>=3.0.0  
python-dotenv>=1.0.0  
sentence-transformers>=2.2.0  
ollama>=0.1.0  
google-generativeai>=0.3.0
```

---

## Configuration

### For Google ADK Mode (Cloud)

#### Get Google API Key:

1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Create a new API key
3. Add it to your `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

#### Google ADK Benefits:

* Optimized models: Gemini models are efficient and powerful
* Cost-effective: Generous free tier
* Fast inference: Optimized for production use
* Regular updates: Access to latest model improvements

---

### For Local Mode (Private)

#### Install Ollama:

* Download from [ollama.ai](https://ollama.ai)
* Pull a lightweight model:

```bash
# For low-power machines:
ollama pull phi3:mini

# For better quality (requires more RAM):
ollama pull llama3.1:8b
```

---

## Usage

1. Add your documents:

   * Create a `documents` folder in your project
   * Add PDFs and text files you want to query

2. Run the application:

```bash
python app.py
```

3. Choose your preferred mode when prompted:

   * **Local**: Private, offline processing using Ollama
   * **Google ADK**: Cloud-based processing using Gemini models

---

## Example Usage

Once running, you can ask questions like:

* *"What are the main points from the document about project planning?"*
* *"Summarize the key findings from the research paper"*
* *"What does the document say about machine learning best practices?"*

---

## Mode Comparison

| Aspect       | Local Mode             | Google ADK Mode            |
| ------------ | ---------------------- | -------------------------- |
| **Privacy**  | ✅ 100% local           | ❌ Data sent to Google      |
| **Internet** | ✅ Works offline        | ❌ Requires connection      |
| **Speed**    | ⚠️ Depends on hardware | ✅ Very fast                |
| **Cost**     | ✅ Completely free      | ⚠️ Usage-based (free tier) |
| **Hardware** | ❌ Requires RAM         | ✅ Minimal requirements     |

---

## Model Options

### Google ADK Models (`rag_system.py`):

```python
model_options = {
    "fast": "gemini-1.5-flash",      # Fastest, most efficient
    "balanced": "gemini-1.5-pro",    # Better quality, still efficient
    "high_quality": "gemini-1.5-pro" # Best quality
}
```

### Local Models (Recommended for Low-Power Systems):

* `phi3:mini (3.8B)` – Best for low-power systems
* `llama3.2:3b (3B)` – New, optimized
* `qwen2.5:0.5b (0.5B)` – Minimal requirements
* `llama3.1:8b (8B)` – Better quality (needs 8GB+ RAM)

---

## Performance Tips for Low-Power Machines

* Use smaller models: `phi3:mini` or `llama3.2:3b`
* Limit chunk size: 512–1000 characters
* Reduce retrieved documents: Top 2–3 matches only
* Process in batches for large document collections
* Choose **Local Mode**: For privacy + enough RAM
* Choose **Google ADK**: For speed + internet access

---

## Switching Modes

You can easily switch between modes by:

* Restarting the application
* Choosing your preferred mode at startup
* Using different `.env` configurations for different use cases

---

## Next Steps

You now have a **hybrid document Q&A system**! You can extend it by:

* Adding support for more file types (Word, Excel)
* Implementing a web interface with Streamlit
* Adding document management (add/remove documents)
* Implementing conversation history
* Adding batch processing for large document collections

---

## Need Help?

* **Local mode issues**: Check [Ollama documentation](https://ollama.ai) and verify models are downloaded
* **Google ADK issues**: Check API key and internet connection
* **Performance issues**: Try smaller models or reduce chunk sizes