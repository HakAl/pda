import os
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from domain import QAService
from cli import CLI


def main():
    """
    Composition Root: Initializes and wires up all application components.
    """
    print("🚀 Initializing Document Q&A Assistant...")
    print("=" * 50)

    processor = DocumentProcessor()
    vector_store, bm25_index, bm25_chunks = processor.load_existing_vectorstore()

    if vector_store is None:
        print("\nNo existing database found. Processing documents...")
        docs_path = "./documents"
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
            print(f"📁 Created '{docs_path}' folder. Please add files and restart.")
            return

        vector_store, bm25_index, bm25_chunks = processor.process_documents(docs_path)
        if vector_store is None:
            print("❌ Could not process documents. Exiting.")
            return

    print("\n✅ Document database is ready.")

    llm_config = CLI._choose_llm_config()
    if not llm_config:
        print("❌ LLM not configured. Exiting.")
        return

    try:
        rag_system = RAGSystem(
            vector_store=vector_store,
            llm_config=llm_config,
            bm25_index=bm25_index,
            bm25_chunks=bm25_chunks
        )
        qa_service = QAService(rag_system)
    except Exception as e:
        print(f"❌ Failed to initialize the RAG system: {e}")
        return

    app_cli = CLI(
        qa_service=qa_service,
        processor=processor,
        initial_vector_store=vector_store,
        initial_bm25_index=bm25_index,
        initial_bm25_chunks=bm25_chunks
    )

    app_cli.run()


if __name__ == "__main__":
    main()