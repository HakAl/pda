"""
Performance testing suite for Q&A retrieval pipeline.

This script profiles the core components of the RAG system, allowing you
to measure the impact of different configurations like reranking and caching.

Setup:
    Ensure you have an existing vector store. If not, run your main app once
    to process the documents and create the database.

Usage:
    # Run all tests with default settings
    python tests/test_retrieval_performance.py

    # Run only the reranking comparison test
    python tests/test_retrieval_performance.py --test reranking

    # Run with more iterations for higher accuracy
    python tests/test_retrieval_performance.py --iterations 20

    # Provide a custom set of queries for testing
    python tests/test_retrieval_performance.py --queries custom_queries.txt
"""

import sys
from pathlib import Path

# Add project root to path so we can import app modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import argparse

# --- Application Component Imports ---
from document_store import DocumentStore
from document_processor import DocumentProcessor
from llm_factory import LLMFactory
from rag_system import RAGSystem
from hybrid_retriever import create_hybrid_retrieval_pipeline


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    query: str
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    times: List[float] = field(default_factory=list)
    num_docs_retrieved: int = 0
    cache_hit: bool = False

    def __str__(self) -> str:
        # Display time in milliseconds for readability
        return (
            f"Query: '{self.query[:50]}...'\n"
            f"  Mean: {self.mean_time*1000:.1f}ms | "
            f"Median: {self.median_time*1000:.1f}ms | "
            f"Min: {self.min_time*1000:.1f}ms | "
            f"Max: {self.max_time*1000:.1f}ms\n"
            f"  Std Dev: {self.std_dev*1000:.1f}ms | "
            f"Docs: {self.num_docs_retrieved} | "
            f"Cache: {'HIT' if self.cache_hit else 'MISS'}"
        )


class RetrievalProfiler:
    """Profile retrieval performance and end-to-end RAG pipeline."""

    def __init__(self, document_store: DocumentStore, rag_system: RAGSystem = None):
        self.document_store = document_store
        self.rag_system = rag_system

    def profile_retriever(
        self,
        queries: List[str],
        iterations: int = 10,
        warmup: int = 2
    ) -> List[PerformanceMetrics]:
        """
        Profile retriever-only performance across multiple queries.
        This isolates the performance of document retrieval from the LLM.
        """
        retriever = self.document_store.get_retriever()
        results = []

        print(f"\n🔬 Profiling retriever with {iterations} iterations per query...")
        print(f"   Warmup: {warmup} iterations\n")

        for i, query in enumerate(queries):
            print(f"({i+1}/{len(queries)}) Testing: '{query[:60]}...'")
            times = []
            num_docs = 0

            # Warmup runs are not timed but ensure caches (e.g., model loading) are hot
            for _ in range(warmup):
                retriever.invoke(query)

            # Actual timed measurements
            for _ in range(iterations):
                start = time.perf_counter()
                docs = retriever.invoke(query)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                # This assumes the number of docs is consistent
                num_docs = len(docs)

            metrics = PerformanceMetrics(
                query=query,
                mean_time=statistics.mean(times),
                median_time=statistics.median(times),
                min_time=min(times),
                max_time=max(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0,
                times=times,
                num_docs_retrieved=num_docs
            )
            results.append(metrics)
            print(f"  ✓ Mean: {metrics.mean_time*1000:.1f}ms | Docs: {num_docs}\n")

        return results

    def profile_rag_with_cache(
        self,
        queries: List[str],
        iterations: int = 10
    ) -> Tuple[List[PerformanceMetrics], List[PerformanceMetrics]]:
        """
        Profiles the full RAG system to measure cache effectiveness.
        It separates the first (cold) run from subsequent (warm) runs.

        Returns:
            A tuple of (cold_metrics_list, warm_metrics_list)
        """
        if not self.rag_system:
            raise ValueError("RAGSystem must be provided for cache profiling.")

        print(f"\n🔬 Profiling RAG pipeline with query cache...")
        cold_results, warm_results = [], []

        for i, query in enumerate(queries):
            print(f"({i+1}/{len(queries)}) Testing: '{query[:60]}...'")
            self.rag_system.clear_cache()

            # --- Cold query (cache miss) ---
            start = time.perf_counter()
            # Corrected to use the actual method name 'ask_question'
            result = self.rag_system.ask_question(query)
            cold_time = time.perf_counter() - start

            cold_metrics = PerformanceMetrics(
                query=query, mean_time=cold_time, median_time=cold_time,
                min_time=cold_time, max_time=cold_time, std_dev=0,
                times=[cold_time], cache_hit=False,
                num_docs_retrieved=len(result.get("source_documents", []))
            )
            cold_results.append(cold_metrics)

            # --- Warm queries (should be cache hits) ---
            warm_times = []
            for _ in range(iterations): # Run multiple times to get stable stats
                start = time.perf_counter()
                self.rag_system.ask_question(query)
                warm_time = time.perf_counter() - start
                warm_times.append(warm_time)

            warm_metrics = PerformanceMetrics(
                query=query,
                mean_time=statistics.mean(warm_times),
                median_time=statistics.median(warm_times),
                min_time=min(warm_times),
                max_time=max(warm_times),
                std_dev=statistics.stdev(warm_times) if len(warm_times) > 1 else 0,
                times=warm_times, cache_hit=True,
                num_docs_retrieved=len(result.get("source_documents", []))
            )
            warm_results.append(warm_metrics)

            speedup = cold_time / warm_metrics.mean_time if warm_metrics.mean_time > 0 else float('inf')
            print(f"  Cold: {cold_time*1000:.1f}ms | Warm (avg): {warm_metrics.mean_time*1000:.1f}ms")
            print(f"  Cache Speedup: {speedup:.1f}x\n")

        return cold_results, warm_results


class ConfigurationComparison:
    """Compare performance across different retrieval configurations."""

    @staticmethod
    def compare_reranking(
        document_store: DocumentStore,
        queries: List[str],
        iterations: int,
        warmup: int
    ) -> Dict[str, List[PerformanceMetrics]]:
        """Compare performance with and without the reranking step."""
        print("\n" + "="*70)
        print("📊 COMPARING: Reranking ON vs OFF")
        print("="*70)
        results = {}

        # --- Test with reranking (Default) ---
        print("\n🔄 Testing WITH reranking...")
        # NOTE: This is a test-only "hack" to reconfigure the retriever.
        # It rebuilds the retriever with reranking enabled.
        document_store._retriever = create_hybrid_retrieval_pipeline(
            vector_store=document_store.vector_store,
            bm25_index=document_store.bm25_index,
            bm25_chunks=document_store.bm25_chunks,
            use_reranking=True
        )
        profiler = RetrievalProfiler(document_store)
        results['with_reranking'] = profiler.profile_retriever(queries, iterations, warmup)

        # --- Test without reranking ---
        print("\n🔄 Testing WITHOUT reranking...")
        # Rebuild the retriever with reranking disabled
        document_store._retriever = create_hybrid_retrieval_pipeline(
            vector_store=document_store.vector_store,
            bm25_index=document_store.bm25_index,
            bm25_chunks=document_store.bm25_chunks,
            use_reranking=False
        )
        profiler = RetrievalProfiler(document_store)
        results['without_reranking'] = profiler.profile_retriever(queries, iterations, warmup)

        return results


def print_summary(results: Dict[str, List[PerformanceMetrics]], title: str):
    """Print a clear comparison summary of the results."""
    print("\n" + "="*70)
    print(f"📈 PERFORMANCE SUMMARY: {title}")
    print("="*70)

    averages = {}
    for config_name, metrics_list in results.items():
        if not metrics_list: continue
        avg_time = statistics.mean([m.mean_time for m in metrics_list])
        avg_docs = statistics.mean([m.num_docs_retrieved for m in metrics_list])
        averages[config_name] = avg_time
        print(f"\n{config_name.upper().replace('_', ' ')}:")
        print(f"  Average Time / Query: {avg_time*1000:.1f}ms")
        print(f"  Average Docs Retrieved: {avg_docs:.1f}")

    # Calculate and print the speedup between the two configurations
    if len(results) == 2:
        configs = list(results.keys())
        baseline_name, comparison_name = configs[0], configs[1]
        baseline_avg = averages[baseline_name]
        comparison_avg = averages[comparison_name]

        if baseline_avg > 0 and comparison_avg > 0:
            if comparison_avg < baseline_avg:
                speedup = baseline_avg / comparison_avg
                print(f"\n⚡ Analysis: '{comparison_name}' is {speedup:.2f}x faster than '{baseline_name}'.")
            else:
                slowdown = comparison_avg / baseline_avg
                print(f"\n⚡ Analysis: '{comparison_name}' is {slowdown:.2f}x slower than '{baseline_name}'.")
    print("\n" + "="*70)


DEFAULT_QUERIES = [
    "What is the main topic of the document?",
    "Who are the key people mentioned?",
    "What are the important dates?",
    "Explain the methodology used",
    "What are the conclusions?",
    "How does this compare to previous work?",
    "What are the limitations discussed?",
    "What future work is suggested?",
]


def load_queries_from_file(filepath: str) -> List[str]:
    """Load queries from a text file, one per line."""
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: Query file not found at '{filepath}'")
        sys.exit(1)


def initialize_components() -> Tuple[DocumentStore, RAGSystem]:
    """Loads all necessary components for the application."""
    print("🔧 Initializing components...")
    processor = DocumentProcessor()
    vector_store, bm25_index, bm25_chunks = processor.load_existing_vectorstore()

    if vector_store is None:
        print("❌ Error: No existing vector store found.")
        print("   Please run your main application first to process documents.")
        sys.exit(1)

    print("✅ Vector store loaded.")

    document_store = DocumentStore(
        vector_store=vector_store,
        bm25_index=bm25_index,
        bm25_chunks=bm25_chunks
    )

    # Use a local LLM for performance testing to avoid network latency.
    # This ensures we are testing the RAG components, not the API provider.
    llm_config = LLMFactory.create_from_mode(mode="local")

    rag_system = RAGSystem(
        document_store=document_store,
        llm_config=llm_config,
        enable_cache=True  # Ensure cache is on for testing
    )
    print("✅ DocumentStore and RAGSystem are ready.")
    return document_store, rag_system


def main():
    parser = argparse.ArgumentParser(description='Profile Q&A retrieval performance')
    parser.add_argument('--queries', type=str, help='Path to a file with test queries (one per line).')
    parser.add_argument('--iterations', type=int, default=10, help='Number of timed iterations per query.')
    parser.add_argument('--warmup', type=int, default=2, help='Number of untimed warmup iterations.')
    parser.add_argument('--test', choices=['reranking', 'cache', 'all'], default='all',
                       help='Which performance test to run.')
    args = parser.parse_args()

    if args.queries:
        queries = load_queries_from_file(args.queries)
        print(f"✓ Loaded {len(queries)} queries from '{args.queries}'")
    else:
        queries = DEFAULT_QUERIES[:5]
        print(f"✓ Using {len(queries)} default queries. Use --queries for more.")

    document_store, rag_system = initialize_components()

    # --- Run Selected Tests ---
    if args.test in ['reranking', 'all']:
        reranking_results = ConfigurationComparison.compare_reranking(
            document_store, queries, args.iterations, args.warmup
        )
        print_summary(reranking_results, title="Reranking Comparison")

    if args.test in ['cache', 'all']:
        profiler = RetrievalProfiler(document_store, rag_system)
        cold_metrics, warm_metrics = profiler.profile_rag_with_cache(
            queries, iterations=args.iterations
        )
        cache_results = {
            "cache_miss (cold)": cold_metrics,
            "cache_hit (warm)": warm_metrics
        }
        print_summary(cache_results, title="Semantic Cache Performance")


if __name__ == "__main__":
    main()