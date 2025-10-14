import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    google_api_key: Optional[str] = None
    mode: str = "local"
    model_name: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            mode=os.getenv("MODE", "local"),
            model_name=os.getenv("MODEL_NAME")
        )


@dataclass
class RetrieverConfig:
    vector_k: int = 4
    vector_fetch_k: int = 12
    bm25_top_k: int = 2
    lambda_mult: float = 0.5
    min_docs_before_bm25: int = 4

    @classmethod
    def from_env(cls) -> "RetrieverConfig":
        return cls(
            vector_k=int(os.getenv("VECTOR_K", 4)),
            vector_fetch_k=int(os.getenv("VECTOR_FETCH_K", 12)),
            bm25_top_k=int(os.getenv("BM25_TOP_K", 2)),
            lambda_mult=float(os.getenv("MMR_LAMBDA", 0.5)),
            min_docs_before_bm25=int(os.getenv("MIN_DOCS_BEFORE_BM25", 4))
        )


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100))
        )


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    top_k: int = 2
    batch_size: int = 8

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        return cls(
            model_name=os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-TinyBERT-L-2-v2"),
            top_k=int(os.getenv("RERANKER_TOP_K", 2)),
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", 8))
        )


@dataclass
class AppConfig:
    """Main configuration class that combines all config sections"""
    config: Config
    retriever: RetrieverConfig
    chunking: ChunkingConfig
    reranker: RerankerConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            config=Config.from_env(),
            retriever=RetrieverConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            reranker=RerankerConfig.from_env()
        )


app_config = AppConfig.from_env()