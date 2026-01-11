"""
API dependency injection.
"""
from functools import lru_cache
from typing import AsyncGenerator

from fastapi import Depends

from backend.services.embeddings.service import EmbeddingService, EmbeddingProvider, EmbeddingServiceFactory
from backend.services.retrieval.vector_db import VectorDBFactory, BaseVectorDB
from backend.services.retrieval.engine import RetrievalEngine
from backend.services.generation.service import ResponseGenerator
from backend.services.safety.filter import SafetyFilter
from backend.services.logging.mongo_logger import MongoLogger
from backend.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get embedding service - prefer NVIDIA if API key is available, else use Sentence Transformer."""
    # Check if NVIDIA API key is available
    if settings.nvidia_embedding_api_key:
        try:
            service = EmbeddingService(
                provider=EmbeddingProvider.NVIDIA,
                config={
                    "api_key": settings.nvidia_embedding_api_key,
                    "model_name": settings.nvidia_embedding_model,
                    "dimension": 1024  # NVIDIA embedding dimension
                }
            )
            return service
        except Exception as e:
            logger.warning(f"Failed to create NVIDIA embedding service, falling back to Sentence Transformer: {e}")
    
    # Fallback to Sentence Transformer (384 dimensions)
    service = EmbeddingService(
        provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
        config={"model_name": "all-MiniLM-L6-v2", "dimension": 384}
    )
    return service

@lru_cache()
def get_vector_db() -> BaseVectorDB:
    return VectorDBFactory.create()

@lru_cache()
def get_retrieval_engine(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db: BaseVectorDB = Depends(get_vector_db)
) -> RetrievalEngine:
    return RetrievalEngine(embedding_service, vector_db)

@lru_cache()
def get_response_generator() -> ResponseGenerator:
    return ResponseGenerator()

@lru_cache()
def get_safety_filter() -> SafetyFilter:
    return SafetyFilter()

@lru_cache()
def get_logger_service() -> MongoLogger:
    return MongoLogger()

