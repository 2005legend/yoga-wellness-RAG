"""
Base classes for embedding services.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    model_name: str
    dimension: int
    max_tokens: int = 8192
    batch_size: int = 32
    normalize: bool = True
    
    class Config:
        # Fix Pydantic protected namespace warning
        protected_namespaces = ()


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    embeddings: List[List[float]]
    model_name: str
    dimension: int
    token_counts: List[int]
    
    class Config:
        arbitrary_types_allowed = True
        # Fix Pydantic protected namespace warning
        protected_namespaces = ()


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding model."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors."""
        if not self.config.normalize:
            return embeddings
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches for processing."""
        batch_size = self.config.batch_size
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]