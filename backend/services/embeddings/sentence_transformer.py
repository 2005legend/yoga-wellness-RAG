"""
SentenceTransformer-based embedding service.
"""
import asyncio
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .base import BaseEmbeddingService, EmbeddingConfig, EmbeddingResult
from ...core.exceptions import EmbeddingError
from ...core.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerConfig(EmbeddingConfig):
    """Configuration for SentenceTransformer embedding service."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    device: Optional[str] = None  # Auto-detect if None
    trust_remote_code: bool = False


class SentenceTransformerService(BaseEmbeddingService):
    """SentenceTransformer-based embedding service."""
    
    def __init__(self, config: SentenceTransformerConfig):
        super().__init__(config)
        self.config: SentenceTransformerConfig = config
        self._model: Optional[SentenceTransformer] = None
        self._device = None
    
    async def initialize(self) -> None:
        """Initialize the SentenceTransformer model."""
        try:
            logger.info(f"Initializing SentenceTransformer model: {self.config.model_name}")
            
            # Determine device
            if self.config.device:
                self._device = self.config.device
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Using device: {self._device}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                self._load_model
            )
            
            # Verify model dimensions
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.config.dimension:
                logger.warning(
                    f"Model dimension mismatch: expected {self.config.dimension}, "
                    f"got {actual_dim}. Updating config."
                )
                self.config.dimension = actual_dim
            
            logger.info(f"Model initialized successfully. Dimension: {self.config.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model: {e}")
            raise EmbeddingError(f"Model initialization failed: {e}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model (runs in thread pool)."""
        return SentenceTransformer(
            self.config.model_name,
            device=self._device,
            trust_remote_code=self.config.trust_remote_code
        )
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        if not self._model:
            raise EmbeddingError("Model not initialized. Call initialize() first.")
        
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name=self.config.model_name,
                dimension=self.config.dimension,
                token_counts=[]
            )
        
        try:
            logger.debug(f"Embedding {len(texts)} texts")
            
            # Process in batches to manage memory
            all_embeddings = []
            all_token_counts = []
            
            batches = self._batch_texts(texts)
            
            for batch in batches:
                # Run embedding in thread pool
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None,
                    self._embed_batch,
                    batch
                )
                
                all_embeddings.extend(batch_embeddings.tolist())
                
                # Estimate token counts (rough approximation)
                batch_token_counts = [len(text.split()) * 1.3 for text in batch]
                all_token_counts.extend([int(count) for count in batch_token_counts])
            
            logger.debug(f"Generated {len(all_embeddings)} embeddings")
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.config.model_name,
                dimension=self.config.dimension,
                token_counts=all_token_counts
            )
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise EmbeddingError(f"Text embedding failed: {e}")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts (runs in thread pool)."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=min(len(texts), self.config.batch_size)
        )
        
        # Normalize if configured
        if self.config.normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        if not query.strip():
            raise EmbeddingError("Query cannot be empty")
        
        result = await self.embed_texts([query])
        return result.embeddings[0]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._model:
            return {"status": "not_initialized"}
        
        return {
            "model_name": self.config.model_name,
            "dimension": self.config.dimension,
            "device": self._device,
            "max_tokens": self.config.max_tokens,
            "batch_size": self.config.batch_size,
            "normalize": self.config.normalize,
            "status": "initialized"
        }