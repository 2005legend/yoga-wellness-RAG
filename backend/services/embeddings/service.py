"""
Main embedding service with factory pattern and caching.
"""
import asyncio
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import hashlib
import json
from datetime import datetime, timedelta

from .base import BaseEmbeddingService, EmbeddingConfig, EmbeddingResult
from .sentence_transformer import SentenceTransformerService, SentenceTransformerConfig
from .nvidia_service import NvidiaEmbeddingService, NvidiaEmbeddingConfig
from ...core.exceptions import EmbeddingError, ConfigurationError
from ...core.logging import get_logger
from ...config import settings

logger = get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    NVIDIA = "nvidia"
    # Future providers can be added here
    # OPENAI = "openai"
    # HUGGINGFACE = "huggingface"


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._generate_key(text, model_name)
        
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["embedding"]
            else:
                # Remove expired entry
                del self._cache[key]
        
        return None
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]["timestamp"]
            )
            del self._cache[oldest_key]
        
        key = self._generate_key(text, model_name)
        self._cache[key] = {
            "embedding": embedding,
            "timestamp": datetime.now()
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class EmbeddingServiceFactory:
    """Factory for creating embedding services."""
    
    @staticmethod
    def create_service(
        provider: EmbeddingProvider,
        config: Dict[str, Any]
    ) -> BaseEmbeddingService:
        """Create embedding service based on provider and config."""
        
        if provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            service_config = SentenceTransformerConfig(**config)
            return SentenceTransformerService(service_config)
        
        elif provider == EmbeddingProvider.NVIDIA:
            # Get API key from config or settings
            api_key = config.get("api_key") or getattr(settings, "nvidia_embedding_api_key", None)
            if not api_key:
                raise ConfigurationError("NVIDIA embedding API key is required")
            
            nvidia_config = NvidiaEmbeddingConfig(
                api_key=api_key,
                api_url=config.get("api_url", "https://integrate.api.nvidia.com/v1/embeddings"),
                model_name=config.get("model_name", "nvidia/nv-embedqa-e5-v5"),
                dimension=config.get("dimension", 1024),
                max_tokens=config.get("max_tokens", 512),
                batch_size=config.get("batch_size", 10),
                normalize=config.get("normalize", True)
            )
            return NvidiaEmbeddingService(nvidia_config)
        
        # Future providers can be added here
        # elif provider == EmbeddingProvider.OPENAI:
        #     service_config = OpenAIConfig(**config)
        #     return OpenAIService(service_config)
        
        else:
            raise ConfigurationError(f"Unsupported embedding provider: {provider}")


class EmbeddingService:
    """
    Main embedding service with caching and provider abstraction.
    """
    
    def __init__(
        self,
        provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMER,
        config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl_hours: int = 24
    ):
        self.provider = provider
        self.config = config or {}
        self.enable_cache = enable_cache
        
        # Initialize cache
        if enable_cache:
            self.cache = EmbeddingCache(cache_size, cache_ttl_hours)
        else:
            self.cache = None
        
        # Initialize service
        self._service: Optional[BaseEmbeddingService] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the embedding service."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Initializing embedding service with provider: {self.provider}")
            
            # Create service instance
            self._service = EmbeddingServiceFactory.create_service(
                self.provider,
                self.config
            )
            
            # Initialize the underlying service
            await self._service.initialize()
            
            self._initialized = True
            logger.info("Embedding service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise EmbeddingError(f"Service initialization failed: {e}")
    
    async def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            use_cache: Whether to use caching for individual texts
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name=self._service.config.model_name,
                dimension=self._service.config.dimension,
                token_counts=[]
            )
        
        # Check cache for individual texts if enabled
        cached_embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        if self.enable_cache and use_cache and self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self._service.config.model_name)
                if cached:
                    cached_embeddings.append((i, cached))
                else:
                    texts_to_embed.append(text)
                    cache_indices.append(i)
        else:
            texts_to_embed = texts
            cache_indices = list(range(len(texts)))
        
        # Embed texts not in cache
        if texts_to_embed:
            logger.debug(f"Embedding {len(texts_to_embed)} texts (cache hits: {len(cached_embeddings)})")
            result = await self._service.embed_texts(texts_to_embed)
            
            # Cache new embeddings
            if self.enable_cache and use_cache and self.cache:
                for text, embedding in zip(texts_to_embed, result.embeddings):
                    self.cache.set(text, self._service.config.model_name, embedding)
        else:
            # All texts were cached
            result = EmbeddingResult(
                embeddings=[],
                model_name=self._service.config.model_name,
                dimension=self._service.config.dimension,
                token_counts=[]
            )
        
        # Combine cached and new embeddings in original order
        final_embeddings = [None] * len(texts)
        final_token_counts = [None] * len(texts)
        
        # Place cached embeddings
        for i, embedding in cached_embeddings:
            final_embeddings[i] = embedding
            final_token_counts[i] = len(texts[i].split()) * 1.3  # Rough estimate
        
        # Place new embeddings
        for i, (cache_idx, embedding, token_count) in enumerate(
            zip(cache_indices, result.embeddings, result.token_counts)
        ):
            final_embeddings[cache_idx] = embedding
            final_token_counts[cache_idx] = token_count
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            model_name=result.model_name,
            dimension=result.dimension,
            token_counts=[int(count) for count in final_token_counts]
        )
    
    async def embed_query(self, query: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector as list of floats
        """
        if not query.strip():
            raise EmbeddingError("Query cannot be empty")
        
        # Check cache first
        if self.enable_cache and use_cache and self.cache:
            cached = self.cache.get(query, self._service.config.model_name)
            if cached:
                logger.debug("Query embedding found in cache")
                return cached
        
        # Generate new embedding
        result = await self.embed_texts([query], use_cache=use_cache)
        return result.embeddings[0]
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the embedding service."""
        info = {
            "provider": self.provider,
            "initialized": self._initialized,
            "cache_enabled": self.enable_cache,
        }
        
        if self.cache:
            info["cache_size"] = self.cache.size()
        
        if self._service:
            if hasattr(self._service, 'get_model_info'):
                info.update(self._service.get_model_info())
            else:
                info.update({
                    "model_name": self._service.config.model_name,
                    "dimension": self._service.config.dimension
                })
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the embedding service."""
        try:
            if not self._initialized:
                return {"status": "unhealthy", "reason": "not_initialized"}
            
            # Test with a simple query
            test_embedding = await self.embed_query("health check test", use_cache=False)
            
            if not test_embedding or len(test_embedding) != self._service.config.dimension:
                return {"status": "unhealthy", "reason": "invalid_embedding_output"}
            
            return {
                "status": "healthy",
                "service_info": self.get_service_info()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "reason": str(e)}