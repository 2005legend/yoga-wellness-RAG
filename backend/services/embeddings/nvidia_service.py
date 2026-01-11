"""
NVIDIA NIM API embedding service implementation.
"""
import aiohttp
from typing import List, Dict, Any, Optional
import numpy as np

from ...core.logging import get_logger
from ...core.exceptions import EmbeddingError, ConfigurationError
from ...config import settings
from .base import BaseEmbeddingService, EmbeddingConfig, EmbeddingResult

logger = get_logger(__name__)


class NvidiaEmbeddingConfig(EmbeddingConfig):
    """Configuration for NVIDIA embedding service."""
    api_key: str
    api_url: str = "https://integrate.api.nvidia.com/v1/embeddings"
    model_name: str = "nvidia/nv-embedqa-e5-v5"
    dimension: int = 1024
    max_tokens: int = 512
    batch_size: int = 10  # NVIDIA API batch limit
    normalize: bool = True


class NvidiaEmbeddingService(BaseEmbeddingService):
    """NVIDIA NIM API embedding service."""
    
    def __init__(self, config: NvidiaEmbeddingConfig):
        super().__init__(config)
        self.config: NvidiaEmbeddingConfig = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize the NVIDIA embedding service."""
        try:
            if not self.config.api_key:
                raise ConfigurationError("NVIDIA API key is required")
            
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            logger.info(f"NVIDIA embedding service initialized with model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA embedding service: {e}")
            raise EmbeddingError(f"Initialization failed: {e}")
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using NVIDIA API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not self.session:
            await self.initialize()
        
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name=self.config.model_name,
                dimension=self.config.dimension,
                token_counts=[]
            )
        
        # Truncate texts that exceed max_tokens
        # Use conservative estimate: 1 token ≈ 3-4 chars (closer to 3 for English)
        # Better to be safe and truncate more aggressively
        max_chars = int(self.config.max_tokens * 3)  # Conservative: 3 chars per token
        truncated_texts = []
        for i, text in enumerate(texts):
            if len(text) > max_chars:
                logger.warning(f"Truncating text {i+1} from {len(text)} chars to {max_chars} chars (max tokens: {self.config.max_tokens})")
                # Truncate at word boundary if possible
                truncated = text[:max_chars]
                # Try to find last space to avoid cutting words
                last_space = truncated.rfind(' ')
                if last_space > max_chars * 0.9:  # If we can find a space in the last 10%
                    truncated = truncated[:last_space]
                truncated_texts.append(truncated)
            else:
                truncated_texts.append(text)
        
        embeddings = []
        token_counts = []
        
        # Process in batches
        batches = self._batch_texts(truncated_texts)
        
        for batch in batches:
            try:
                batch_embeddings, batch_tokens = await self._embed_batch(batch)
                embeddings.extend(batch_embeddings)
                token_counts.extend(batch_tokens)
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                # Fallback: create zero embeddings for failed batch
                for text in batch:
                    embeddings.append([0.0] * self.config.dimension)
                    token_counts.append(len(text.split()))
        
        # Normalize embeddings if configured
        if embeddings and self.config.normalize:
            embeddings_array = np.array(embeddings)
            embeddings_array = self._normalize_embeddings(embeddings_array)
            embeddings = embeddings_array.tolist()
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.config.model_name,
            dimension=self.config.dimension,
            token_counts=token_counts
        )
    
    async def _embed_batch(self, texts: List[str]) -> tuple[List[List[float]], List[int]]:
        """Embed a batch of texts using NVIDIA API."""
        if not self.session:
            raise EmbeddingError("Session not initialized")
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "input": texts,
            "input_type": "query"  # or "passage" for documents
        }
        
        try:
            async with self.session.post(
                self.config.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"NVIDIA API error {response.status}: {error_text}")
                    logger.error(f"Request payload: {payload}")
                    raise EmbeddingError(
                        f"NVIDIA API error {response.status}: {error_text}"
                    )
                
                data = await response.json()
                logger.debug(f"NVIDIA API response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                # Extract embeddings from response
                embeddings = []
                token_counts = []
                
                # Try different response formats
                if isinstance(data, dict) and "data" in data:
                    # Standard format: {"data": [{"embedding": [...], ...}, ...]}
                    for i, item in enumerate(data["data"]):
                        if isinstance(item, dict) and "embedding" in item:
                            embeddings.append(item["embedding"])
                            token_count = item.get("usage", {}).get("total_tokens") if isinstance(item.get("usage"), dict) else None
                            if not token_count and i < len(texts):
                                token_count = len(texts[i].split())
                            token_counts.append(token_count or 100)
                elif isinstance(data, list):
                    # Direct list format: [{"embedding": [...], ...}, ...]
                    for i, item in enumerate(data):
                        if isinstance(item, dict) and "embedding" in item:
                            embeddings.append(item["embedding"])
                            token_count = item.get("usage", {}).get("total_tokens") if isinstance(item.get("usage"), dict) else None
                            if not token_count and i < len(texts):
                                token_count = len(texts[i].split())
                            token_counts.append(token_count or 100)
                elif isinstance(data, dict) and "embeddings" in data:
                    # Alternative format: {"embeddings": [[...], [...], ...]}
                    for i, embedding in enumerate(data["embeddings"]):
                        embeddings.append(embedding)
                        token_counts.append(len(texts[i].split()) if i < len(texts) else 100)
                else:
                    # Log the actual response for debugging
                    logger.error(f"Unexpected NVIDIA API response format. Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                    logger.error(f"Response sample: {str(data)[:500]}")
                    raise EmbeddingError(f"Unexpected response format. Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                if not embeddings:
                    raise EmbeddingError(f"No embeddings found in NVIDIA API response. Response: {str(data)[:200]}")
                
                # Ensure we have the same number of embeddings as texts
                if len(embeddings) != len(texts):
                    logger.warning(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                    # Pad or truncate as needed
                    while len(embeddings) < len(texts):
                        embeddings.append([0.0] * self.config.dimension)
                        token_counts.append(100)
                    embeddings = embeddings[:len(texts)]
                    token_counts = token_counts[:len(texts)]
                
                return embeddings, token_counts
                
        except aiohttp.ClientError as e:
            raise EmbeddingError(f"HTTP error calling NVIDIA API: {e}")
        except Exception as e:
            raise EmbeddingError(f"Unexpected error: {e}")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = await self.embed_texts([query])
        return result.embeddings[0]
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "provider": "nvidia",
            "model_name": self.config.model_name,
            "dimension": self.config.dimension,
            "max_tokens": self.config.max_tokens,
            "normalize": self.config.normalize
        }

