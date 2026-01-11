"""
Unit tests for embedding services.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from backend.services.embeddings.base import EmbeddingConfig, EmbeddingResult
from backend.services.embeddings.sentence_transformer import (
    SentenceTransformerService,
    SentenceTransformerConfig
)
from backend.services.embeddings.service import (
    EmbeddingService,
    EmbeddingProvider,
    EmbeddingCache,
    EmbeddingServiceFactory
)
from backend.core.exceptions import EmbeddingError, ConfigurationError


class TestEmbeddingConfig:
    """Test embedding configuration."""
    
    def test_embedding_config_creation(self):
        """Test creating embedding configuration."""
        config = EmbeddingConfig(
            model_name="test-model",
            dimension=384,
            max_tokens=512,
            batch_size=16
        )
        
        assert config.model_name == "test-model"
        assert config.dimension == 384
        assert config.max_tokens == 512
        assert config.batch_size == 16
        assert config.normalize is True  # default
    
    def test_sentence_transformer_config(self):
        """Test SentenceTransformer specific configuration."""
        config = SentenceTransformerConfig(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            device="cpu",
            trust_remote_code=True
        )
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.trust_remote_code is True


class TestEmbeddingResult:
    """Test embedding result model."""
    
    def test_embedding_result_creation(self):
        """Test creating embedding result."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            dimension=3,
            token_counts=[5, 7]
        )
        
        assert result.embeddings == embeddings
        assert result.model_name == "test-model"
        assert result.dimension == 3
        assert result.token_counts == [5, 7]


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_cache_set_and_get(self):
        """Test setting and getting from cache."""
        cache = EmbeddingCache(max_size=10, ttl_hours=1)
        
        embedding = [0.1, 0.2, 0.3]
        cache.set("test text", "test-model", embedding)
        
        retrieved = cache.get("test text", "test-model")
        assert retrieved == embedding
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = EmbeddingCache(max_size=10, ttl_hours=1)
        
        retrieved = cache.get("nonexistent text", "test-model")
        assert retrieved is None
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache = EmbeddingCache(max_size=2, ttl_hours=1)
        
        # Add 3 items to cache with size limit of 2
        cache.set("text1", "model", [0.1])
        cache.set("text2", "model", [0.2])
        cache.set("text3", "model", [0.3])
        
        # Should only have 2 items
        assert cache.size() == 2
        
        # First item should be evicted
        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") == [0.2]
        assert cache.get("text3", "model") == [0.3]
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = EmbeddingCache(max_size=10, ttl_hours=1)
        
        cache.set("text1", "model", [0.1])
        cache.set("text2", "model", [0.2])
        
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("text1", "model") is None


class TestEmbeddingServiceFactory:
    """Test embedding service factory."""
    
    def test_create_sentence_transformer_service(self):
        """Test creating SentenceTransformer service."""
        config = {
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384
        }
        
        service = EmbeddingServiceFactory.create_service(
            EmbeddingProvider.SENTENCE_TRANSFORMER,
            config
        )
        
        assert isinstance(service, SentenceTransformerService)
        assert service.config.model_name == "all-MiniLM-L6-v2"
        assert service.config.dimension == 384
    
    def test_unsupported_provider(self):
        """Test error for unsupported provider."""
        with pytest.raises(ConfigurationError):
            EmbeddingServiceFactory.create_service("unsupported", {})


class TestSentenceTransformerService:
    """Test SentenceTransformer embedding service."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SentenceTransformerConfig(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            device="cpu"
        )
    
    @pytest.fixture
    def service(self, config):
        """Create test service."""
        return SentenceTransformerService(config)
    
    @patch('src.services.embeddings.sentence_transformer.SentenceTransformer')
    @patch('src.services.embeddings.sentence_transformer.torch')
    async def test_initialize_success(self, mock_torch, mock_st_class, service):
        """Test successful initialization."""
        # Mock torch.cuda.is_available
        mock_torch.cuda.is_available.return_value = False
        
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4] * 96])  # 384 dims
        mock_st_class.return_value = mock_model
        
        await service.initialize()
        
        assert service._model is not None
        assert service._device == "cpu"
        mock_st_class.assert_called_once()
    
    @patch('src.services.embeddings.sentence_transformer.SentenceTransformer')
    async def test_initialize_failure(self, mock_st_class, service):
        """Test initialization failure."""
        mock_st_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(EmbeddingError):
            await service.initialize()
    
    async def test_embed_texts_not_initialized(self, service):
        """Test embedding texts without initialization."""
        with pytest.raises(EmbeddingError):
            await service.embed_texts(["test"])
    
    @patch('src.services.embeddings.sentence_transformer.SentenceTransformer')
    @patch('src.services.embeddings.sentence_transformer.torch')
    async def test_embed_texts_success(self, mock_torch, mock_st_class, service):
        """Test successful text embedding."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        
        # Mock initialization
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3, 0.4] * 96]),  # For initialization test
            np.array([[0.1, 0.2], [0.3, 0.4]])      # For actual embedding
        ]
        mock_st_class.return_value = mock_model
        
        # Initialize service
        await service.initialize()
        
        # Update config dimension based on mock
        service.config.dimension = 2
        
        # Test embedding
        texts = ["hello world", "test text"]
        result = await service.embed_texts(texts)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert result.model_name == service.config.model_name
        assert result.dimension == 2
        assert len(result.token_counts) == 2
    
    async def test_embed_texts_empty_list(self, service):
        """Test embedding empty text list."""
        # Mock initialization
        with patch('src.services.embeddings.sentence_transformer.SentenceTransformer') as mock_st_class, \
             patch('src.services.embeddings.sentence_transformer.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4] * 96])
            mock_st_class.return_value = mock_model
            
            await service.initialize()
        
        result = await service.embed_texts([])
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == []
        assert result.token_counts == []
    
    @patch('src.services.embeddings.sentence_transformer.SentenceTransformer')
    @patch('src.services.embeddings.sentence_transformer.torch')
    async def test_embed_query(self, mock_torch, mock_st_class, service):
        """Test single query embedding."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3, 0.4] * 96]),  # For initialization
            np.array([[0.1, 0.2, 0.3]])             # For query
        ]
        mock_st_class.return_value = mock_model
        
        await service.initialize()
        service.config.dimension = 3
        
        embedding = await service.embed_query("test query")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3]
    
    async def test_embed_query_empty(self, service):
        """Test embedding empty query."""
        with pytest.raises(EmbeddingError):
            await service.embed_query("")
    
    @patch('src.services.embeddings.sentence_transformer.SentenceTransformer')
    @patch('src.services.embeddings.sentence_transformer.torch')
    async def test_get_model_info(self, mock_torch, mock_st_class, service):
        """Test getting model information."""
        # Test before initialization
        info = service.get_model_info()
        assert info["status"] == "not_initialized"
        
        # Setup mocks and initialize
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4] * 96])
        mock_st_class.return_value = mock_model
        
        await service.initialize()
        
        # Test after initialization
        info = service.get_model_info()
        assert info["status"] == "initialized"
        assert info["model_name"] == service.config.model_name
        assert info["device"] == "cpu"


class TestEmbeddingService:
    """Test main embedding service."""
    
    @pytest.fixture
    def service(self):
        """Create test service."""
        config = {
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "device": "cpu"
        }
        return EmbeddingService(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            config=config,
            enable_cache=True,
            cache_size=10
        )
    
    def test_service_creation(self, service):
        """Test service creation."""
        assert service.provider == EmbeddingProvider.SENTENCE_TRANSFORMER
        assert service.enable_cache is True
        assert service.cache is not None
        assert service._initialized is False
    
    @patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service')
    async def test_initialize(self, mock_factory, service):
        """Test service initialization."""
        # Mock the underlying service
        mock_underlying_service = AsyncMock()
        mock_factory.return_value = mock_underlying_service
        
        await service.initialize()
        
        assert service._initialized is True
        mock_underlying_service.initialize.assert_called_once()
    
    @patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service')
    async def test_embed_texts_with_cache(self, mock_factory, service):
        """Test embedding texts with caching."""
        # Mock the underlying service
        mock_underlying_service = AsyncMock()
        mock_underlying_service.config.model_name = "test-model"
        mock_underlying_service.config.dimension = 3
        mock_underlying_service.embed_texts.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model_name="test-model",
            dimension=3,
            token_counts=[5, 7]
        )
        mock_factory.return_value = mock_underlying_service
        
        await service.initialize()
        
        # First call - should hit underlying service
        texts = ["hello", "world"]
        result1 = await service.embed_texts(texts)
        
        assert len(result1.embeddings) == 2
        mock_underlying_service.embed_texts.assert_called_once()
        
        # Second call with same texts - should use cache
        mock_underlying_service.embed_texts.reset_mock()
        result2 = await service.embed_texts(texts)
        
        assert len(result2.embeddings) == 2
        # Should not call underlying service due to cache
        mock_underlying_service.embed_texts.assert_not_called()
    
    @patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service')
    async def test_embed_query(self, mock_factory, service):
        """Test single query embedding."""
        # Mock the underlying service
        mock_underlying_service = AsyncMock()
        mock_underlying_service.config.model_name = "test-model"
        mock_underlying_service.config.dimension = 3
        mock_underlying_service.embed_texts.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model_name="test-model",
            dimension=3,
            token_counts=[5]
        )
        mock_factory.return_value = mock_underlying_service
        
        await service.initialize()
        
        embedding = await service.embed_query("test query")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3]
    
    async def test_embed_query_empty(self, service):
        """Test embedding empty query."""
        with pytest.raises(EmbeddingError):
            await service.embed_query("")
    
    @patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service')
    async def test_health_check(self, mock_factory, service):
        """Test health check."""
        # Mock the underlying service
        mock_underlying_service = AsyncMock()
        mock_underlying_service.config.model_name = "test-model"
        mock_underlying_service.config.dimension = 3
        mock_underlying_service.embed_texts.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model_name="test-model",
            dimension=3,
            token_counts=[5]
        )
        mock_factory.return_value = mock_underlying_service
        
        await service.initialize()
        
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert "service_info" in health
    
    def test_get_service_info(self, service):
        """Test getting service information."""
        info = service.get_service_info()
        
        assert info["provider"] == EmbeddingProvider.SENTENCE_TRANSFORMER
        assert info["initialized"] is False
        assert info["cache_enabled"] is True
        assert info["cache_size"] == 0
    
    def test_clear_cache(self, service):
        """Test clearing cache."""
        # Add something to cache
        service.cache.set("test", "model", [0.1, 0.2])
        assert service.cache.size() == 1
        
        service.clear_cache()
        assert service.cache.size() == 0
