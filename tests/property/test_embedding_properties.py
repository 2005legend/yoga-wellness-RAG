"""
Property-based tests for embedding generation services.
**Feature: wellness-rag-application**
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from typing import List

from backend.services.embeddings.service import (
    EmbeddingService,
    EmbeddingProvider,
    EmbeddingCache
)
from backend.services.embeddings.base import EmbeddingResult
from backend.core.exceptions import EmbeddingError


# Test data generators
@st.composite
def text_lists(draw):
    """Generate lists of text strings for testing."""
    # Generate 1-10 texts
    size = draw(st.integers(min_value=1, max_value=10))
    texts = []
    for _ in range(size):
        # Generate meaningful text (alphanumeric words separated by spaces)
        words = draw(st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
                min_size=1,
                max_size=20
            ),
            min_size=1,
            max_size=10
        ))
        text = ' '.join(words)
        assume(len(text.strip()) > 0)  # Ensure non-empty text
        texts.append(text)
    return texts


@st.composite
def embedding_dimensions(draw):
    """Generate valid embedding dimensions."""
    return draw(st.integers(min_value=128, max_value=1536))


@st.composite
def yoga_queries(draw):
    """Generate yoga-related query strings."""
    yoga_terms = [
        "downward dog", "warrior pose", "tree pose", "child's pose", "mountain pose",
        "sun salutation", "pranayama", "meditation", "flexibility", "balance",
        "yoga benefits", "beginner yoga", "advanced poses", "breathing exercises"
    ]
    
    base_query = draw(st.sampled_from(yoga_terms))
    
    # Add some variation
    variations = [
        f"How do I do {base_query}?",
        f"What are the benefits of {base_query}?",
        f"Is {base_query} safe for beginners?",
        f"Can you explain {base_query}?",
        f"Tell me about {base_query}"
    ]
    
    return draw(st.sampled_from(variations))


class TestEmbeddingGenerationProperties:
    """Property-based tests for embedding generation completeness."""
    
    @given(texts=text_lists())
    @settings(max_examples=100, deadline=30000)
    def test_embedding_generation_completeness(self, texts):
        """
        Property 8: Embedding Generation Consistency
        For any list of valid texts, the embedding service should generate 
        embeddings for all texts with consistent dimensions.
        **Validates: Requirements 1.2, 2.1**
        """
        async def run_test():
            # Setup
            dimension = 384  # Standard dimension for all-MiniLM-L6-v2
            
            # Create mock embeddings
            mock_embeddings = []
            for _ in texts:
                embedding = [0.1] * dimension  # Simple consistent embedding
                mock_embeddings.append(embedding)
            
            # Mock the underlying service
            with patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service') as mock_factory:
                mock_underlying_service = AsyncMock()
                mock_underlying_service.config.model_name = "test-model"
                mock_underlying_service.config.dimension = dimension
                mock_underlying_service.embed_texts.return_value = EmbeddingResult(
                    embeddings=mock_embeddings,
                    model_name="test-model",
                    dimension=dimension,
                    token_counts=[len(text.split()) for text in texts]
                )
                mock_factory.return_value = mock_underlying_service
                
                # Create service
                service = EmbeddingService(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                    config={"model_name": "test-model", "dimension": dimension}
                )
                
                # Test
                result = await service.embed_texts(texts)
                
                # Property assertions
                assert len(result.embeddings) == len(texts), \
                    f"Should generate embeddings for all {len(texts)} texts"
                
                assert all(len(emb) == dimension for emb in result.embeddings), \
                    f"All embeddings should have dimension {dimension}"
                
                assert result.model_name == "test-model", \
                    "Result should include correct model name"
                
                assert result.dimension == dimension, \
                    f"Result should report correct dimension {dimension}"
                
                assert len(result.token_counts) == len(texts), \
                    "Should provide token counts for all texts"
                
                # Verify underlying service was called correctly
                mock_underlying_service.embed_texts.assert_called_once_with(texts)
        
        # Run the async test
        asyncio.run(run_test())
    
    @given(
        texts=text_lists(),
        dimension=embedding_dimensions()
    )
    @settings(max_examples=100, deadline=30000)
    def test_embedding_dimension_consistency(self, texts, dimension):
        """
        Property: Embedding Dimension Consistency
        For any valid texts and embedding dimension, all generated embeddings 
        should have the same dimension as configured.
        **Validates: Requirements 1.2**
        """
        async def run_test():
            # Create mock embeddings with correct dimension
            mock_embeddings = []
            for _ in texts:
                embedding = [0.1] * dimension
                mock_embeddings.append(embedding)
            
            # Mock the underlying service
            with patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service') as mock_factory:
                mock_underlying_service = AsyncMock()
                mock_underlying_service.config.model_name = "test-model"
                mock_underlying_service.config.dimension = dimension
                mock_underlying_service.embed_texts.return_value = EmbeddingResult(
                    embeddings=mock_embeddings,
                    model_name="test-model",
                    dimension=dimension,
                    token_counts=[len(text.split()) for text in texts]
                )
                mock_factory.return_value = mock_underlying_service
                
                # Create service
                service = EmbeddingService(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                    config={"model_name": "test-model", "dimension": dimension}
                )
                
                # Test
                result = await service.embed_texts(texts)
                
                # Property assertions
                for i, embedding in enumerate(result.embeddings):
                    assert len(embedding) == dimension, \
                        f"Embedding {i} should have dimension {dimension}, got {len(embedding)}"
                
                assert result.dimension == dimension, \
                    f"Result dimension should match configured dimension {dimension}"
        
        # Run the async test
        asyncio.run(run_test())
    
    @given(texts=text_lists())
    @settings(max_examples=100, deadline=30000)
    def test_embedding_deterministic_behavior(self, texts):
        """
        Property: Embedding Deterministic Behavior
        For any list of texts, calling embed_texts multiple times with the same 
        inputs should produce identical results (when caching is disabled).
        **Validates: Requirements 1.2**
        """
        async def run_test():
            dimension = 384
            
            # Create consistent mock embeddings
            mock_embeddings = []
            for i, _ in enumerate(texts):
                # Create deterministic embeddings based on text index
                embedding = [0.1 + (i * 0.01)] * dimension
                mock_embeddings.append(embedding)
            
            # Mock the underlying service to return consistent results
            with patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service') as mock_factory:
                mock_underlying_service = AsyncMock()
                mock_underlying_service.config.model_name = "test-model"
                mock_underlying_service.config.dimension = dimension
                mock_underlying_service.embed_texts.return_value = EmbeddingResult(
                    embeddings=mock_embeddings,
                    model_name="test-model",
                    dimension=dimension,
                    token_counts=[len(text.split()) for text in texts]
                )
                mock_factory.return_value = mock_underlying_service
                
                # Create service with caching disabled
                service = EmbeddingService(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                    config={"model_name": "test-model", "dimension": dimension},
                    enable_cache=False
                )
                
                # Test multiple calls
                result1 = await service.embed_texts(texts, use_cache=False)
                result2 = await service.embed_texts(texts, use_cache=False)
                
                # Property assertions
                assert len(result1.embeddings) == len(result2.embeddings), \
                    "Multiple calls should return same number of embeddings"
                
                for i, (emb1, emb2) in enumerate(zip(result1.embeddings, result2.embeddings)):
                    assert emb1 == emb2, \
                        f"Embedding {i} should be identical across calls"
                
                assert result1.model_name == result2.model_name, \
                    "Model name should be consistent across calls"
                
                assert result1.dimension == result2.dimension, \
                    "Dimension should be consistent across calls"
        
        # Run the async test
        asyncio.run(run_test())
    
    @given(texts=text_lists())
    @settings(max_examples=100, deadline=30000)
    def test_embedding_cache_effectiveness(self, texts):
        """
        Property: Embedding Cache Effectiveness
        For any list of texts, when caching is enabled, the second call with 
        the same texts should use cached results and not call the underlying service.
        **Validates: Requirements 8.2**
        """
        async def run_test():
            dimension = 384
            
            # Create unique texts to avoid cache conflicts with duplicates
            unique_texts = list(dict.fromkeys(texts))  # Remove duplicates while preserving order
            if not unique_texts:
                unique_texts = ["test_text"]
            
            # Create consistent mock embeddings based on text content, not index
            mock_embeddings = []
            for text in unique_texts:
                # Create deterministic embedding based on text hash
                text_hash = hash(text) % 100
                embedding = [0.1 + (text_hash * 0.001)] * dimension
                mock_embeddings.append(embedding)
            
            # Mock the underlying service
            with patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service') as mock_factory:
                mock_underlying_service = AsyncMock()
                mock_underlying_service.config.model_name = "test-model"
                mock_underlying_service.config.dimension = dimension
                
                # Create a consistent result that will be returned every time
                consistent_result = EmbeddingResult(
                    embeddings=mock_embeddings.copy(),  # Use copy to ensure consistency
                    model_name="test-model",
                    dimension=dimension,
                    token_counts=[len(text.split()) for text in unique_texts]
                )
                mock_underlying_service.embed_texts.return_value = consistent_result
                mock_factory.return_value = mock_underlying_service
                
                # Create service with caching enabled
                service = EmbeddingService(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                    config={"model_name": "test-model", "dimension": dimension},
                    enable_cache=True,
                    cache_size=100
                )
                
                # First call - should hit underlying service
                result1 = await service.embed_texts(unique_texts, use_cache=True)
                
                # Verify underlying service was called
                assert mock_underlying_service.embed_texts.call_count == 1
                
                # Second call - should use cache (don't reset mock, just check call count)
                result2 = await service.embed_texts(unique_texts, use_cache=True)
                
                # Property assertions
                assert len(result1.embeddings) == len(result2.embeddings), \
                    "Cached results should have same number of embeddings"
                
                for i, (emb1, emb2) in enumerate(zip(result1.embeddings, result2.embeddings)):
                    assert emb1 == emb2, \
                        f"Cached embedding {i} should be identical to original"
                
                # Verify underlying service was called only once (cache hit on second call)
                assert mock_underlying_service.embed_texts.call_count == 1, \
                    "Underlying service should only be called once due to caching"
        
        # Run the async test
        asyncio.run(run_test())
    
    @given(query_text=yoga_queries())
    @settings(max_examples=100, deadline=30000)
    def test_single_query_embedding(self, query_text):
        """
        Property: Single Query Embedding
        For any valid yoga query text, embed_query should return a single embedding 
        vector with the correct dimension.
        **Validates: Requirements 2.1**
        """
        async def run_test():
            dimension = 384
            mock_embedding = [0.1] * dimension
            
            # Mock the underlying service
            with patch('src.services.embeddings.service.EmbeddingServiceFactory.create_service') as mock_factory:
                mock_underlying_service = AsyncMock()
                mock_underlying_service.config.model_name = "test-model"
                mock_underlying_service.config.dimension = dimension
                mock_underlying_service.embed_texts.return_value = EmbeddingResult(
                    embeddings=[mock_embedding],
                    model_name="test-model",
                    dimension=dimension,
                    token_counts=[len(query_text.split())]
                )
                mock_factory.return_value = mock_underlying_service
                
                # Create service and ensure it's properly initialized
                service = EmbeddingService(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                    config={"model_name": "test-model", "dimension": dimension}
                )
                
                # Manually set the service to ensure it's not None
                service._service = mock_underlying_service
                
                # Test
                result = await service.embed_query(query_text)
                
                # Property assertions
                assert isinstance(result, list), \
                    "Query embedding should return a list"
                
                assert len(result) == dimension, \
                    f"Query embedding should have dimension {dimension}"
                
                assert all(isinstance(x, (int, float)) for x in result), \
                    "All embedding values should be numeric"
                
                # Verify underlying service was called with single text
                mock_underlying_service.embed_texts.assert_called_once_with([query_text])
        
        # Run the async test
        asyncio.run(run_test())


class TestEmbeddingCacheProperties:
    """Property-based tests for embedding cache functionality."""
    
    @given(
        texts=text_lists(),
        cache_size=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=30000)
    def test_cache_size_limit_enforcement(self, texts, cache_size):
        """
        Property: Cache Size Limit Enforcement
        For any cache size limit, the cache should never exceed that limit 
        regardless of how many items are added.
        **Validates: Requirements 8.2**
        """
        cache = EmbeddingCache(max_size=cache_size, ttl_hours=1)
        
        # Add more items than cache size
        for i, text in enumerate(texts):
            embedding = [0.1 + (i * 0.01)] * 384
            cache.set(text, "test-model", embedding)
        
        # Property assertion
        assert cache.size() <= cache_size, \
            f"Cache size {cache.size()} should not exceed limit {cache_size}"
        
        # If we added more items than cache size, verify LRU eviction
        if len(texts) > cache_size:
            assert cache.size() == cache_size, \
                f"Cache should be at maximum size {cache_size} when overfilled"
    
    @given(texts=text_lists())
    @settings(max_examples=100, deadline=30000)
    def test_cache_round_trip_consistency(self, texts):
        """
        Property: Cache Round Trip Consistency
        For any text and embedding, storing in cache and retrieving should 
        return the exact same embedding.
        **Validates: Requirements 8.2**
        """
        cache = EmbeddingCache(max_size=100, ttl_hours=1)
        dimension = 384
        
        # Store and retrieve each text
        for i, text in enumerate(texts):
            original_embedding = [0.1 + (i * 0.01)] * dimension
            
            # Store in cache
            cache.set(text, "test-model", original_embedding)
            
            # Retrieve from cache
            retrieved_embedding = cache.get(text, "test-model")
            
            # Property assertion
            assert retrieved_embedding == original_embedding, \
                f"Retrieved embedding should match original for text: {text[:50]}..."
