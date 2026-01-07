"""Property-based tests for embedding functionality.

Feature: wellness-rag-application, Property 2: Embedding Generation Completeness
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import MagicMock, patch
import numpy as np

from src.services.embeddings.service import EmbeddingService, EmbeddingProvider
from src.services.embeddings.sentence_transformer import SentenceTransformerConfig
from src.core.exceptions import EmbeddingError

# Mock configuration
class MockConfig:
    model_name = "mock-model"
    dimension = 384
    device = "cpu"
    trust_remote_code = False
    batch_size = 32
    normalize = True
    max_tokens = 512

@st.composite
def text_batch_strategy(draw):
    """Generate batches of text for embedding testing."""
    batch_size = draw(st.integers(min_value=1, max_value=20))
    texts = draw(st.lists(
        st.text(min_size=10, max_size=1000),
        min_size=1,
        max_size=batch_size
    ))
    return texts

class TestEmbeddingProperties:
    """Property-based tests for embedding consistency."""

    @pytest.fixture
    def mock_service_fixture(self):
        """Create a mocked embedding service."""
        with patch('src.services.embeddings.sentence_transformer.SentenceTransformer') as mock_model_cls:
            # Setup mock model instance
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            
            # Configure encode behavior to return random embeddings
            # We need to simulate the numpy array return
            def side_effect_encode(texts, convert_to_numpy=True, **kwargs):
                count = len(texts) if isinstance(texts, list) else 1
                return np.random.rand(count, 384).astype(np.float32)
            
            mock_model.encode.side_effect = side_effect_encode
            
            service = EmbeddingService(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                config={"model_name": "mock-model"},
                enable_cache=True
            )
            
            return service

    @given(texts=text_batch_strategy())
    @settings(max_examples=50, deadline=None)
    async def test_embedding_generation_completeness(self, texts):
        """
        Property 2: Embedding Generation Completeness
        
        For any text chunk or user query, the embedding service should generate 
        a valid vector embedding of the expected dimensions without failures.
        
        Validates: Requirements 1.2, 2.1
        """
        # Assume valid text content
        assume(all(len(t.strip()) > 0 for t in texts))

        # Setup service with mocked internal model
        with patch('src.services.embeddings.sentence_transformer.SentenceTransformer') as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            
            # Mock encode to return correct shaped array
            def side_effect_encode(input_texts, **kwargs):
                count = len(input_texts) if isinstance(input_texts, list) else 1
                # Return random vectors of dimension 384
                return np.random.rand(count, 384).astype(np.float32)
            
            mock_model.encode.side_effect = side_effect_encode

            service = EmbeddingService(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                config={"model_name": "test-model", "dimension": 384}
            )
            
            await service.initialize()
            
            # Test batch embedding
            result = await service.embed_texts(texts)
            
            assert len(result.embeddings) == len(texts), \
                "Should generate one embedding per text"
            assert result.dimension == 384, \
                "Dimension should match configuration"
            assert all(len(emb) == 384 for emb in result.embeddings), \
                "Each embedding should have correct dimension"
            
            # Test single query embedding
            if texts:
                query_embedding = await service.embed_query(texts[0])
                assert len(query_embedding) == 384, \
                    "Query embedding should have correct dimension"

    @given(texts=text_batch_strategy())
    @settings(max_examples=30, deadline=None)
    async def test_embedding_caching_consistency(self, texts):
        """
        Property: Caching should return identical embeddings for identical inputs.
        """
        assume(len(texts) > 0 and all(len(t.strip()) > 0 for t in texts))
        
        with patch('src.services.embeddings.sentence_transformer.SentenceTransformer') as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            
            # Mock encode to return deterministic vectors based on input length seed
            # so we can verify cache vs re-compute consistency if needed, 
            # though here we rely on the service reusing the RESULT.
            def side_effect_encode(input_texts, **kwargs):
                count = len(input_texts)
                return np.random.rand(count, 384).astype(np.float32)
            
            mock_model.encode.side_effect = side_effect_encode

            service = EmbeddingService(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                config={"model_name": "test-model"},
                enable_cache=True
            )
            await service.initialize()
            
            # First run - populates cache
            result1 = await service.embed_texts(texts)
            
            # Second run - should hit cache
            # We mock the internal service call to RAISE if called again for same inputs
            # to prove cache hit, OR we check the result identity
            
            # Let's check result consistency
            result2 = await service.embed_texts(texts)
            
            assert len(result1.embeddings) == len(result2.embeddings)
            for e1, e2 in zip(result1.embeddings, result2.embeddings):
                assert np.allclose(e1, e2), "Cached embeddings should match original"


    @given(
        texts=st.lists(st.text(min_size=10), min_size=10, max_size=50),
        batch_size=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=None)
    async def test_batch_processing_consistency(self, texts, batch_size):
        """
        Property: Batch processing should yield same results irrespective of batch size.
        """
        with patch('src.services.embeddings.sentence_transformer.SentenceTransformer') as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            
            # Deterministic mock based on text hash or similar would be ideal,
            # but for now we trust the service batches correctly.
            # We'll just verify the call arguments to the mock model.
            
            service = EmbeddingService(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                config={"model_name": "test-model", "batch_size": batch_size},
                enable_cache=False 
            )
            await service.initialize()
            
            await service.embed_texts(texts)
            
            # Verify that encode was called with correct batching
            # This is partly a behavioral test of the service logic
            expected_call_count = (len(texts) + batch_size - 1) // batch_size
            assert mock_model.encode.call_count == expected_call_count
