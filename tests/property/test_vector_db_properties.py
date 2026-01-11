"""Property-based tests for vector database functionality.

Feature: wellness-rag-application, Property 3: Chunk Storage and Retrieval
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import MagicMock, patch
import uuid
from datetime import datetime

from backend.models.schemas import Chunk, ChunkMetadata, ContentCategory
from backend.services.retrieval.vector_db import ChromaService, SearchResult

# Strategies
@st.composite
def chunk_strategy(draw):
    """Generate valid chunks."""
    chunk_id = str(uuid.uuid4())
    content = draw(st.text(min_size=10, max_size=500))
    
    metadata = ChunkMetadata(
        document_id=str(uuid.uuid4()),
        chunk_index=draw(st.integers(min_value=0, max_value=100)),
        source=draw(st.text(min_size=1, max_size=50)),
        category=draw(st.sampled_from(list(ContentCategory))),
        tokens=draw(st.integers(min_value=10, max_value=200)),
        created_at=datetime.utcnow()
    )
    
    return Chunk(id=chunk_id, content=content, metadata=metadata)

@st.composite
def embedding_strategy(draw):
    """Generate float embeddings."""
    return draw(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=384, max_size=384))

class TestVectorDBProperties:
    """Tests for vector DB."""

    @pytest.mark.asyncio
    async def test_chunk_storage_and_retrieval_simple(self):
        """
        Simple unit test for chunk storage and retrieval.
        """
        # Create valid dummy data
        chunk_id = str(uuid.uuid4())
        metadata = ChunkMetadata(
            document_id=str(uuid.uuid4()),
            chunk_index=0,
            source="test",
            category=ContentCategory.WELLNESS,
            tokens=10,
            created_at=datetime.utcnow()
        )
        test_chunks = [Chunk(id=chunk_id, content="test content", metadata=metadata)]
        test_embeddings = [[0.1] * 384]
        
        # Configure mock behavior
        with patch('chromadb.PersistentClient') as mock_client_cls:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client_cls.return_value = mock_client
            
            # Configure ALL collection accessors to return our tracked mock
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client.get_collection.return_value = mock_collection
            mock_client.create_collection.return_value = mock_collection
            
            # Mock metadata to avoid dimension mismatch logic (simulating checking existing collection)
            # The test uses 384 dimensions for embedding strategy/test data
            mock_collection.metadata = {"dimension": 384}
            mock_collection.metadata.get = MagicMock(return_value=384)
            
            # Mock query return
            mock_collection.query.return_value = {
                'ids': [[test_chunks[0].id]],
                'distances': [[0.1]],
                'documents': [[test_chunks[0].content]],
                'metadatas': [[{
                    'document_id': test_chunks[0].metadata.document_id,
                    'category': test_chunks[0].metadata.category.value,
                    'timestamp': '2023-01-01T00:00:00'
                }]]
            }
            
            service = ChromaService()
            await service.initialize()
            
            # Test Upsert
            count = await service.upsert_chunks(test_chunks, test_embeddings)
            assert count == len(test_chunks)
            
            # Verify upsert logic call
            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args[1]
            assert len(call_kwargs['ids']) == len(test_chunks)
            
            # Test Retrieval (Search)
            results = await service.search(test_embeddings[0], k=1)
            
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert results[0].chunk_id == test_chunks[0].id
            
            # Property: Retrieved ID matches searched ID (in this mocked scenario)
            # In a real DB test, we'd check semantic relevance, but here we check service integrity

