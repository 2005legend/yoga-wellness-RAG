"""Property-based tests for retrieval engine functionality.

Feature: wellness-rag-application, Property 5: Similarity-Based Retrieval Accuracy
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import AsyncMock, MagicMock

from backend.services.retrieval.engine import RetrievalEngine, RetrievalResult
from backend.services.embeddings.service import EmbeddingService
from backend.services.retrieval.vector_db import BaseVectorDB, SearchResult
from backend.models.schemas import Chunk, ChunkMetadata, ContentCategory

@st.composite
def retrieval_result_strategy(draw):
    """Generate search results."""
    score = draw(st.floats(min_value=0.0, max_value=1.0))
    chunk_id = draw(st.text(min_size=5))
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        content=draw(st.text(min_size=10)),
        metadata={
            "document_id": "doc1",
            "chunk_index": 0,
            "source": "test",
            "category": "WELLNESS",
            "tokens": 100
        }
    )

class TestRetrievalProperties:
    """Tests for RetrievalEngine."""

    @pytest.mark.asyncio
    @given(
        query=st.text(min_size=5),
        search_results=st.lists(retrieval_result_strategy(), min_size=1, max_size=10)
    )
    async def test_retrieval_ranking(self, query, search_results):
        """
        Property 5: Similarity-Based Retrieval Accuracy
        
        For any user query, retrieved chunks should be ranked by semantic similarity.
        """
        # Sort search results by score descending (simulating vector DB behavior)
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        # Mock dependencies
        mock_embedding_service = AsyncMock(spec=EmbeddingService)
        mock_embedding_service.embed_query.return_value = [0.1] * 384
        
        mock_vector_db = AsyncMock(spec=BaseVectorDB)
        mock_vector_db.search.return_value = sorted_results
        
        engine = RetrievalEngine(mock_embedding_service, mock_vector_db)
        
        # Execute retrieval
        results = await engine.retrieve_relevant_chunks(query, max_results=len(search_results), min_similarity=0.0)
        
        # Verification
        assert len(results) <= len(search_results)
        
        # Check ranking order
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i+1].similarity_score, \
                    "Results should be ranked by similarity score descending"
                    
        # Check conversion integrity
        for res in results:
            assert isinstance(res, RetrievalResult)
            assert isinstance(res.chunk, Chunk)
            # Verify rank assignment
            assert res.relevance_rank > 0
            
    @pytest.mark.asyncio
    @given(query=st.text(min_size=5))
    async def test_min_similarity_filtering(self, query):
        """
        Verify min_similarity filters out low scores.
        """
        results = [
            SearchResult(chunk_id="1", score=0.9, content="high", metadata={}),
            SearchResult(chunk_id="2", score=0.5, content="low", metadata={})
        ]
        
        mock_embedding_service = AsyncMock(spec=EmbeddingService)
        mock_embedding_service.embed_query.return_value = [0.0] * 384
        
        mock_vector_db = AsyncMock(spec=BaseVectorDB)
        mock_vector_db.search.return_value = results
        
        engine = RetrievalEngine(mock_embedding_service, mock_vector_db)
        
        # Retrieve with threshold 0.8
        filtered_results = await engine.retrieve_relevant_chunks(query, min_similarity=0.8)
        
        assert len(filtered_results) == 1
        assert filtered_results[0].similarity_score >= 0.8
        assert filtered_results[0].chunk.content == "high"


