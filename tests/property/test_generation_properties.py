"""Unit tests for response generation."""

import pytest
from backend.services.generation.service import ResponseGenerator
from backend.models.schemas import RetrievalResult, Chunk, ChunkMetadata, ContentCategory

class TestGenerationUnit:
    """Tests for ResponseGenerator."""
    
    @pytest.mark.asyncio
    async def test_source_citation_completeness(self):
        """
        Verify source citation completeness.
        """
        generator = ResponseGenerator()
        generator.client = None # Ensure mock mode
        
        chunk = Chunk(
            id="c1",
            content="test content",
            metadata=ChunkMetadata(
                document_id="d1",
                chunk_index=0,
                source="source1",
                category=ContentCategory.WELLNESS,
                tokens=10
            )
        )
        context = [RetrievalResult(chunk=chunk, similarity_score=0.9, relevance_rank=1)]
        
        # This will use the mock fallback in service.py
        response = await generator.generate_response("query", context)
        
        assert len(response.sources) == 1
        assert response.sources[0].chunk_id == "c1"
        assert response.sources[0].source == "source1"

