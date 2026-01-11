"""
Retrieval engine implementation.
"""
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from ...core.logging import get_logger
from ...core.exceptions import RetrievalError
from ...models.schemas import RetrievalResult, Chunk, ChunkMetadata, ContentCategory
from ..embeddings.service import EmbeddingService
from .vector_db import BaseVectorDB, SearchResult

logger = get_logger(__name__)


class RetrievalEngine:
    """
    Retrieval engine orchestrating embedding and vector search.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db: BaseVectorDB
    ):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        
    async def initialize(self) -> None:
        """Initialize dependencies."""
        await self.embedding_service.initialize()
        await self.vector_db.initialize()
        
    async def retrieve_relevant_chunks(
        self,
        query: str,
        max_results: int = 5,
        min_similarity: float = 0.6
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query using semantic search.
        
        Args:
            query: User query
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of retrieval results
        """
        try:
            # Ensure services are initialized
            if not hasattr(self.embedding_service, '_initialized') or not self.embedding_service._initialized:
                await self.embedding_service.initialize()
            
            if not hasattr(self.vector_db, 'collection') and not hasattr(self.vector_db, 'index'):
                await self.vector_db.initialize()
            
            # 1. Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            
            # 2. Search vector database
            try:
                search_results = await self.vector_db.search(
                    query_embedding=query_embedding,
                    k=max_results
                )
            except Exception as search_error:
                logger.error(f"Vector database search error: {search_error}", exc_info=True)
                # Return empty results rather than failing
                search_results = []
            
            # 3. Format and filter results
            results = []
            rank = 1
            for res in search_results:
                if res.score < min_similarity:
                    continue
                
                # Reconstruct Chunk object from metadata/content
                # Ensure metadata has required fields or defaults
                meta_dict = res.metadata.copy() if res.metadata else {}
                
                # Provide defaults for required ChunkMetadata fields
                document_id = meta_dict.get('document_id', res.chunk_id.split('_chunk_')[0] if '_chunk_' in res.chunk_id else res.chunk_id)
                chunk_index = meta_dict.get('chunk_index', 0)
                source = meta_dict.get('source', 'unknown')
                
                # Handle category - could be string or enum
                category_str = meta_dict.get('category', 'WELLNESS')
                if isinstance(category_str, str):
                    try:
                        category = ContentCategory(category_str)
                    except ValueError:
                        category = ContentCategory.WELLNESS
                else:
                    category = category_str
                
                # Handle tokens - estimate if missing
                tokens = meta_dict.get('tokens', len(res.content.split()) * 1.3)  # Rough estimate
                if not isinstance(tokens, int):
                    tokens = int(tokens)
                
                # Handle created_at timestamp
                created_at = meta_dict.get('created_at')
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        created_at = datetime.utcnow()
                elif created_at is None:
                    created_at = datetime.utcnow()
                elif not isinstance(created_at, datetime):
                    created_at = datetime.utcnow()
                
                # Create ChunkMetadata with all required fields
                chunk_metadata = ChunkMetadata(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    source=source,
                    category=category,
                    tokens=tokens,
                    created_at=created_at
                )
                
                # Create Chunk object
                chunk = Chunk(
                    id=res.chunk_id,
                    content=res.content,
                    metadata=chunk_metadata
                )
                
                results.append(RetrievalResult(
                    chunk=chunk,
                    similarity_score=res.score,
                    relevance_rank=rank
                ))
                rank += 1
                
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")

    async def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search (not fully implemented in VectorDB yet, falling back to semantic).
        """
        # Placeholder for hybrid search if vector DB supports it
        # For now, just alias to retrieve_relevant_chunks
        return await self.retrieve_relevant_chunks(query, max_results)
