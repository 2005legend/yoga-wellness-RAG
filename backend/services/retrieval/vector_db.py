"""Vector database service supporting multiple backends."""

from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime

from pydantic import BaseModel, Field

from backend.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import ConfigurationError, RetrievalError
from backend.models.schemas import Chunk, ChunkMetadata

logger = get_logger(__name__)

class SearchResult(BaseModel):
    """Result from vector similarity search."""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

class BaseVectorDB(ABC):
    """Abstract base class for vector database backends."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection to vector database."""
        pass
        
    @abstractmethod
    async def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """
        Upsert chunks and their embeddings.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors corresponding to chunks
            
        Returns:
            Number of chunks successfully upserted
        """
        pass
        
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results sorted by similarity score
        """
        pass
        
    @abstractmethod
    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete chunks by ID."""
        pass
        
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass

class PineconeService(BaseVectorDB):
    """Pinecone vector database implementation."""
    
    def __init__(self):
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            raise ConfigurationError("Pinecone client not installed")
            
        self.api_key = settings.pinecone_api_key
        self.environment = settings.pinecone_environment
        self.index_name = settings.pinecone_index_name
        self.index = None
        
    async def initialize(self) -> None:
        """Initialize Pinecone connection."""
        try:
            self.pinecone.init(api_key=self.api_key, environment=self.environment)
            if self.index_name not in self.pinecone.list_indexes():
                 # Create index if not exists (simplified, typical prod setup might differ)
                 # using typical defaults for this app
                 self.pinecone.create_index(
                    name=self.index_name,
                    dimension=settings.embedding_dimension,
                    metric="cosine"
                 )
            self.index = self.pinecone.Index(self.index_name)
            logger.info(f"Pinecone initialized with index {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise RetrievalError(f"Pinecone initialization failed: {e}")

    async def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        if not self.index:
            await self.initialize()
            
        try:
            # Prepare vectors for upsert
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                # Flatten metadata for Pinecone (no nested objects allowed usually)
                metadata = chunk.metadata.model_dump(exclude={'created_at'})
                metadata['content'] = chunk.content # Store content in metadata or separate DB? Storing here for simplicity
                metadata['timestamp'] = chunk.metadata.created_at.isoformat()
                
                vectors.append((chunk.id, embedding, metadata))
            
            # Upsert in batches of 100
            batch_size = 100
            upsert_count = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                upsert_count += len(batch)
                
            return upsert_count
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            raise RetrievalError(f"Upsert failed: {e}")

    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if not self.index:
            await self.initialize()
            
        try:
            result = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            search_results = []
            for match in result.matches:
                search_results.append(SearchResult(
                    chunk_id=match.id,
                    score=match.score,
                    content=match.metadata.get('content', ''),
                    metadata=match.metadata
                ))
            return search_results
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            raise RetrievalError(f"Search failed: {e}")

    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        if not self.index:
            await self.initialize()
            
        try:
            # Pinecone delete is void return, check availability
            self.index.delete(ids=chunk_ids)
            return len(chunk_ids) # Assume success
        except Exception as e:
            logger.error(f"Pinecone delete failed: {e}")
            raise RetrievalError(f"Delete failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        if not self.index:
            await self.initialize()
        return self.index.describe_index_stats().to_dict()


class ChromaService(BaseVectorDB):
    """ChromaDB vector database implementation."""
    
    def __init__(self):
        try:
            import chromadb
            self.chromadb = chromadb
        except ImportError:
            raise ConfigurationError("ChromaDB client not installed")
            
        self.persist_directory = settings.chroma_persist_directory
        self.collection_name = settings.chroma_collection_name
        self.client = None
        self.collection = None
        
    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        try:
            # Use persistent client
            self.client = self.chromadb.PersistentClient(path=self.persist_directory)
            
            # Check if collection exists and has correct dimension
            try:
                existing_collection = self.client.get_collection(name=self.collection_name)
                # Check if dimension matches expected (from settings or embedding service)
                # If dimension mismatch, delete and recreate
                if hasattr(existing_collection, 'metadata') and existing_collection.metadata:
                    existing_dim = existing_collection.metadata.get('dimension')
                    expected_dim = settings.embedding_dimension
                    if existing_dim and existing_dim != expected_dim:
                        logger.warning(f"Collection dimension mismatch: {existing_dim} vs {expected_dim}. Recreating collection.")
                        self.client.delete_collection(name=self.collection_name)
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            metadata={"dimension": expected_dim}
                        )
                    else:
                        self.collection = existing_collection
                else:
                    self.collection = existing_collection
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": settings.embedding_dimension}
                )
            
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RetrievalError(f"ChromaDB initialization failed: {e}")

    async def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        if not self.collection:
            await self.initialize()
            
        try:
            ids = [c.id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = []
            for c in chunks:
                meta = c.metadata.model_dump(exclude={'created_at'})
                # Ensure category is a string
                if 'category' in meta and hasattr(meta['category'], 'value'):
                    meta['category'] = meta['category'].value
                meta['timestamp'] = c.metadata.created_at.isoformat()
                metadatas.append(meta)
                
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return len(chunks)
        except Exception as e:
            logger.error(f"ChromaDB upsert failed: {e}")
            raise RetrievalError(f"Upsert failed: {e}")

    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if not self.collection:
            await self.initialize()
            
        try:
            # Check dimension mismatch before querying
            embedding_dim = len(query_embedding)
            
            # Try to detect dimension mismatch by checking collection metadata or sample data
            try:
                collection_count = self.collection.count()
                if collection_count > 0:
                    # Try to peek at existing data to check dimension
                    sample = self.collection.peek(limit=1)
                    if sample and 'embeddings' in sample and sample['embeddings'] and len(sample['embeddings']) > 0:
                        existing_dim = len(sample['embeddings'][0]) if sample['embeddings'][0] else None
                        if existing_dim and existing_dim != embedding_dim:
                            logger.warning(f"Embedding dimension mismatch: query={embedding_dim}, collection={existing_dim}. Recreating collection.")
                            try:
                                self.client.delete_collection(name=self.collection_name)
                            except Exception:
                                pass
                            self.collection = self.client.create_collection(
                                name=self.collection_name,
                                metadata={"dimension": embedding_dim}
                            )
                            logger.info(f"Recreated ChromaDB collection with dimension {embedding_dim}")
            except Exception as dim_check_error:
                logger.debug(f"Could not check collection dimension: {dim_check_error}")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
            
            search_results = []
            # Chroma returns lists of lists
            # Handle empty collection gracefully
            if not results or 'ids' not in results or not results['ids'] or len(results['ids']) == 0:
                logger.warning("ChromaDB collection is empty - no chunks found. Please run the knowledge base processing script.")
                return search_results
            
            if results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        chunk_id=results['ids'][0][i],
                        score= 1.0 - (results['distances'][0][i] if 'distances' in results and results['distances'] and len(results['distances'][0]) > i else 0.0), # Convert distance to similarity if needed, though Chroma usage varies
                        content=results['documents'][0][i] if results['documents'] and len(results['documents'][0]) > i else "",
                        metadata=results['metadatas'][0][i] if results['metadatas'] and len(results['metadatas'][0]) > i else {}
                    ))
            return search_results
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            raise RetrievalError(f"Search failed: {e}")
            
    async def delete_chunks(self, chunk_ids: List[str]) -> int:
         if not self.collection:
            await self.initialize()
         try:
             self.collection.delete(ids=chunk_ids)
             return len(chunk_ids)
         except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}")
            raise RetrievalError(f"Delete failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        if not self.collection:
            await self.initialize()
        return {"count": self.collection.count()}


class VectorDBFactory:
    """Factory for creating vector DB services."""
    
    @staticmethod
    def create() -> BaseVectorDB:
        if settings.use_pinecone:
            return PineconeService()
        else:
            return ChromaService()

