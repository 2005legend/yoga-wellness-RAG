"""Vector database service supporting multiple backends."""

from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime

from pydantic import BaseModel, Field

from src.config import settings
from src.core.logging import get_logger
from src.core.exceptions import ConfigurationError, RetrievalError
from src.models.schemas import Chunk, ChunkMetadata

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
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
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
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
            
            search_results = []
            # Chroma returns lists of lists
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        chunk_id=results['ids'][0][i],
                        score= 1.0 - (results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0), # Convert distance to similarity if needed, though Chroma usage varies
                        content=results['documents'][0][i] if results['documents'] else "",
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
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
