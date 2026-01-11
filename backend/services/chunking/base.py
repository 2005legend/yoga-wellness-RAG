"""Base classes and interfaces for document chunking."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from backend.models.schemas import Chunk, ChunkMetadata, ContentCategory


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 512  # Target chunk size in tokens
    chunk_overlap: int = 50  # Overlap between chunks in tokens
    min_chunk_size: int = 100  # Minimum chunk size in tokens
    max_chunk_size: int = 800  # Maximum chunk size in tokens
    preserve_sentences: bool = True  # Try to preserve sentence boundaries
    preserve_paragraphs: bool = True  # Try to preserve paragraph boundaries


class DocumentChunker(ABC):
    """Abstract base class for document chunkers."""
    
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
    
    @abstractmethod
    def chunk_document(
        self, 
        content: str, 
        document_id: str,
        source: str,
        category: ContentCategory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk a document into smaller segments.
        
        Args:
            content: The document content to chunk
            document_id: Unique identifier for the document
            source: Source of the document
            category: Content category
            metadata: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated token count
        """
        pass
