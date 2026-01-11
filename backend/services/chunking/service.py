"""Main chunking service that orchestrates document processing and chunking."""

import re
import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from backend.core.logging import LoggerMixin
from backend.core.exceptions import ChunkingError
from backend.config import settings
from backend.models.schemas import Chunk, ContentCategory
from backend.services.chunking.base import ChunkingConfig
from backend.services.chunking.semantic_chunker import SemanticChunker
from backend.services.chunking.document_processor import DocumentProcessor


class ChunkingService(LoggerMixin):
    """
    Main service for document chunking operations.
    
    This service orchestrates the entire chunking pipeline:
    1. Document processing (file format handling)
    2. Content cleaning and preprocessing
    3. Semantic chunking with metadata preservation
    4. Chunk validation and optimization
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        """
        Initialize the chunking service.
        
        Args:
            config: Chunking configuration, uses defaults if None
        """
        self.config = config or ChunkingConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        self.chunker = SemanticChunker(self.config)
        self.processor = DocumentProcessor()
        
        self.log_event(
            "ChunkingService initialized",
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def chunk_file(
        self, 
        file_path: str,
        document_id: Optional[str] = None,
        category: Optional[ContentCategory] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Process and chunk a file.
        
        Args:
            file_path: Path to the file to process
            document_id: Optional document ID, generated if not provided
            category: Optional content category, estimated if not provided
            metadata: Additional metadata to include
            
        Returns:
            List of chunks with metadata
            
        Raises:
            ChunkingError: If file processing or chunking fails
        """
        try:
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())
            
            self.log_event(
                "Starting file chunking",
                file_path=file_path,
                document_id=document_id
            )
            
            # Process the file
            content, file_metadata = self.processor.process_file(file_path)
            
            # Use estimated category if not provided
            if category is None:
                category = file_metadata.get('estimated_category', ContentCategory.WELLNESS)
            
            # Merge metadata
            combined_metadata = {**(metadata or {}), **file_metadata}
            
            # Chunk the content
            chunks = self.chunker.chunk_document(
                content=content,
                document_id=document_id,
                source=file_path,
                category=category,
                metadata=combined_metadata
            )
            
            # Validate chunks
            validated_chunks = self._validate_chunks(chunks)
            
            self.log_event(
                "File chunking completed",
                file_path=file_path,
                document_id=document_id,
                chunks_created=len(validated_chunks)
            )
            
            return validated_chunks
            
        except Exception as e:
            self.log_error(e, {"file_path": file_path, "document_id": document_id})
            raise ChunkingError(f"Failed to chunk file {file_path}: {str(e)}")
    
    def chunk_text(
        self,
        content: str,
        source: str,
        document_id: Optional[str] = None,
        category: Optional[ContentCategory] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Process and chunk raw text content.
        
        Args:
            content: Raw text content to chunk
            source: Source identifier for the content
            document_id: Optional document ID, generated if not provided
            category: Optional content category, estimated if not provided
            metadata: Additional metadata to include
            
        Returns:
            List of chunks with metadata
            
        Raises:
            ChunkingError: If text processing or chunking fails
        """
        try:
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())
            
            self.log_event(
                "Starting text chunking",
                source=source,
                document_id=document_id,
                content_length=len(content)
            )
            
            # Process the text content
            processed_content, text_metadata = self.processor.process_text_content(
                content, source
            )
            
            # Use estimated category if not provided
            if category is None:
                category = text_metadata.get('estimated_category', ContentCategory.WELLNESS)
            
            # Merge metadata
            combined_metadata = {**(metadata or {}), **text_metadata}
            
            # Chunk the content
            chunks = self.chunker.chunk_document(
                content=processed_content,
                document_id=document_id,
                source=source,
                category=category,
                metadata=combined_metadata
            )
            
            # Validate chunks
            validated_chunks = self._validate_chunks(chunks)
            
            self.log_event(
                "Text chunking completed",
                source=source,
                document_id=document_id,
                chunks_created=len(validated_chunks)
            )
            
            return validated_chunks
            
        except Exception as e:
            self.log_error(e, {"source": source, "document_id": document_id})
            raise ChunkingError(f"Failed to chunk text from {source}: {str(e)}")
    
    def chunk_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict[str, List[Chunk]]:
        """
        Process and chunk multiple items in batch.
        
        Args:
            items: List of items to process, each containing:
                - 'type': 'file' or 'text'
                - 'content' or 'file_path': Content or file path
                - 'source': Source identifier
                - 'document_id': Optional document ID
                - 'category': Optional content category
                - 'metadata': Optional additional metadata
                
        Returns:
            Dictionary mapping document IDs to their chunks
            
        Raises:
            ChunkingError: If batch processing fails
        """
        try:
            self.log_event(
                "Starting batch chunking",
                batch_size=len(items)
            )
            
            results = {}
            errors = []
            
            for i, item in enumerate(items):
                try:
                    item_type = item.get('type')
                    document_id = item.get('document_id', str(uuid.uuid4()))
                    
                    if item_type == 'file':
                        chunks = self.chunk_file(
                            file_path=item['file_path'],
                            document_id=document_id,
                            category=item.get('category'),
                            metadata=item.get('metadata')
                        )
                    elif item_type == 'text':
                        chunks = self.chunk_text(
                            content=item['content'],
                            source=item['source'],
                            document_id=document_id,
                            category=item.get('category'),
                            metadata=item.get('metadata')
                        )
                    else:
                        raise ChunkingError(f"Invalid item type: {item_type}")
                    
                    results[document_id] = chunks
                    
                except Exception as e:
                    error_info = {
                        'index': i,
                        'item': item,
                        'error': str(e)
                    }
                    errors.append(error_info)
                    self.logger.error(f"Failed to process batch item {i}: {e}")
            
            self.log_event(
                "Batch chunking completed",
                total_items=len(items),
                successful_items=len(results),
                failed_items=len(errors)
            )
            
            if errors:
                self.logger.warning(f"Batch processing completed with {len(errors)} errors")
            
            return results
            
        except Exception as e:
            self.log_error(e, {"batch_size": len(items)})
            raise ChunkingError(f"Failed to process batch: {str(e)}")
    
    def _validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Validate and filter chunks based on quality criteria.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of validated chunks
        """
        validated_chunks = []
        
        for chunk in chunks:
            # Check minimum content length
            if len(chunk.content.strip()) < 10:
                self.logger.debug(f"Skipping chunk {chunk.id}: too short")
                continue
            
            # Check for meaningful content (not just whitespace/punctuation)
            if not re.search(r'[a-zA-Z]{3,}', chunk.content):
                self.logger.debug(f"Skipping chunk {chunk.id}: no meaningful content")
                continue
            
            # Check token count - be lenient for small documents
            # Only filter out chunks that are extremely small (< 5 tokens)
            # The min_chunk_size is more of a guideline for optimal chunking
            if chunk.metadata.tokens < 5:
                self.logger.debug(f"Skipping chunk {chunk.id}: too few tokens ({chunk.metadata.tokens})")
                continue
            
            validated_chunks.append(chunk)
        
        return validated_chunks
    
    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about a set of chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_tokens': 0,
                'avg_tokens_per_chunk': 0,
                'min_tokens': 0,
                'max_tokens': 0,
                'categories': {}
            }
        
        token_counts = [chunk.metadata.tokens for chunk in chunks]
        categories = {}
        
        for chunk in chunks:
            category = chunk.metadata.category.value
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'categories': categories
        }
