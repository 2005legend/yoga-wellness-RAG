"""Semantic document chunker implementation."""

import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import tiktoken

from src.core.logging import LoggerMixin
from src.core.exceptions import ChunkingError
from src.models.schemas import Chunk, ChunkMetadata, ContentCategory
from src.services.chunking.base import DocumentChunker, ChunkingConfig


class SemanticChunker(DocumentChunker, LoggerMixin):
    """
    Semantic document chunker that preserves meaning and structure.
    
    This chunker attempts to create semantically coherent chunks by:
    1. Preserving sentence and paragraph boundaries
    2. Maintaining optimal token counts (256-512 tokens)
    3. Adding overlap between chunks for context preservation
    4. Extracting and preserving metadata
    """
    
    def __init__(self, config: ChunkingConfig, encoding_name: str = "cl100k_base") -> None:
        super().__init__(config)
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise ChunkingError(f"Failed to initialize tokenizer: {str(e)}")
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_separator = re.compile(r'\n\s*\n')
    
    def chunk_document(
        self, 
        content: str, 
        document_id: str,
        source: str,
        category: ContentCategory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk a document using semantic boundaries.
        
        Args:
            content: The document content to chunk
            document_id: Unique identifier for the document
            source: Source of the document
            category: Content category
            metadata: Additional metadata
            
        Returns:
            List of semantically coherent chunks
        """
        try:
            self.log_event(
                "Starting document chunking",
                document_id=document_id,
                content_length=len(content),
                category=category.value
            )
            
            # Clean and preprocess content
            cleaned_content = self._preprocess_content(content)
            
            # Split into paragraphs first
            paragraphs = self._split_into_paragraphs(cleaned_content)
            
            # Create chunks from paragraphs
            chunks = self._create_chunks_from_paragraphs(
                paragraphs, document_id, source, category, metadata
            )
            
            self.log_event(
                "Document chunking completed",
                document_id=document_id,
                chunks_created=len(chunks),
                avg_chunk_size=sum(c.metadata.tokens for c in chunks) / len(chunks) if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            self.log_error(e, {"document_id": document_id, "source": source})
            raise ChunkingError(f"Failed to chunk document {document_id}: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text using tiktoken.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated token count
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            # Fallback to rough estimation if tokenizer fails
            self.logger.warning(f"Tokenizer failed, using fallback estimation: {e}")
            return len(text.split()) * 1.3  # Rough approximation
    
    def _preprocess_content(self, content: str) -> str:
        """
        Clean and preprocess document content.
        
        Args:
            content: Raw document content
            
        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Normalize line breaks
        content = re.sub(r'\r\n|\r', '\n', content)
        
        # Remove excessive newlines but preserve paragraph structure
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """
        Split content into paragraphs.
        
        Args:
            content: Preprocessed content
            
        Returns:
            List of paragraphs
        """
        paragraphs = self.paragraph_separator.split(content)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _create_chunks_from_paragraphs(
        self,
        paragraphs: List[str],
        document_id: str,
        source: str,
        category: ContentCategory,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Create chunks from paragraphs, respecting token limits.
        
        Args:
            paragraphs: List of paragraphs
            document_id: Document identifier
            source: Document source
            category: Content category
            metadata: Additional metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.estimate_tokens(paragraph)
            
            # If paragraph is too large, split it further
            if paragraph_tokens > self.config.max_chunk_size:
                # First, add current chunk if it has content
                if current_chunk_text:
                    chunk = self._create_chunk(
                        current_chunk_text,
                        chunk_index,
                        document_id,
                        source,
                        category,
                        metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_text = ""
                    current_chunk_tokens = 0
                
                # Split large paragraph into sentences
                sentence_chunks = self._split_large_paragraph(
                    paragraph, chunk_index, document_id, source, category, metadata
                )
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                
            else:
                # Check if adding this paragraph would exceed chunk size
                if (current_chunk_tokens + paragraph_tokens > self.config.chunk_size and 
                    current_chunk_text):
                    
                    # Create chunk with current content
                    chunk = self._create_chunk(
                        current_chunk_text,
                        chunk_index,
                        document_id,
                        source,
                        category,
                        metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap if configured
                    if self.config.chunk_overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk_text)
                        current_chunk_text = overlap_text + "\n\n" + paragraph
                        current_chunk_tokens = (self.estimate_tokens(overlap_text) + 
                                              paragraph_tokens)
                    else:
                        current_chunk_text = paragraph
                        current_chunk_tokens = paragraph_tokens
                else:
                    # Add paragraph to current chunk
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + paragraph
                    else:
                        current_chunk_text = paragraph
                    current_chunk_tokens += paragraph_tokens
        
        # Add final chunk if it has content
        if current_chunk_text and current_chunk_tokens >= self.config.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk_text,
                chunk_index,
                document_id,
                source,
                category,
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        start_index: int,
        document_id: str,
        source: str,
        category: ContentCategory,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Split a large paragraph into sentence-based chunks.
        
        Args:
            paragraph: Large paragraph to split
            start_index: Starting chunk index
            document_id: Document identifier
            source: Document source
            category: Content category
            metadata: Additional metadata
            
        Returns:
            List of chunks from the paragraph
        """
        sentences = self._split_into_sentences(paragraph)
        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_index = start_index
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            if (current_chunk_tokens + sentence_tokens > self.config.chunk_size and 
                current_chunk_text):
                
                # Create chunk with current sentences
                chunk = self._create_chunk(
                    current_chunk_text,
                    chunk_index,
                    document_id,
                    source,
                    category,
                    metadata
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk_text)
                    current_chunk_text = overlap_text + " " + sentence
                    current_chunk_tokens = (self.estimate_tokens(overlap_text) + 
                                          sentence_tokens)
                else:
                    current_chunk_text = sentence
                    current_chunk_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + sentence
                else:
                    current_chunk_text = sentence
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                current_chunk_text,
                chunk_index,
                document_id,
                source,
                category,
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of a chunk.
        
        Args:
            text: Source text
            
        Returns:
            Overlap text
        """
        words = text.split()
        overlap_words = words[-self.config.chunk_overlap:] if len(words) > self.config.chunk_overlap else words
        return " ".join(overlap_words)
    
    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        document_id: str,
        source: str,
        category: ContentCategory,
        metadata: Optional[Dict[str, Any]]
    ) -> Chunk:
        """
        Create a chunk object with metadata.
        
        Args:
            content: Chunk content
            chunk_index: Index of the chunk in the document
            document_id: Document identifier
            source: Document source
            category: Content category
            metadata: Additional metadata
            
        Returns:
            Chunk object
        """
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        tokens = self.estimate_tokens(content)
        
        chunk_metadata = ChunkMetadata(
            document_id=document_id,
            chunk_index=chunk_index,
            source=source,
            category=category,
            tokens=tokens,
            created_at=datetime.utcnow()
        )
        
        return Chunk(
            id=chunk_id,
            content=content.strip(),
            metadata=chunk_metadata
        )