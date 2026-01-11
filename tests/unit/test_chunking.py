"""Unit tests for the chunking service."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from backend.services.chunking.service import ChunkingService
from backend.services.chunking.base import ChunkingConfig
from backend.services.chunking.semantic_chunker import SemanticChunker
from backend.services.chunking.document_processor import DocumentProcessor
from backend.models.schemas import ContentCategory, Chunk
from backend.core.exceptions import ChunkingError


class TestSemanticChunker:
    """Test cases for SemanticChunker."""
    
    def test_initialization(self):
        """Test chunker initialization."""
        config = ChunkingConfig(chunk_size=256, chunk_overlap=25)
        chunker = SemanticChunker(config)
        
        assert chunker.config.chunk_size == 256
        assert chunker.config.chunk_overlap == 25
        assert chunker.tokenizer is not None
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        config = ChunkingConfig()
        chunker = SemanticChunker(config)
        
        text = "This is a test sentence for token estimation."
        tokens = chunker.estimate_tokens(text)
        
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_chunk_document_simple(self):
        """Test basic document chunking."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = SemanticChunker(config)
        
        content = """
        This is the first paragraph about yoga breathing techniques.
        It contains important information about pranayama practices.
        
        This is the second paragraph about meditation.
        It discusses mindfulness and awareness practices.
        
        This is the third paragraph about wellness.
        It covers general health and wellbeing topics.
        """
        
        chunks = chunker.chunk_document(
            content=content,
            document_id="test_doc",
            source="test_source.txt",
            category=ContentCategory.YOGA
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata.document_id == "test_doc" for chunk in chunks)
        assert all(chunk.metadata.category == ContentCategory.YOGA for chunk in chunks)
    
    def test_chunk_document_large_paragraph(self):
        """Test chunking of large paragraphs."""
        config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
        chunker = SemanticChunker(config)
        
        # Create a large paragraph that should be split
        large_paragraph = " ".join([
            "This is a very long paragraph that should be split into multiple chunks."
        ] * 10)
        
        chunks = chunker.chunk_document(
            content=large_paragraph,
            document_id="test_doc",
            source="test_source.txt",
            category=ContentCategory.WELLNESS
        )
        
        assert len(chunks) > 1
        # Check that chunks have reasonable token counts
        for chunk in chunks:
            assert chunk.metadata.tokens <= config.max_chunk_size
    
    def test_preprocess_content(self):
        """Test content preprocessing."""
        config = ChunkingConfig()
        chunker = SemanticChunker(config)
        
        messy_content = "  This   has    excessive   whitespace.  \n\n\n\n  And multiple newlines.  "
        cleaned = chunker._preprocess_content(messy_content)
        
        assert "   " not in cleaned  # No excessive whitespace
        assert "\n\n\n" not in cleaned  # No excessive newlines
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor()
        
        assert '.txt' in processor.supported_extensions
        assert '.md' in processor.supported_extensions
        assert '.html' in processor.supported_extensions
    
    def test_process_text_content(self):
        """Test processing of raw text content."""
        processor = DocumentProcessor()
        
        content = "This is a test content about yoga practices."
        processed_content, metadata = processor.process_text_content(content, "test_source")
        
        assert processed_content == content
        assert metadata['source_name'] == "test_source"
        assert metadata['content_length'] == len(content)
        assert 'estimated_category' in metadata
    
    @patch("builtins.open", new_callable=mock_open, read_data="Test file content")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.stat")
    def test_process_text_file(self, mock_stat, mock_exists, mock_file):
        """Test processing of text file."""
        mock_stat.return_value.st_size = 100
        
        processor = DocumentProcessor()
        content, metadata = processor.process_file("test.txt")
        
        assert content == "Test file content"
        assert metadata['format'] == 'text'
        assert metadata['file_name'] == 'test.txt'
    
    def test_estimate_category_yoga(self):
        """Test category estimation for yoga content."""
        processor = DocumentProcessor()
        
        content = "This content discusses yoga poses and asana practices."
        category = processor._estimate_category(content, "yoga_guide.txt")
        
        assert category == ContentCategory.YOGA
    
    def test_estimate_category_meditation(self):
        """Test category estimation for meditation content."""
        processor = DocumentProcessor()
        
        content = "This content covers meditation and mindfulness techniques."
        category = processor._estimate_category(content, "meditation_guide.txt")
        
        assert category == ContentCategory.MEDITATION
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        processor = DocumentProcessor()
        
        messy_text = "  This\r\nhas\tmixed   whitespace\n\n\nand newlines.  "
        cleaned = processor._clean_text(messy_text)
        
        assert "\r" not in cleaned
        assert "  " not in cleaned  # No double spaces
        assert cleaned.startswith("This")
        assert cleaned.endswith("newlines.")


class TestChunkingService:
    """Test cases for ChunkingService."""
    
    def test_initialization(self):
        """Test service initialization."""
        service = ChunkingService()
        
        assert service.config is not None
        assert service.chunker is not None
        assert service.processor is not None
    
    def test_initialization_with_config(self):
        """Test service initialization with custom config."""
        config = ChunkingConfig(chunk_size=256, chunk_overlap=25)
        service = ChunkingService(config)
        
        assert service.config.chunk_size == 256
        assert service.config.chunk_overlap == 25
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        service = ChunkingService()
        
        content = """
        This is a comprehensive guide to yoga breathing techniques.
        Pranayama is the practice of breath control in yoga.
        
        There are many different types of breathing exercises.
        Each technique has its own benefits and applications.
        """
        
        chunks = service.chunk_text(
            content=content,
            source="test_guide",
            category=ContentCategory.YOGA
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata.category == ContentCategory.YOGA for chunk in chunks)
        assert all(chunk.metadata.source == "test_guide" for chunk in chunks)
    
    @patch.object(DocumentProcessor, 'process_file')
    def test_chunk_file(self, mock_process_file):
        """Test file chunking functionality."""
        # Mock file processing
        mock_process_file.return_value = (
            "Test file content about yoga practices.",
            {
                'format': 'text',
                'file_name': 'test.txt',
                'estimated_category': ContentCategory.YOGA
            }
        )
        
        service = ChunkingService()
        chunks = service.chunk_file("test.txt")
        
        assert len(chunks) > 0
        mock_process_file.assert_called_once_with("test.txt")
    
    def test_chunk_batch(self):
        """Test batch chunking functionality."""
        service = ChunkingService()
        
        items = [
            {
                'type': 'text',
                'content': 'First document about yoga practices.',
                'source': 'doc1',
                'category': ContentCategory.YOGA
            },
            {
                'type': 'text',
                'content': 'Second document about meditation techniques.',
                'source': 'doc2',
                'category': ContentCategory.MEDITATION
            }
        ]
        
        results = service.chunk_batch(items)
        
        assert len(results) == 2
        assert all(isinstance(chunks, list) for chunks in results.values())
    
    def test_validate_chunks(self):
        """Test chunk validation."""
        service = ChunkingService()
        
        # Create real Chunk objects with different qualities
        from backend.models.schemas import ChunkMetadata
        
        good_chunk = Chunk(
            id="good_chunk",
            content="This is a good chunk with meaningful content.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=0,
                source="test",
                category=ContentCategory.YOGA,
                tokens=50
            )
        )
        
        short_chunk = Chunk(
            id="short_chunk",
            content="Short",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=1,
                source="test",
                category=ContentCategory.YOGA,
                tokens=5
            )
        )
        
        empty_chunk = Chunk(
            id="empty_chunk",
            content="   ",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=2,
                source="test",
                category=ContentCategory.YOGA,
                tokens=1
            )
        )
        
        chunks = [good_chunk, short_chunk, empty_chunk]
        validated = service._validate_chunks(chunks)
        
        assert len(validated) == 1
        assert validated[0] == good_chunk
    
    def test_get_chunking_stats(self):
        """Test chunking statistics generation."""
        service = ChunkingService()
        
        # Create real Chunk objects
        from backend.models.schemas import ChunkMetadata
        
        chunk1 = Chunk(
            id="chunk1",
            content="Test content 1",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=0,
                source="test",
                category=ContentCategory.YOGA,
                tokens=100
            )
        )
        
        chunk2 = Chunk(
            id="chunk2",
            content="Test content 2",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_index=1,
                source="test",
                category=ContentCategory.YOGA,
                tokens=150
            )
        )
        
        chunk3 = Chunk(
            id="chunk3",
            content="Test content 3",
            metadata=ChunkMetadata(
                document_id="doc2",
                chunk_index=0,
                source="test",
                category=ContentCategory.MEDITATION,
                tokens=200
            )
        )
        
        chunks = [chunk1, chunk2, chunk3]
        stats = service.get_chunking_stats(chunks)
        
        assert stats['total_chunks'] == 3
        assert stats['total_tokens'] == 450
        assert stats['avg_tokens_per_chunk'] == 150
        assert stats['min_tokens'] == 100
        assert stats['max_tokens'] == 200
        assert stats['categories']['YOGA'] == 2
        assert stats['categories']['MEDITATION'] == 1
    
    def test_get_chunking_stats_empty(self):
        """Test chunking statistics with empty chunk list."""
        service = ChunkingService()
        
        stats = service.get_chunking_stats([])
        
        assert stats['total_chunks'] == 0
        assert stats['total_tokens'] == 0
        assert stats['avg_tokens_per_chunk'] == 0


class TestChunkingErrors:
    """Test error handling in chunking services."""
    
    def test_chunking_service_invalid_file(self):
        """Test error handling for invalid file."""
        service = ChunkingService()
        
        with pytest.raises(ChunkingError):
            service.chunk_file("nonexistent_file.txt")
    
    def test_chunking_service_invalid_batch_item(self):
        """Test error handling for invalid batch item."""
        service = ChunkingService()
        
        items = [
            {
                'type': 'invalid_type',
                'content': 'Test content',
                'source': 'test'
            }
        ]
        
        results = service.chunk_batch(items)
        assert len(results) == 0  # Should handle error gracefully
