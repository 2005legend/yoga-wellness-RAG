"""Document processing utilities for various file formats."""

import io
import re
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from src.core.logging import LoggerMixin
from src.core.exceptions import ChunkingError
from src.models.schemas import ContentCategory


class DocumentProcessor(LoggerMixin):
    """
    Document processor for various file formats.
    
    Supports:
    - Plain text (.txt)
    - Markdown (.md)
    - PDF (.pdf) - requires PyPDF2
    - HTML (.html) - requires BeautifulSoup4
    """
    
    def __init__(self) -> None:
        self.supported_extensions = {
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.html': self._process_html,
            '.htm': self._process_html,
        }
        
        if PDF_AVAILABLE:
            self.supported_extensions['.pdf'] = self._process_pdf
    
    def process_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a file and extract text content with metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (content, metadata)
            
        Raises:
            ChunkingError: If file processing fails
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise ChunkingError(f"File not found: {file_path}")
            
            extension = path.suffix.lower()
            
            if extension not in self.supported_extensions:
                raise ChunkingError(f"Unsupported file format: {extension}")
            
            self.log_event(
                "Processing file",
                file_path=file_path,
                extension=extension,
                file_size=path.stat().st_size
            )
            
            processor = self.supported_extensions[extension]
            content, metadata = processor(path)
            
            # Add common metadata
            metadata.update({
                'file_path': str(path),
                'file_name': path.name,
                'file_extension': extension,
                'file_size': path.stat().st_size,
                'estimated_category': self._estimate_category(content, path.name)
            })
            
            self.log_event(
                "File processing completed",
                file_path=file_path,
                content_length=len(content),
                category=metadata['estimated_category']
            )
            
            return content, metadata
            
        except Exception as e:
            self.log_error(e, {"file_path": file_path})
            raise ChunkingError(f"Failed to process file {file_path}: {str(e)}")
    
    def process_text_content(
        self, 
        content: str, 
        source_name: str = "text_input"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process raw text content.
        
        Args:
            content: Raw text content
            source_name: Name/identifier for the content source
            
        Returns:
            Tuple of (processed_content, metadata)
        """
        try:
            # Basic text cleaning
            processed_content = self._clean_text(content)
            
            metadata = {
                'source_name': source_name,
                'content_length': len(processed_content),
                'estimated_category': self._estimate_category(processed_content, source_name)
            }
            
            return processed_content, metadata
            
        except Exception as e:
            self.log_error(e, {"source_name": source_name})
            raise ChunkingError(f"Failed to process text content: {str(e)}")
    
    def _process_text(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process plain text file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self._clean_text(content), {'format': 'text'}
    
    def _process_markdown(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process Markdown file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Basic markdown processing - remove common markdown syntax
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)  # Headers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Italic
        content = re.sub(r'`(.*?)`', r'\1', content)  # Inline code
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Code blocks
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Links
        
        return self._clean_text(content), {'format': 'markdown'}
    
    def _process_html(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process HTML file."""
        if not BS4_AVAILABLE:
            raise ChunkingError("BeautifulSoup4 is required for HTML processing")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '') if meta_description else ""
        
        metadata = {
            'format': 'html',
            'title': title_text,
            'description': description
        }
        
        return self._clean_text(text), metadata
    
    def _process_pdf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process PDF file."""
        if not PDF_AVAILABLE:
            raise ChunkingError("PyPDF2 is required for PDF processing")
        
        text_content = []
        metadata = {'format': 'pdf', 'pages': 0}
        
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            metadata['pages'] = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
        
        content = '\n\n'.join(text_content)
        return self._clean_text(content), metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remove excessive newlines but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        
        return text.strip()
    
    def _estimate_category(self, content: str, filename: str) -> ContentCategory:
        """
        Estimate content category based on content and filename.
        
        Args:
            content: Text content
            filename: Source filename
            
        Returns:
            Estimated content category
        """
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Keyword-based category estimation
        yoga_keywords = ['yoga', 'asana', 'pranayama', 'vinyasa', 'hatha', 'pose', 'posture']
        meditation_keywords = ['meditation', 'mindfulness', 'breathing', 'awareness', 'zen']
        nutrition_keywords = ['nutrition', 'diet', 'food', 'eating', 'vitamin', 'mineral']
        exercise_keywords = ['exercise', 'workout', 'fitness', 'training', 'strength']
        
        # Check filename first
        for keyword in yoga_keywords:
            if keyword in filename_lower:
                return ContentCategory.YOGA
        
        for keyword in meditation_keywords:
            if keyword in filename_lower:
                return ContentCategory.MEDITATION
        
        for keyword in nutrition_keywords:
            if keyword in filename_lower:
                return ContentCategory.NUTRITION
        
        for keyword in exercise_keywords:
            if keyword in filename_lower:
                return ContentCategory.EXERCISE
        
        # Check content
        yoga_count = sum(1 for keyword in yoga_keywords if keyword in content_lower)
        meditation_count = sum(1 for keyword in meditation_keywords if keyword in content_lower)
        nutrition_count = sum(1 for keyword in nutrition_keywords if keyword in content_lower)
        exercise_count = sum(1 for keyword in exercise_keywords if keyword in content_lower)
        
        counts = {
            ContentCategory.YOGA: yoga_count,
            ContentCategory.MEDITATION: meditation_count,
            ContentCategory.NUTRITION: nutrition_count,
            ContentCategory.EXERCISE: exercise_count
        }
        
        # Return category with highest count, default to WELLNESS
        max_category = max(counts, key=counts.get)
        return max_category if counts[max_category] > 0 else ContentCategory.WELLNESS