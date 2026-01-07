"""Custom exceptions for the wellness RAG application."""

from typing import Any, Dict, Optional


class WellnessRAGException(Exception):
    """Base exception for wellness RAG application."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class ConfigurationError(WellnessRAGException):
    """Raised when there's a configuration issue."""
    pass


class DatabaseError(WellnessRAGException):
    """Raised when there's a database operation error."""
    pass


class VectorDatabaseError(WellnessRAGException):
    """Raised when there's a vector database operation error."""
    pass


class EmbeddingError(WellnessRAGException):
    """Raised when there's an embedding generation error."""
    pass


class RetrievalError(WellnessRAGException):
    """Raised when there's a retrieval operation error."""
    pass


class ResponseGenerationError(WellnessRAGException):
    """Raised when there's a response generation error."""
    pass


class SafetyFilterError(WellnessRAGException):
    """Raised when there's a safety filtering error."""
    pass


class RateLimitError(WellnessRAGException):
    """Raised when rate limit is exceeded."""
    pass


class ValidationError(WellnessRAGException):
    """Raised when input validation fails."""
    pass


class ChunkingError(WellnessRAGException):
    """Raised when document chunking fails."""
    pass