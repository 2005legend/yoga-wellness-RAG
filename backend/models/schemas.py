"""Pydantic schemas for the wellness RAG application."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ContentCategory(str, Enum):
    """Content categories for knowledge base documents."""
    YOGA = "YOGA"
    WELLNESS = "WELLNESS"
    MEDITATION = "MEDITATION"
    NUTRITION = "NUTRITION"
    EXERCISE = "EXERCISE"


class SafetyFlagType(str, Enum):
    """Types of safety flags."""
    MEDICAL_ADVICE = "MEDICAL_ADVICE"
    EMERGENCY = "EMERGENCY"
    INAPPROPRIATE = "INAPPROPRIATE"
    DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"
    PRESCRIPTION_REQUEST = "PRESCRIPTION_REQUEST"
    TREATMENT_REQUEST = "TREATMENT_REQUEST"


class RiskLevel(str, Enum):
    """Risk levels for safety assessment."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ChunkMetadata(BaseModel):
    """Metadata for a knowledge base chunk."""
    document_id: str
    chunk_index: int
    source: str
    category: ContentCategory
    tokens: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A chunk of knowledge base content."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetyFlag(BaseModel):
    """A safety flag raised during content evaluation."""
    type: SafetyFlagType
    severity: float = Field(ge=0.0, le=1.0)
    description: str
    mitigation_action: str


class SafetyAssessment(BaseModel):
    """Safety assessment result."""
    flags: List[SafetyFlag] = Field(default_factory=list)
    risk_level: RiskLevel
    allow_response: bool
    required_disclaimers: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result from chunk retrieval."""
    chunk: Chunk
    similarity_score: float = Field(ge=0.0, le=1.0)
    relevance_rank: int = Field(ge=1)


class SourceCitation(BaseModel):
    """Source citation for generated responses."""
    source: str
    chunk_id: str
    relevance_score: float


class GeneratedResponse(BaseModel):
    """Generated response from the RAG system."""
    content: str
    sources: List[SourceCitation]
    confidence: float = Field(ge=0.0, le=1.0)
    safety_notices: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    query: str = Field(min_length=1, max_length=1000)
    max_chunks: Optional[int] = Field(default=5, ge=1, le=20)
    min_similarity: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    user_id: Optional[str] = Field(default="anonymous")
    session_id: Optional[str] = None
    
    @validator('query')
    def validate_query(cls, v: str) -> str:
        """Validate and clean the query."""
        return v.strip()


class QueryResponse(BaseModel):
    """Response from the RAG system."""
    query: str
    response: GeneratedResponse
    retrieval_results: List[RetrievalResult]
    safety_assessment: SafetyAssessment
    processing_time_ms: int
    session_id: str


class UserInteractionLog(BaseModel):
    """Log entry for user interactions."""
    query_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query: str
    retrieved_chunks: List[str]
    response_content: str
    processing_time_ms: float
    safety_flags: List[SafetyFlag]
    feedback: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetyIncident(BaseModel):
    """Safety incident log entry."""
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    incident_type: SafetyFlagType
    severity: RiskLevel
    query: str
    flags: List[SafetyFlag]
    resolved: bool = False
    review_required: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class KnowledgeDocument(BaseModel):
    """Knowledge base document."""
    id: str
    title: str
    content: str
    category: ContentCategory
    source: str
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    components: Dict[str, str]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }