import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from backend.api.main import app
from backend.api.dependencies import (
    get_retrieval_engine,
    get_response_generator,
    get_safety_filter,
    get_logger_service
)
from backend.models.schemas import (
    QueryRequest,
    SafetyAssessment,
    RiskLevel,
    GeneratedResponse,
    RetrievalResult,
    Chunk,
    ChunkMetadata,
    ContentCategory,
    SourceCitation
)

# Create TestClient
client = TestClient(app)

# Mocks
mock_retrieval_engine = AsyncMock()
mock_response_generator = AsyncMock()
mock_safety_filter = AsyncMock()
mock_logger_service = MagicMock()  # Logger methods are tailored for background tasks

# Override dependencies
app.dependency_overrides[get_retrieval_engine] = lambda: mock_retrieval_engine
app.dependency_overrides[get_response_generator] = lambda: mock_response_generator
app.dependency_overrides[get_safety_filter] = lambda: mock_safety_filter
app.dependency_overrides[get_logger_service] = lambda: mock_logger_service

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks before each test."""
    mock_retrieval_engine.reset_mock()
    mock_response_generator.reset_mock()
    mock_safety_filter.reset_mock()
    mock_logger_service.reset_mock()
    
    # Configure default mock behaviors
    mock_retrieval_engine.initialize.return_value = None
    mock_retrieval_engine.retrieve_relevant_chunks.return_value = []
    
    mock_response_generator.initialize.return_value = None
    
    # Default safety assessment: Safe
    mock_safety_filter.evaluate_query.return_value = SafetyAssessment(
        flags=[],
        risk_level=RiskLevel.LOW,
        allow_response=True,
        required_disclaimers=[]
    )
    
    # Default logging behaviors (mocking background tasks adds complexity, but the service calls are sync/async)
    # BackgroundTasks handling usually works with TestClient, but we might need to mock the service methods to be async if they are awaited?
    # In routes.py: background_tasks.add_task(logger_service.log_safety_incident, incident)
    # The logger service methods are likely synchronous or asynchronous. Let's assume they are handled by background tasks.

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_question_valid():
    # Setup Mocks
    query = "What is yoga?"
    
    # Mock Retrieval
    mock_chunk = Chunk(
        id="test_chunk_1",
        content="Yoga is a practice.",
        embedding=[0.1]*384,
        metadata=ChunkMetadata(
            document_id="doc_1",
            chunk_index=0,
            tokens=5,
            source="test_doc",
            category=ContentCategory.WELLNESS
        )
    )
    mock_retrieval_result = RetrievalResult(
        chunk=mock_chunk,
        similarity_score=0.9,
        relevance_rank=1
    )
    mock_retrieval_engine.retrieve_relevant_chunks.return_value = [mock_retrieval_result]
    
    # Mock Generation
    mock_response = GeneratedResponse(
        content="Yoga is an ancient practice connecting mind and body.",
        sources=[SourceCitation(source="test_doc", chunk_id="test_chunk_1", relevance_score=0.9)],
        confidence=0.95,
        safety_notices=[]
    )
    mock_response_generator.generate_response.return_value = mock_response

    # Make Request
    response = client.post(
        "/api/v1/ask",
        json={"query": query, "user_id": "test_user"}
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == query
    assert data["response"]["content"] == mock_response.content
    assert len(data["retrieval_results"]) == 1
    assert data["retrieval_results"][0]["chunk"]["id"] == "test_chunk_1"
    
    # Verify mock calls
    mock_safety_filter.evaluate_query.assert_called_once()
    mock_retrieval_engine.retrieve_relevant_chunks.assert_called_once()
    mock_response_generator.generate_response.assert_called_once()

def test_ask_question_safety_blocked():
    # Setup Mock Safety Filter to BLOCK
    mock_safety_filter.evaluate_query.return_value = SafetyAssessment(
        flags=[], # flags usually populated if blocked, but for test brevity empty list handled by code defaults?
        # Actually code says: incident_type=safety_assessment.flags[0].type if safety_assessment.flags else SafetyFlagType.MEDICAL_ADVICE
        risk_level=RiskLevel.HIGH,
        allow_response=False,
        required_disclaimers=["Consult a doctor."]
    )
    
    query = "Unsafe medical question"
    response = client.post(
        "/api/v1/ask",
        json={"query": query}
    )
    
    assert response.status_code == 200 # It returns 200 with a blocked message
    data = response.json()
    assert "I cannot answer this query" in data["response"]["content"]
    assert "Consult a doctor." in data["response"]["content"]
    assert data["safety_assessment"]["allow_response"] is False
    
    # Verify retrieval/generation NOT called
    mock_retrieval_engine.retrieve_relevant_chunks.assert_not_called()
    mock_response_generator.generate_response.assert_not_called()

def test_feedback_submission():
    response = client.post(
        "/api/v1/feedback",
        params={"query_id": "test_id", "feedback": "Great answer!"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "feedback_received"

