"""
Unit tests for API routes.
"""
import pytest
import sys
import importlib
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from backend.core.rate_limiter import get_rate_limiter
from backend.models.schemas import (
    SafetyAssessment,
    GeneratedResponse,
    RiskLevel,
    SafetyFlag,
    SafetyFlagType
)

# Fixture to safely setup app with mocked heavy dependencies
@pytest.fixture
def client():
    # Context manager to mock modules only within this fixture's setup
    with patch.dict(sys.modules, {
        "src.services.embeddings.sentence_transformer": MagicMock(),
        "sentence_transformers": MagicMock(),
        "src.services.retrieval.vector_db": MagicMock(),
    }):
        # Import (or reload) app inside the patched environment
        if "src.api.main" in sys.modules:
            importlib.reload(sys.modules["src.api.main"])
        else:
            import backend.api.main
            
        from backend.api.main import app
        from backend.api.dependencies import (
            get_retrieval_engine,
            get_response_generator,
            get_safety_filter,
            get_logger_service
        )
        
        # Override dependencies
        app.dependency_overrides[get_retrieval_engine] = lambda: AsyncMock(
            retrieve_relevant_chunks=AsyncMock(return_value=[]),
            initialize=AsyncMock()
        )
        app.dependency_overrides[get_response_generator] = lambda: AsyncMock(
            generate_response=AsyncMock(return_value=GeneratedResponse(
                content="Test response content", sources=[], confidence=0.9, safety_notices=[]
            )),
            initialize=AsyncMock()
        )
        app.dependency_overrides[get_safety_filter] = lambda: AsyncMock(
            evaluate_query=AsyncMock(return_value=SafetyAssessment(
                flags=[], risk_level=RiskLevel.LOW, allow_response=True, required_disclaimers=[]
            ))
        )
        app.dependency_overrides[get_logger_service] = lambda: AsyncMock(
            log_interaction=AsyncMock(), log_safety_incident=AsyncMock()
        )
        app.dependency_overrides[get_rate_limiter] = lambda: AsyncMock(
            is_rate_limited=AsyncMock(return_value=False)
        )
        
        test_client = TestClient(app)
        yield test_client
        
        # Cleanup overrides
        app.dependency_overrides = {}
        
    # Force reload/cleanup of affected modules to ensure mocks don't persist in other tests
    modules_to_clear = [
        "src.api.main",
        "src.api.dependencies",
        "src.services.embeddings.service",
        "src.services.embeddings.sentence_transformer",
        "src.services.retrieval.vector_db",
        "src.services.retrieval.engine",
        "src.services.generation.service",
        "src.services.safety.filter",
        "src.services.logging.mongo_logger"
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
        
    # Also clear any re-imported generic modules if they were mocked at sys.modules level
    # (The patch.dict handles usage during the block, but if we deleted them from sys.modules during execution to force reload, we need to be careful. 
    # Here we relied on patch.dict resetting them, but the consumers kept the references. 
    # Deleting the consumers from sys.modules forces them to re-import the REAL modules next time they are needed.)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_question_success(client):
    """Test successful query processing."""
    payload = {
        "query": "What is yoga?",
        "user_id": "test_user",
        "session_id": "test_session"
    }
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is yoga?"
    assert data["response"]["content"] == "Test response content"
    assert "retrieval_results" in data

def test_ask_question_safety_blocked(client):
    """Test safety blocking mechanism."""
    # We need to access the dependencies to override them for this specific test
    from backend.api.dependencies import get_safety_filter
    
    # Define blocking mock
    mock_safety_filter = AsyncMock()
    mock_safety_filter.evaluate_query.return_value = SafetyAssessment(
        flags=[SafetyFlag(type=SafetyFlagType.MEDICAL_ADVICE, severity=0.9, description="unsafe", mitigation_action="block")],
        risk_level=RiskLevel.HIGH,
        allow_response=False,
        required_disclaimers=["Unsafe query"]
    )
    
    # Apply override to the app instance used by the client
    client.app.dependency_overrides[get_safety_filter] = lambda: mock_safety_filter
    
    payload = {"query": "Unsafe medical advice please"}
    response = client.post("/api/v1/ask", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "I cannot answer this query" in data["response"]["content"]
    assert not data["safety_assessment"]["allow_response"]

def test_rate_limiting_trigger(client):
    """Test that rate limiter triggers 429."""
    from backend.core.rate_limiter import get_rate_limiter
    from fastapi import HTTPException, Request
    
    async def raise_limit(request: Request):
        raise HTTPException(status_code=429, detail="Rate limited")
        
    client.app.dependency_overrides[get_rate_limiter] = raise_limit
    
    payload = {"query": "spam"}
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 429
    assert "Rate limited" in response.json()["detail"]

def test_feedback_submission(client):
    """Test feedback endpoint."""
    response = client.post(
        "/api/v1/feedback",
        params={"query_id": "123", "feedback": "good"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "feedback_received"

