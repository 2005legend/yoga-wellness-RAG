"""Pytest configuration and fixtures for the wellness RAG application."""

import asyncio
import os
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

from src.config import settings


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> Generator[MagicMock, None, None]:
    """Mock settings for testing."""
    mock = MagicMock()
    mock.mongodb_url = "mongodb://localhost:27017"
    mock.mongodb_database = "test_wellness_rag"
    mock.openai_api_key = "test_key"
    mock.debug = True
    mock.safety_enabled = True
    yield mock


@pytest.fixture
async def mock_mongodb() -> AsyncGenerator[AsyncMock, None]:
    """Mock MongoDB client for testing."""
    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_collection = AsyncMock()
    
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    
    yield mock_client


@pytest.fixture
def mock_openai_client() -> Generator[AsyncMock, None, None]:
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock()
    
    # Mock embeddings response
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1536)
    ]
    
    # Mock chat completion response
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    
    yield mock_client


@pytest.fixture
def sample_chunk_data() -> dict:
    """Sample chunk data for testing."""
    return {
        "id": "test_chunk_1",
        "content": "This is a test chunk about yoga breathing techniques.",
        "embedding": [0.1] * 1536,
        "metadata": {
            "document_id": "test_doc_1",
            "chunk_index": 0,
            "source": "test_source.txt",
            "category": "YOGA",
            "tokens": 15
        }
    }


@pytest.fixture
def sample_query_request() -> dict:
    """Sample query request for testing."""
    return {
        "query": "What are the benefits of deep breathing in yoga?",
        "max_chunks": 5,
        "min_similarity": 0.7
    }


@pytest.fixture
def sample_safety_flags() -> list:
    """Sample safety flags for testing."""
    return [
        {
            "type": "MEDICAL_ADVICE",
            "severity": 0.8,
            "description": "Query requests medical advice",
            "mitigation_action": "Add medical disclaimer"
        }
    ]