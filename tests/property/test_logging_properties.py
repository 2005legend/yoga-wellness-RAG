"""Unit tests for logging functionality."""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from backend.services.logging.mongo_logger import MongoLogger
from backend.models.schemas import UserInteractionLog

class TestLoggingUnit:
    """Tests for MongoLogger."""

    @pytest.mark.asyncio
    async def test_log_interaction_structure(self):
        """
        Verify log interaction calls insert with correct structure.
        Uses direct attribute mocking to avoid complex patches.
        """
        log = UserInteractionLog(
            query_id="q1",
            user_id="u1",
            timestamp=datetime.utcnow(),
            query="test query",
            retrieved_chunks=["c1"],
            response_content="response",
            processing_time_ms=100.0,
            safety_flags=[],
            feedback=None
        )
        
        # Instantiate service (relies on motor being available or caught exception)
        # We verified motor is available via debug_tests.py
        logger_service = MongoLogger()
        
        # Manually mock the collection to capture the insert call
        mock_collection = AsyncMock()
        logger_service.logs_collection = mock_collection
        
        await logger_service.log_interaction(log)
        
        # Verification
        mock_collection.insert_one.assert_called_once()
        call_arg = mock_collection.insert_one.call_args[0][0]
        
        assert call_arg['query_id'] == "q1"
        assert call_arg['query'] == "test query"

