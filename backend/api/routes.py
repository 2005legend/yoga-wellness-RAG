"""
API routes for the RAG application.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Optional
from datetime import datetime
import uuid

from backend.core.logging import get_logger
logger = get_logger(__name__)

from backend.models.schemas import (
    QueryRequest, 
    QueryResponse, 
    UserInteractionLog, 
    SafetyIncident, 
    RiskLevel,
    SafetyFlagType
)
from backend.services.retrieval.engine import RetrievalEngine
from backend.services.generation.service import ResponseGenerator
from backend.services.safety.filter import SafetyFilter
from backend.services.logging.mongo_logger import MongoLogger
from .dependencies import (
    get_retrieval_engine, 
    get_response_generator, 
    get_safety_filter, 
    get_logger_service
)
from backend.core.rate_limiter import RateLimiter, get_rate_limiter

router = APIRouter()

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
    response_generator: ResponseGenerator = Depends(get_response_generator),
    safety_filter: SafetyFilter = Depends(get_safety_filter),
    logger_service: MongoLogger = Depends(get_logger_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """
    Process a user query through the RAG pipeline.
    """
    query_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    # 1. Safety Check (Query)
    try:
        safety_assessment = await safety_filter.evaluate_query(request.query)
    except Exception as e:
        logger.error(f"Safety filter error: {e}", exc_info=True)
        from backend.models.schemas import SafetyAssessment, RiskLevel
        safety_assessment = SafetyAssessment(
            flags=[],
            risk_level=RiskLevel.LOW,
            allow_response=True,
            required_disclaimers=[]
        )
    
    session_id = request.session_id or str(uuid.uuid4())
    
    if not safety_assessment.allow_response:
        # Log critical safety incident
        incident = SafetyIncident(
            id=query_id,
            session_id=session_id,
            incident_type=safety_assessment.flags[0].type if safety_assessment.flags else SafetyFlagType.MEDICAL_ADVICE,
            severity=safety_assessment.risk_level,
            query=request.query,
            flags=safety_assessment.flags
        )
        background_tasks.add_task(logger_service.log_safety_incident, incident)
        
        from backend.models.schemas import GeneratedResponse, SourceCitation
        blocked_response = GeneratedResponse(
            content="I cannot answer this query due to safety guidelines. " + 
                   " ".join(safety_assessment.required_disclaimers),
            sources=[],
            confidence=0.0,
            safety_notices=safety_assessment.required_disclaimers
        )
        
        return QueryResponse(
            query=request.query,
            response=blocked_response,
            retrieval_results=[],
            safety_assessment=safety_assessment,
            processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            session_id=session_id
        )

    # 2. Retrieval
    retrieved_chunks = []
    try:
        # Ensure retrieval engine is initialized
        try:
            await retrieval_engine.initialize()
        except Exception as init_error:
            # Check if it's already initialized or a different error
            if "already" not in str(init_error).lower():
                logger.warning(f"Retrieval engine initialization: {init_error}")
        
        retrieved_chunks = await retrieval_engine.retrieve_relevant_chunks(
            request.query, 
            max_results=request.max_chunks or 5,
            min_similarity=request.min_similarity or 0.7
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        # Retrieval might yield empty results, proceed to generation (which handles empty context)
        # Don't fail the request, just continue with empty chunks

    # 3. Generation
    try:
        # Ensure response generator is initialized
        try:
            await response_generator.initialize()
        except Exception as init_error:
            # Check if it's already initialized or a different error
            if "already" not in str(init_error).lower() and "not implemented" not in str(init_error).lower():
                logger.warning(f"Response generator initialization: {init_error}")
        
        generated_response = await response_generator.generate_response(
            query=request.query,
            context=retrieved_chunks,
            safety_assessment=safety_assessment
        )
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        from backend.models.schemas import GeneratedResponse, SourceCitation
        generated_response = GeneratedResponse(
            content=f"I apologize, but I encountered an error while generating a response. Please try again. Error: {str(e)[:100]}",
            sources=[],
            confidence=0.0,
            safety_notices=[]
        )

    # 4. Log Interaction
    processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    log = UserInteractionLog(
        query_id=query_id,
        user_id=request.user_id or "anonymous",
        timestamp=start_time,
        query=request.query,
        retrieved_chunks=[r.chunk.id for r in retrieved_chunks],
        response_content=generated_response.content,
        processing_time_ms=processing_time_ms,
        safety_flags=safety_assessment.flags,
        feedback=None
    )
    background_tasks.add_task(logger_service.log_interaction, log)

    return QueryResponse(
        query=request.query,
        response=generated_response,
        retrieval_results=retrieved_chunks,
        safety_assessment=safety_assessment,
        processing_time_ms=processing_time_ms,
        session_id=session_id
    )

@router.post("/feedback")
async def submit_feedback(
    query_id: str,
    feedback: str,
    logger_service: MongoLogger = Depends(get_logger_service)
):
    """
    Submit feedback for a previous query.
    """
    # In a real implementation, we'd update the interaction log with feedback
    # For now, just log it
    logger.info(f"Feedback received for query {query_id}: {feedback}")
    return {"status": "feedback_received", "query_id": query_id}

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

