"""
Response generation service using LLM.
"""
from typing import List, Dict, Any, Optional

from ...core.logging import get_logger
from ...core.exceptions import ResponseGenerationError
from ...models.schemas import (
    GeneratedResponse, 
    RetrievalResult, 
    SourceCitation, 
    SafetyFlag,
    SafetyAssessment
)
from ...config import settings
from .prompts import YOGA_EXPERT_SYSTEM_PROMPT

logger = get_logger(__name__)

class ResponseGenerator:
    """
    Generates responses using retrieved context and LLM.
    """
    
    def __init__(self):
        self.openai_client = None
        self.nvidia_client = None
        
        # Initialize NVIDIA LLM if API key is available
        if settings.use_nvidia_llm:
            try:
                from .nvidia_llm import NvidiaLLMService
                self.nvidia_client = NvidiaLLMService(
                    api_key=settings.nvidia_llm_api_key,
                    model_name=settings.nvidia_llm_model,
                    api_url=settings.nvidia_llm_api_url
                )
                logger.info("NVIDIA LLM service configured")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA LLM: {e}")
        
        # Fallback to OpenAI if available
        if settings.use_openai and not self.nvidia_client:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client configured")
            except ImportError:
                logger.warning("OpenAI client not installed or configured.")
    
    async def initialize(self) -> None:
        """Initialize the LLM service."""
        if self.nvidia_client:
            await self.nvidia_client.initialize()
    
    async def generate_response(
        self,
        query: str,
        context: List[RetrievalResult],
        safety_assessment: Optional[SafetyAssessment] = None
    ) -> GeneratedResponse:
        """
        Generate a response based on query and context.
        """
        try:
            # 1. Format Context
            context_text = "\n\n".join([
                f"Source {i+1} ({result.chunk.metadata.source}):\n{result.chunk.content}"
                for i, result in enumerate(context)
            ])
            
            # 2. Prepare Prompt
            prompt = YOGA_EXPERT_SYSTEM_PROMPT.format(
                context=context_text,
                query=query
            )
            
            # 3. Call LLM (or mock if not available)
            response_content = ""
            confidence = 1.0 # Placeholder
            
            # Try NVIDIA LLM first
            if self.nvidia_client:
                try:
                    messages = [
                        {"role": "system", "content": "You are a helpful yoga assistant with expertise in yoga poses, breathing techniques, and wellness practices."},
                        {"role": "user", "content": prompt}
                    ]
                    response_content = await self.nvidia_client.generate(
                        messages=messages,
                        temperature=settings.openai_temperature,
                        max_tokens=settings.openai_max_tokens
                    )
                    logger.info("Successfully generated response using NVIDIA LLM")
                except Exception as e:
                    logger.error(f"NVIDIA LLM API call failed: {e}")
                    # Fallback to OpenAI or mock
                    if self.openai_client and settings.use_openai:
                        try:
                            completion = await self.openai_client.chat.completions.create(
                                model=settings.openai_model,
                                messages=[
                                    {"role": "system", "content": "You are a helpful yoga assistant."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=settings.openai_temperature,
                                max_tokens=settings.openai_max_tokens
                            )
                            response_content = completion.choices[0].message.content
                            logger.info("Fallback to OpenAI successful")
                        except Exception as openai_error:
                            logger.error(f"OpenAI fallback also failed: {openai_error}")
                            response_content = "I apologize, but I am unable to generate a detailed response at the moment due to a technical issue. Please try again."
                            confidence = 0.0
                    else:
                        response_content = "I apologize, but I am unable to generate a detailed response at the moment due to a technical issue. Please try again."
                        confidence = 0.0
            # Fallback to OpenAI
            elif self.openai_client and settings.use_openai:
                try:
                    completion = await self.openai_client.chat.completions.create(
                        model=settings.openai_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful yoga assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=settings.openai_temperature,
                        max_tokens=settings.openai_max_tokens
                    )
                    response_content = completion.choices[0].message.content
                except Exception as e:
                    logger.error(f"OpenAI API call failed: {e}")
                    response_content = "I apologize, but I am unable to generate a detailed response at the moment due to a technical issue. Please try again."
                    confidence = 0.0
            else:
                # Mock response for when no LLM is configured
                response_content = f"**[MOCK RESPONSE]**\n\nBased on your query '{query}', here is some information from our knowledge base:\n\n"
                if context:
                    response_content += f"{context[0].chunk.content[:200]}...\n\n(Note: LLM is not configured, showing raw context excerpt)"
                else:
                    response_content += "No relevant information found in the knowledge base."
                    confidence = 0.0

            # 4. Format Citations
            citations = []
            for result in context:
                citations.append(SourceCitation(
                    source=result.chunk.metadata.source,
                    chunk_id=result.chunk.id,
                    relevance_score=result.similarity_score
                ))
                
            # 5. Handle Safety Notices
            safety_notices = []
            if safety_assessment and not safety_assessment.allow_response:
                 # If unsafe, we might overwrite content or just append warning
                 # But typically if allow_response is false, this method might not even be called
                 # checking flags just in case
                 if safety_assessment.flags:
                     safety_notices = safety_assessment.required_disclaimers
            
            return GeneratedResponse(
                content=response_content,
                sources=citations,
                confidence=confidence,
                safety_notices=safety_notices
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise ResponseGenerationError(f"Response generation failed: {e}")
