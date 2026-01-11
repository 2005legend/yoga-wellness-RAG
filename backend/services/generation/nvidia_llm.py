"""
NVIDIA LLM service for response generation.
"""
from typing import List, Dict, Any, Optional
import aiohttp
from ...core.logging import get_logger
from ...core.exceptions import ResponseGenerationError
from ...config import settings

logger = get_logger(__name__)


class NvidiaLLMService:
    """NVIDIA LLM service using NIM API."""
    
    def __init__(self, api_key: str, model_name: str, api_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        if not self.api_key:
            raise ResponseGenerationError("NVIDIA LLM API key is required")
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        logger.info(f"NVIDIA LLM service initialized with model: {self.model_name}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Generate a response using NVIDIA LLM API."""
        if not self.session:
            await self.initialize()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with self.session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"NVIDIA LLM API error {response.status}: {error_text}")
                    raise ResponseGenerationError(f"NVIDIA LLM API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Handle response format
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content", "")
                    return content
                elif "content" in data:
                    return data["content"]
                else:
                    logger.error(f"Unexpected response format: {data}")
                    raise ResponseGenerationError("Unexpected response format from NVIDIA LLM API")
                    
        except aiohttp.ClientError as e:
            logger.error(f"NVIDIA LLM API request failed: {e}")
            raise ResponseGenerationError(f"NVIDIA LLM API request failed: {e}")
        except Exception as e:
            logger.error(f"NVIDIA LLM generation failed: {e}")
            raise ResponseGenerationError(f"NVIDIA LLM generation failed: {e}")
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()

