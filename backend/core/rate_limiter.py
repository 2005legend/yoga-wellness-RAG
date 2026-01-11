"""
Rate limiter implementation with Redis backend and token bucket algorithm.
"""
import time
import redis.asyncio as redis
from fastapi import HTTPException, Request, Depends
from typing import Optional, Tuple
from backend.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """
    Rate limiter using Redis with a fixed window counter.
    Falls back to in-memory implementation if Redis is unavailable.
    """
    
    def __init__(
        self, 
        requests_limit: int = settings.rate_limit_requests, 
        window_seconds: int = settings.rate_limit_window
    ):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.redis_client: Optional[redis.Redis] = None
        self._in_memory_store = {}
        
    async def initialize(self):
        """Initialize Redis connection."""
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url, 
                    encoding="utf-8", 
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.debug(f"Rate limiter connected to Redis at {settings.redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for rate limiting: {e}. using in-memory fallback.")
                self.redis_client = None

    async def is_rate_limited(self, key: str) -> bool:
        """
        Check if the key is rate limited.
        Returns True if limited, False otherwise.
        """
        if not self.redis_client:
            # In-memory fallback (simple window reset based on time)
            current_time = time.time()
            if key not in self._in_memory_store:
                self._in_memory_store[key] = {"count": 1, "start_time": current_time}
                return False
            
            data = self._in_memory_store[key]
            if current_time - data["start_time"] > self.window_seconds:
                # Reset window
                self._in_memory_store[key] = {"count": 1, "start_time": current_time}
                return False
            
            if data["count"] >= self.requests_limit:
                return True
            
            data["count"] += 1
            return False

        # Redis implementation
        try:
            # Use a pipeline for atomicity
            pipe = self.redis_client.pipeline()
            now = time.time()
            window_key = f"rate_limit:{key}:{int(now // self.window_seconds)}"
            
            # Increment counter
            pipe.incr(window_key)
            # Set expiry
            pipe.expire(window_key, self.window_seconds + 1)
            
            results = await pipe.execute()
            count = results[0]
            
            return count > self.requests_limit
            
        except Exception as e:
            logger.error(f"Error checking rate limit in Redis: {e}")
            # Fail open if Redis errors
            return False

# Global instance
_rate_limiter = RateLimiter()

async def get_rate_limiter(request: Request):
    """
    Dependency to check rate limits.
    Uses client IP as the key.
    """
    # Ensure initialized (lazy init)
    if _rate_limiter.redis_client is None and settings.redis_url != "redis://mock":
        await _rate_limiter.initialize()
        
    client_ip = request.client.host if request.client else "unknown"
    is_limited = await _rate_limiter.is_rate_limited(client_ip)
    
    if is_limited:
        raise HTTPException(
            status_code=429, 
            detail="Too many requests. Please try again later."
        )
    
    return _rate_limiter

