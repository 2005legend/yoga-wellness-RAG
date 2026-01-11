"""
Redis cache implementation for storing operation results.
"""
import json
import redis.asyncio as redis
from typing import Optional, Any, Callable, TypeVar
import functools
import hashlib
from backend.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

class RedisCache:
    """
    Redis cache wrapper.
    """
    def __init__(self, url: str = settings.redis_url):
        self.url = url
        self.client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize Redis connection."""
        if self.client is None:
            try:
                self.client = redis.from_url(
                    self.url, 
                    encoding="utf-8", 
                    decode_responses=True
                )
                await self.client.ping()
                logger.debug(f"Cache connected to Redis at {self.url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for caching: {e}")
                self.client = None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        try:
            val = await self.client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: int = settings.cache_ttl):
        """Set value in cache."""
        if not self.client:
            return
        try:
            await self.client.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

# Global instance
_cache = RedisCache()

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a unique cache key based on function arguments."""
    key_str = f"{prefix}:{str(args)}:{str(kwargs)}"
    return hashlib.md5(key_str.encode()).hexdigest()

def cache_result(ttl: int = settings.cache_ttl, prefix: str = ""):
    """
    Decorator to cache async function results in Redis.
    """
    def decorator(func: Callable[..., Any]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to populate cache client if needed (lazy load inside operations usually better, but here we check first)
            if _cache.client is None and settings.redis_url != "redis://mock":
                await _cache.initialize()
            
            # Generate key
            key_prefix = prefix or func.__name__
            # Skip 'self' in args if method (heuristic)
            # This is simple; for production might need smarter arg handling
            cache_key = generate_cache_key(key_prefix, *args, **kwargs)
            
            # Check cache
            cached_val = await _cache.get(cache_key)
            if cached_val is not None:
                logger.debug(f"Cache hit for {key_prefix}")
                return cached_val
            
            # Execute
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                # Note: Result must be JSON serializable. Pydantic models need .model_dump() or similar before this if not handled.
                # If result is a Pydantic model, this basic JSON dump might fail unless we convert it.
                # Assuming this specific decorator handles dicts or primitives.
                await _cache.set(cache_key, result, ttl)
                
            return result
        return wrapper
    return decorator

async def get_cache_service():
    """Dependency for valid cache service."""
    if _cache.client is None and settings.redis_url != "redis://mock":
        await _cache.initialize()
    return _cache

