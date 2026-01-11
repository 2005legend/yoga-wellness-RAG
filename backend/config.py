"""Configuration management for the wellness RAG application."""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Configuration
    app_name: str = Field(default="wellness-rag-app", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="API auto-reload")
    
    # Database Configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB URL")
    mongodb_database: str = Field(default="wellness_rag", description="MongoDB database name")
    mongodb_collection_logs: str = Field(default="interaction_logs", description="Logs collection")
    mongodb_collection_safety: str = Field(default="safety_incidents", description="Safety collection")
    
    # Vector Database Configuration
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: str = Field(default="wellness-knowledge", description="Pinecone index name")
    
    chroma_persist_directory: str = Field(default="./data/chroma", description="ChromaDB directory")
    chroma_collection_name: str = Field(default="wellness_chunks", description="ChromaDB collection")
    
    # AI/ML Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")
    openai_embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    openai_max_tokens: int = Field(default=4000, description="Max tokens for generation")
    openai_temperature: float = Field(default=0.7, description="Generation temperature")
    
    # NVIDIA Configuration
    nvidia_embedding_api_key: Optional[str] = Field(default=None, description="NVIDIA Embedding API key")
    nvidia_llm_api_key: Optional[str] = Field(default=None, description="NVIDIA LLM API key")
    nvidia_nim_api_key: Optional[str] = Field(default=None, description="NVIDIA NIM API key")
    nvidia_embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5", description="NVIDIA embedding model")
    nvidia_llm_model: str = Field(default="meta/llama-3.1-8b-instruct", description="NVIDIA LLM model")
    nvidia_llm_api_url: str = Field(default="https://integrate.api.nvidia.com/v1/chat/completions", description="NVIDIA LLM API URL")
    
    # Embedding Configuration
    embedding_dimension: int = Field(default=1024, description="Embedding vector dimension (1024 for NVIDIA, 384 for Sentence Transformer)")
    chunk_size: int = Field(default=512, description="Text chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")
    max_chunks_per_query: int = Field(default=5, description="Max chunks to retrieve per query")
    
    # Safety Configuration
    safety_enabled: bool = Field(default=True, description="Enable safety filtering")
    medical_advice_threshold: float = Field(default=0.8, description="Medical advice detection threshold")
    crisis_detection_threshold: float = Field(default=0.9, description="Crisis detection threshold")
    
    # Caching Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    embedding_cache_ttl: int = Field(default=86400, description="Embedding cache TTL in seconds")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Security
    secret_key: str = Field(default="change-me-in-production", description="Secret key for JWT")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Development
    reload_on_change: bool = Field(default=False, description="Reload on file changes")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )
    
    @property
    def use_pinecone(self) -> bool:
        """Check if Pinecone configuration is available."""
        return bool(self.pinecone_api_key and self.pinecone_environment)
    
    @property
    def use_openai(self) -> bool:
        """Check if OpenAI configuration is available."""
        return bool(self.openai_api_key)
    
    @property
    def use_nvidia_embeddings(self) -> bool:
        """Check if NVIDIA embedding configuration is available."""
        return bool(self.nvidia_embedding_api_key)
    
    @property
    def use_nvidia_llm(self) -> bool:
        """Check if NVIDIA LLM configuration is available."""
        return bool(self.nvidia_llm_api_key)


# Global settings instance
settings = Settings()