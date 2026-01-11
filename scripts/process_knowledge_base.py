"""
Script to process the yoga knowledge base and store it in the vector database.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.chunking.service import ChunkingService
from backend.services.embeddings.service import EmbeddingService, EmbeddingProvider
from backend.services.retrieval.vector_db import VectorDBFactory
from backend.models.schemas import ContentCategory
from backend.core.logging import configure_logging, get_logger
from backend.config import settings

configure_logging()
logger = get_logger(__name__)


async def process_knowledge_base():
    """Process the yoga knowledge base and store in vector database."""
    try:
        # Initialize services
        logger.info("Initializing services...")
        chunking_service = ChunkingService()
        
        # Use NVIDIA embedding service if API key is available, else Sentence Transformer
        if settings.nvidia_embedding_api_key:
            logger.info("Using NVIDIA embedding service")
            embedding_service = EmbeddingService(
                provider=EmbeddingProvider.NVIDIA,
                config={
                    "api_key": settings.nvidia_embedding_api_key,
                    "model_name": settings.nvidia_embedding_model,
                    "dimension": 1024
                }
            )
        else:
            logger.info("Using Sentence Transformer embedding service")
            embedding_service = EmbeddingService(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
                config={"model_name": "all-MiniLM-L6-v2", "dimension": 384}
            )
        
        vector_db = VectorDBFactory.create()
        
        # Initialize async services
        await embedding_service.initialize()
        await vector_db.initialize()
        
        # Read knowledge base file
        knowledge_base_path = Path(__file__).parent.parent / "yoga_knowledge_base.md"
        if not knowledge_base_path.exists():
            logger.error(f"Knowledge base file not found: {knowledge_base_path}")
            return
        
        logger.info(f"Processing knowledge base: {knowledge_base_path}")
        
        # Chunk the knowledge base
        logger.info("Chunking knowledge base...")
        chunks = chunking_service.chunk_file(
            str(knowledge_base_path),
            document_id="yoga_knowledge_base",
            category=ContentCategory.YOGA
        )
        
        logger.info(f"Created {len(chunks)} chunks from knowledge base")
        
        if not chunks:
            logger.warning("No chunks created from knowledge base")
            return
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunk_texts = [chunk.content for chunk in chunks]
        result = await embedding_service.embed_texts(chunk_texts)
        embeddings = result.embeddings
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Store in vector database
        logger.info("Storing chunks in vector database...")
        upserted = await vector_db.upsert_chunks(chunks, embeddings)
        
        logger.info(f"Successfully stored {upserted} chunks in vector database")
        
        # Get stats
        stats = await vector_db.get_stats()
        logger.info(f"Vector database stats: {stats}")
        
        logger.info("Knowledge base processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing knowledge base: {e}", exc_info=True)
        raise
    finally:
        # Cleanup: close embedding service session if it's NVIDIA
        try:
            if 'embedding_service' in locals() and hasattr(embedding_service, '_service'):
                if hasattr(embedding_service._service, 'close'):
                    await embedding_service._service.close()
        except Exception as cleanup_error:
            logger.debug(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    asyncio.run(process_knowledge_base())


