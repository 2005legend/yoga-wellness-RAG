"""
Script to reset ChromaDB collection when dimension mismatch occurs.
This deletes the existing collection so it can be recreated with the correct dimension.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from backend.config import settings
from backend.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


async def reset_chromadb():
    """Reset ChromaDB collection to fix dimension mismatches."""
    try:
        logger.info("Resetting ChromaDB collection...")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        
        # Try to delete the collection if it exists
        try:
            client.delete_collection(name=settings.chroma_collection_name)
            logger.info(f"Deleted existing collection: {settings.chroma_collection_name}")
        except Exception as e:
            logger.info(f"Collection doesn't exist or already deleted: {e}")
        
        # Determine correct dimension based on embedding service
        # NVIDIA uses 1024, Sentence Transformer uses 384
        if settings.nvidia_embedding_api_key:
            expected_dim = 1024
        else:
            expected_dim = 384
        
        # Create a new collection with the correct dimension
        collection = client.create_collection(
            name=settings.chroma_collection_name,
            metadata={"dimension": expected_dim}
        )
        
        logger.info(f"Created new collection '{settings.chroma_collection_name}' with dimension {expected_dim}")
        logger.info("ChromaDB reset complete. You can now run process_knowledge_base.py to populate it.")
        
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(reset_chromadb())


