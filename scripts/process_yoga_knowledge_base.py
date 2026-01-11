#!/usr/bin/env python3
"""
Process yoga knowledge base content for RAG pipeline.

This script:
1. Loads and processes the yoga_knowledge_base.md file
2. Chunks the content using our semantic chunker
3. Generates embeddings using NVIDIA API
4. Stores chunks and embeddings for retrieval
5. Categorizes content by difficulty level and type

Task 2.5: Process yoga knowledge base content
Requirements: 2.1, 2.2, 2.4
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.config import get_config
from backend.core.logging import LoggerMixin
from backend.services.chunking.service import ChunkingService
from backend.services.chunking.base import ChunkingConfig
from backend.services.embeddings.nvidia_service import NVIDIAEmbeddingService
from backend.models.schemas import ContentCategory, Chunk


class YogaKnowledgeProcessor(LoggerMixin):
    """Process yoga knowledge base for RAG pipeline."""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize chunking service with yoga-optimized settings
        chunking_config = ChunkingConfig(
            chunk_size=512,  # Optimal for yoga content
            chunk_overlap=50,
            min_chunk_size=100,
            max_chunk_size=800,
            preserve_sentences=True,
            preserve_paragraphs=True
        )
        self.chunking_service = ChunkingService(chunking_config)
        
        # Initialize NVIDIA embedding service
        self.embedding_service = NVIDIAEmbeddingService()
        
        # Storage for processed content
        self.processed_chunks: List[Chunk] = []
        self.embeddings: Dict[str, List[float]] = {}
        
    def load_knowledge_base(self, file_path: str) -> str:
        """Load the yoga knowledge base markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.log_event(
                "Knowledge base loaded successfully",
                file_path=file_path,
                content_length=len(content)
            )
            return content
            
        except Exception as e:
            self.log_error(e, {"file_path": file_path})
            raise
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract individual sections from the knowledge base."""
        sections = []
        
        # Split by main headers (## level)
        main_sections = re.split(r'\n## ', content)
        
        for i, section in enumerate(main_sections):
            if i == 0:
                # First section includes title, add ## back to others
                section_content = section
            else:
                section_content = "## " + section
            
            # Skip very short sections
            if len(section_content.strip()) < 100:
                continue
            
            # Determine section type and category
            section_info = self._categorize_section(section_content)
            sections.append(section_info)
        
        self.log_event(
            "Sections extracted from knowledge base",
            total_sections=len(sections)
        )
        return sections
    
    def _categorize_section(self, content: str) -> Dict[str, Any]:
        """Categorize a section by type and difficulty level."""
        # Extract title (first line)
        lines = content.strip().split('\n')
        title = lines[0].strip('#').strip() if lines else "Unknown Section"
        
        # Determine content category
        category = ContentCategory.YOGA  # Default
        
        if any(term in content.lower() for term in ['pranayama', 'breathing', 'breath']):
            category = ContentCategory.MEDITATION
        elif any(term in content.lower() for term in ['safety', 'contraindication', 'injury']):
            category = ContentCategory.WELLNESS
        elif any(term in content.lower() for term in ['nutrition', 'diet', 'food']):
            category = ContentCategory.NUTRITION
        elif any(term in content.lower() for term in ['exercise', 'fitness', 'strength']):
            category = ContentCategory.EXERCISE
        
        # Determine difficulty level
        difficulty = "beginner"  # Default
        
        if any(term in content.lower() for term in ['advanced', 'expert', 'headstand', 'wheel pose', 'lotus']):
            difficulty = "advanced"
        elif any(term in content.lower() for term in ['intermediate', 'triangle', 'warrior', 'dancer']):
            difficulty = "intermediate"
        elif any(term in content.lower() for term in ['beginner', 'basic', 'mountain pose', 'child']):
            difficulty = "beginner"
        
        # Extract pose type if applicable
        pose_type = None
        if any(term in content.lower() for term in ['standing', 'mountain', 'warrior', 'triangle']):
            pose_type = "standing"
        elif any(term in content.lower() for term in ['seated', 'lotus', 'vajrasana']):
            pose_type = "seated"
        elif any(term in content.lower() for term in ['backbend', 'cobra', 'wheel', 'camel']):
            pose_type = "backbend"
        elif any(term in content.lower() for term in ['forward fold', 'uttanasana']):
            pose_type = "forward_fold"
        elif any(term in content.lower() for term in ['inversion', 'headstand', 'shoulderstand']):
            pose_type = "inversion"
        elif any(term in content.lower() for term in ['balance', 'tree', 'dancer']):
            pose_type = "balance"
        elif any(term in content.lower() for term in ['twist', 'revolved']):
            pose_type = "twist"
        
        return {
            "title": title,
            "content": content,
            "category": category,
            "difficulty": difficulty,
            "pose_type": pose_type,
            "word_count": len(content.split())
        }
    
    def process_sections(self, sections: List[Dict[str, Any]]) -> List[Chunk]:
        """Process sections into chunks using semantic chunker."""
        all_chunks = []
        
        for i, section in enumerate(sections):
            try:
                # Create chunks for this section
                chunks = self.chunking_service.chunk_text(
                    content=section["content"],
                    source=f"yoga_knowledge_base_section_{i}",
                    category=section["category"]
                )
                
                # Add metadata to chunks
                for chunk in chunks:
                    # Add yoga-specific metadata
                    chunk.metadata.difficulty = section["difficulty"]
                    chunk.metadata.pose_type = section.get("pose_type")
                    chunk.metadata.section_title = section["title"]
                    chunk.metadata.word_count = section["word_count"]
                
                all_chunks.extend(chunks)
                
                self.log_event(
                    "Section processed into chunks",
                    section_title=section["title"],
                    chunks_created=len(chunks),
                    category=section["category"].value,
                    difficulty=section["difficulty"]
                )
                
            except Exception as e:
                self.log_error(e, {
                    "section_title": section["title"],
                    "section_index": i
                })
                continue
        
        self.log_event(
            "All sections processed",
            total_chunks=len(all_chunks),
            total_sections=len(sections)
        )
        
        return all_chunks
    
    async def generate_embeddings(self, chunks: List[Chunk]) -> Dict[str, List[float]]:
        """Generate embeddings for all chunks using NVIDIA API."""
        embeddings = {}
        
        # Extract texts for batch processing
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]
        
        try:
            # Generate embeddings in batches
            batch_size = 10  # Process in smaller batches for stability
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                
                self.log_event(
                    "Processing embedding batch",
                    batch_start=i,
                    batch_size=len(batch_texts)
                )
                
                # Generate embeddings for this batch
                batch_embeddings = await self.embedding_service.generate_embeddings(batch_texts)
                
                # Store embeddings with chunk IDs
                for chunk_id, embedding in zip(batch_ids, batch_embeddings):
                    embeddings[chunk_id] = embedding
                
                # Small delay between batches to avoid rate limiting
                await asyncio.sleep(0.5)
            
            self.log_event(
                "All embeddings generated",
                total_embeddings=len(embeddings)
            )
            
            return embeddings
            
        except Exception as e:
            self.log_error(e, {"total_chunks": len(chunks)})
            raise
    
    def save_processed_data(self, chunks: List[Chunk], embeddings: Dict[str, List[float]], output_dir: str):
        """Save processed chunks and embeddings to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks as JSON
        chunks_data = []
        for chunk in chunks:
            chunk_dict = {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": {
                    "document_id": chunk.metadata.document_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "source": chunk.metadata.source,
                    "category": chunk.metadata.category.value,
                    "tokens": chunk.metadata.tokens,
                    "created_at": chunk.metadata.created_at.isoformat(),
                    "difficulty": getattr(chunk.metadata, 'difficulty', None),
                    "pose_type": getattr(chunk.metadata, 'pose_type', None),
                    "section_title": getattr(chunk.metadata, 'section_title', None),
                    "word_count": getattr(chunk.metadata, 'word_count', None)
                }
            }
            chunks_data.append(chunk_dict)
        
        chunks_file = os.path.join(output_dir, "yoga_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        embeddings_file = os.path.join(output_dir, "yoga_embeddings.json")
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=2)
        
        # Save summary statistics
        stats = self._generate_statistics(chunks)
        stats_file = os.path.join(output_dir, "processing_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        self.log_event(
            "Processed data saved",
            output_dir=output_dir,
            chunks_file=chunks_file,
            embeddings_file=embeddings_file,
            stats_file=stats_file
        )
    
    def _generate_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Generate statistics about the processed content."""
        stats = {
            "total_chunks": len(chunks),
            "categories": {},
            "difficulties": {},
            "pose_types": {},
            "avg_tokens": 0,
            "total_tokens": 0
        }
        
        total_tokens = 0
        
        for chunk in chunks:
            # Category stats
            category = chunk.metadata.category.value
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Difficulty stats
            difficulty = getattr(chunk.metadata, 'difficulty', 'unknown')
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
            
            # Pose type stats
            pose_type = getattr(chunk.metadata, 'pose_type', 'general')
            if pose_type:
                stats["pose_types"][pose_type] = stats["pose_types"].get(pose_type, 0) + 1
            
            # Token stats
            total_tokens += chunk.metadata.tokens
        
        stats["total_tokens"] = total_tokens
        stats["avg_tokens"] = total_tokens / len(chunks) if chunks else 0
        
        return stats
    
    async def process_knowledge_base(self, input_file: str, output_dir: str):
        """Main processing pipeline."""
        self.log_event("Starting yoga knowledge base processing")
        
        try:
            # Step 1: Load knowledge base
            content = self.load_knowledge_base(input_file)
            
            # Step 2: Extract sections
            sections = self.extract_sections(content)
            
            # Step 3: Process into chunks
            chunks = self.process_sections(sections)
            self.processed_chunks = chunks
            
            # Step 4: Generate embeddings
            embeddings = await self.generate_embeddings(chunks)
            self.embeddings = embeddings
            
            # Step 5: Save processed data
            self.save_processed_data(chunks, embeddings, output_dir)
            
            self.log_event(
                "Yoga knowledge base processing completed successfully",
                total_chunks=len(chunks),
                total_embeddings=len(embeddings),
                output_dir=output_dir
            )
            
            return chunks, embeddings
            
        except Exception as e:
            self.log_error(e, {"input_file": input_file, "output_dir": output_dir})
            raise


async def main():
    """Main function to run the processing."""
    processor = YogaKnowledgeProcessor()
    
    # File paths
    input_file = "yoga_knowledge_base.md"
    output_dir = "data/processed_yoga_kb"
    
    try:
        chunks, embeddings = await processor.process_knowledge_base(input_file, output_dir)
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Generated {len(chunks)} chunks with {len(embeddings)} embeddings")
        print(f"💾 Data saved to: {output_dir}")
        
        # Print summary statistics
        stats_file = os.path.join(output_dir, "processing_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"\n📈 Processing Statistics:")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Average tokens per chunk: {stats['avg_tokens']:.1f}")
            print(f"   Categories: {dict(stats['categories'])}")
            print(f"   Difficulties: {dict(stats['difficulties'])}")
            print(f"   Pose types: {dict(stats['pose_types'])}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
