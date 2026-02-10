"""
Text chunker for splitting documents into smaller segments.
Uses token-based chunking with overlap to preserve context.
"""

from typing import List
import logging
from src.utils import setup_logger


class TextChunker:
    """Split text into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = setup_logger(__name__)
        
        self.logger.info(
            f"TextChunker initialized: chunk_size={chunk_size}, overlap={overlap}"
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using whitespace splitting.
        
        Note: This is an approximation. For production, use tiktoken.
        
        Args:
            text: Input text
        
        Returns:
            Approximate token count
        """
        return len(text.split())
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences (simple approach).
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting on period, newline, etc.
        import re
        sentences = re.split(r'[.!?]\s+|\n+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
        
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split into words
        words = text.split()
        total_words = len(words)
        
        if total_words <= self.chunk_size:
            # Text is small enough, return as single chunk
            return [text]
        
        chunks = []
        start_idx = 0
        
        while start_idx < total_words:
            # Get chunk of words
            end_idx = min(start_idx + self.chunk_size, total_words)
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(chunk_text)
            
            # Move forward by (chunk_size - overlap)
            start_idx += (self.chunk_size - self.overlap)
            
            # Avoid infinite loop
            if start_idx >= total_words:
                break
        
        self.logger.info(
            f"Split text ({total_words} words) into {len(chunks)} chunks"
        )
        return chunks
    
    def chunk_documents(
        self, 
        documents: List[tuple]
    ) -> List[dict]:
        """
        Chunk multiple documents and add metadata.
        
        Args:
            documents: List of (filename, content) tuples
        
        Returns:
            List of dicts with keys: filename, chunk_id, chunk_text, total_chunks
        """
        all_chunks = []
        
        for filename, content in documents:
            chunks = self.chunk_text(content)
            
            for idx, chunk_text in enumerate(chunks, start=1):
                chunk_info = {
                    'filename': filename,
                    'chunk_id': idx,
                    'chunk_text': chunk_text,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_info)
        
        self.logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks"
        )
        return all_chunks
