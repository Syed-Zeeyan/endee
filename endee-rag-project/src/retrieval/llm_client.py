"""
Response formatter for generating answers from retrieved context.
No LLM required - uses retrieved chunks directly.
"""

from typing import Optional, List, Dict
import logging
from src.utils import setup_logger


class LLMClient:
    """Client for formatting responses from retrieved context (no external LLM)."""
    
    def __init__(self):
        """Initialize response formatter."""
        self.logger = setup_logger(__name__)
        self.logger.info("Response formatter initialized (local, no LLM)")
    
    def generate_response(
        self, 
        context: str, 
        query: str,
        sources: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Generate a response by formatting retrieved context.
        
        Args:
            context: Retrieved document chunks
            query: User question
            sources: List of source metadata (optional)
        
        Returns:
            Formatted response based on retrieved context
        """
        try:
            self.logger.info(f"Formatting response for query: {query[:50]}...")
            
            # Simple approach: Present the most relevant chunks as the answer
            response_parts = []
            
            response_parts.append(f"Based on the knowledge base, here's what I found:\n")
            
            # If we have structured sources, format them nicely
            if sources:
                for idx, source in enumerate(sources, start=1):
                    chunk_text = source.get('chunk_text', '')
                    source_file = source.get('source_file', 'unknown')
                    score = source.get('similarity_score', 0.0)
                    
                    response_parts.append(f"\n{idx}. From {source_file} (relevance: {score:.2f}):")
                    response_parts.append(f"   {chunk_text}\n")
            else:
                # Fallback: just use the context string
                response_parts.append(f"\n{context}\n")
            
            # Add a simple summary footer
            response_parts.append(
                "\nNote: This answer is compiled directly from the documents in the knowledge base. "
                "For more detailed information, please refer to the source documents listed above."
            )
            
            answer = '\n'.join(response_parts)
            
            self.logger.info(f"Generated response ({len(answer)} chars)")
            return answer
        
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return None
