"""
Query engine for RAG (Retrieval Augmented Generation).
Orchestrates the complete query flow: embed → search → retrieve → generate.
"""

from typing import List, Dict, Any, Optional
import logging
from src.embeddings.embedding_model import EmbeddingModel
from src.endee.endee_client import EndeeClient
from src.retrieval.llm_client import LLMClient
from src.utils import setup_logger


class QueryEngine:
    """RAG query engine orchestrating retrieval and generation."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        endee_client: EndeeClient,
        llm_client: LLMClient,
        collection_name: str,
        top_k: int = 3
    ):
        """
        Initialize query engine.
        
        Args:
            embedding_model: Embedding model instance
            endee_client: Endee client instance
            llm_client: LLM client instance
            collection_name: Collection to search
            top_k: Number of results to retrieve
        """
        self.embedding_model = embedding_model
        self.endee_client = endee_client
        self.llm_client = llm_client
        self.collection_name = collection_name
        self.top_k = top_k
        self.logger = setup_logger(__name__)
        
        self.logger.info(
            f"QueryEngine initialized: collection={collection_name}, top_k={top_k}"
        )
    
    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute RAG query flow.
        
        Args:
            question: User question
            verbose: Whether to return detailed results
        
        Returns:
            Dict with:
                - answer: Generated response
                - sources: Retrieved chunks (if verbose)
                - query: Original question
        """
        self.logger.info(f"Processing query: {question}")
        
        # Step 1: Generate query embedding
        self.logger.info("Step 1: Generating query embedding")
        query_embedding = self.embedding_model.encode(question)
        
        # Step 2: Search Endee for similar chunks
        self.logger.info(f"Step 2: Searching Endee for top-{self.top_k} results")
        try:
            search_results = self.endee_client.search_vectors(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return {
                "query": question,
                "answer": "Error: Could not retrieve relevant information from the knowledge base.",
                "sources": [],
                "error": str(e)
            }
        
        if not search_results:
            self.logger.warning("No results found")
            return {
                "query": question,
                "answer": "I don't have enough information in the knowledge base to answer this question.",
                "sources": []
            }
        
        # Step 3: Extract context from results
        self.logger.info("Step 3: Assembling context from retrieved chunks")
        context_parts = []
        sources_info = []
        
        for idx, result in enumerate(search_results, start=1):
            metadata = result.get('metadata', {})
            chunk_text = metadata.get('chunk_text', '')
            source_file = metadata.get('source_file', 'unknown')
            score = result.get('score', 0.0)
            
            # Add to context
            context_part = f"[Source {idx}: {source_file}]\n{chunk_text}"
            context_parts.append(context_part)
            
            # Track source info
            sources_info.append({
                'source_file': source_file,
                'chunk_id': metadata.get('chunk_id', 0),
                'similarity_score': round(score, 3),
                'chunk_text': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text
            })
        
        # Combine all context
        full_context = '\n\n'.join(context_parts)
        
        # Step 4: Generate response using local formatter
        self.logger.info("Step 4: Formatting response from retrieved context")
        answer = self.llm_client.generate_response(
            context=full_context,
            query=question,
            sources=sources_info
        )
        
        if not answer:
            self.logger.error("LLM generation failed")
            return {
                "query": question,
                "answer": "Error: Could not generate a response.",
                "sources": sources_info if verbose else []
            }
        
        # Step 5: Return results
        self.logger.info("Query completed successfully")
        
        result = {
            "query": question,
            "answer": answer,
        }
        
        if verbose:
            result["sources"] = sources_info
            result["num_sources"] = len(sources_info)
        
        return result
