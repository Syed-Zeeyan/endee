"""
Embedding model wrapper using sentence-transformers.
Generates semantic embeddings for text chunks and queries.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from src.utils import setup_logger


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.logger = setup_logger(__name__)
        self._model = None  # Lazy loading
        self.logger.info(f"Embedding model configured: {model_name}")
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self.logger.info("Model loaded successfully")
    
    def encode(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as list of floats
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        if not texts:
            return []
        
        self.logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")
        embeddings = self._model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding dimension (e.g., 384 for MiniLM)
        """
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
