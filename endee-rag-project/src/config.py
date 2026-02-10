"""
Configuration module for Endee RAG Knowledge Assistant.
Loads environment variables and defines system constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Endee Configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
