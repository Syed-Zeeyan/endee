"""
Endee REST API client for vector database operations.
Handles collection management, vector insertion, and similarity search.
"""

import requests
from typing import List, Dict, Any, Optional
import logging
from src.utils import setup_logger, handle_error


class EndeeClient:
    """Client for interacting with Endee vector database via REST API."""
    
    def __init__(self, base_url: str):
        """
        Initialize Endee client.
        
        Args:
            base_url: Base URL of Endee instance (e.g., http://localhost:8080)
        """
        self.base_url = base_url.rstrip('/')
        self.logger = setup_logger(__name__)
        self.logger.info(f"Initialized Endee client with URL: {self.base_url}")
    
    def create_collection(
        self, 
        name: str, 
        dimension: int
    ) -> Dict[str, Any]:
        """
        Create a new vector index using official Endee API.
        
        Args:
            name: Index name
            dimension: Vector dimension (must match embedding model)
        
        Returns:
            Response dict with index details
        
        Raises:
            Exception: If creation fails
        """
        url = f"{self.base_url}/api/v1/index/create"
        
        # Official Endee API format
        payload = {
            "index_name": str(name),
            "dim": int(dimension),
            "space_type": "cosine"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            self.logger.info(f"Creating index '{name}' with dimension {dimension}")
            self.logger.debug(f"Request URL: {url}")
            self.logger.debug(f"Request payload: {payload}")
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Handle response - Endee might return empty or non-JSON response
            try:
                result = response.json()
            except:
                # If response is not JSON, assume success if status is 2xx
                result = {"status": "created", "index_name": name}
            
            self.logger.info(f"Index '{name}' created successfully")
            return result
        
        except requests.exceptions.RequestException as e:
            # Log detailed error information
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Endee response status: {e.response.status_code}")
                try:
                    error_content = e.response.json()
                    self.logger.error(f"Endee error details: {error_content}")
                except:
                    self.logger.error(f"Endee error text: {e.response.text}")
                
                # Index might already exist (409 Conflict)
                if e.response.status_code == 409:
                    self.logger.warning(f"Index '{name}' already exists")
                    return {"status": "exists", "index_name": name}
            
            handle_error(self.logger, e, "create_collection")
            raise Exception(f"Failed to create index: {str(e)}")
    
    def insert_vectors(
        self, 
        collection_name: str, 
        vectors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Insert vectors into an index using official Endee API.
        
        Args:
            collection_name: Target index name
            vectors: List of vector objects, each with:
                - id: Unique identifier
                - vector: List of floats (embedding)
                - metadata: Dict with chunk_text, source_file, etc.
        
        Returns:
            Response dict with insertion status
        
        Raises:
            Exception: If insertion fails
        """
        # Official Endee API: index name in URL
        url = f"{self.base_url}/api/v1/index/{collection_name}/vector/insert"
        
        # Transform to official Endee format: list of objects with id, vector, meta
        formatted_vectors = []
        for vec in vectors:
            formatted_vec = {
                "id": vec["id"],
                "vector": vec["vector"],
                "meta": {
                    "text": vec["metadata"].get("chunk_text", ""),
                    "source": vec["metadata"].get("source_file", "")
                }
            }
            formatted_vectors.append(formatted_vec)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            self.logger.info(f"Inserting {len(formatted_vectors)} vectors into '{collection_name}'")
            response = requests.post(url, json=formatted_vectors, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Handle response - Endee insert may return empty or non-JSON response
            if response.status_code == 200:
                if not response.text.strip():
                    self.logger.info("Insert successful (empty response body)")
                    return True
                try:
                    return response.json()
                except:
                    self.logger.info("Insert successful (non-json response)")
                    return True
            
            return True
        
        except requests.exceptions.RequestException as e:
            handle_error(self.logger, e, "insert_vectors")
            raise Exception(f"Failed to insert vectors: {str(e)}")
    
    def search_vectors(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        top_k: int = 3,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using official Endee API with robust response parsing.
        
        Args:
            collection_name: Index to search
            query_vector: Query embedding (list of floats)
            top_k: Number of results to return
            include_metadata: Whether to return metadata
        
        Returns:
            List of results, each with:
                - id: Vector ID
                - score: Similarity score (0-1)
                - meta: Associated metadata
        """
        url = f"{self.base_url}/api/v1/index/{collection_name}/search"
        
        payload = {
            "vector": query_vector,
            "k": int(top_k),
            "include_vectors": False
        }
        
        try:
            self.logger.info(f"Searching '{collection_name}' for top-{top_k} similar vectors")
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code != 200:
                self.logger.error(f"Search HTTP {response.status_code}: {response.text}")
                return []
            
            # ---- SAFE PARSE ----
            try:
                data = response.json()
                if isinstance(data, dict):
                    results = data.get("results", [])
                    self.logger.info(f"Found {len(results)} results")
                    return results
                if isinstance(data, list):
                    self.logger.info(f"Found {len(data)} results")
                    return data
                return []
            except Exception:
                self.logger.warning("Non-JSON search response received. Attempting fallback parse.")
                text = response.text.strip()
                if not text:
                    return []
                return []
        
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections.
        
        Returns:
            List of collection objects
        
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}/api/v1/collections"
        
        try:
            self.logger.info("Listing all collections")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            collections = result.get("collections", [])
            
            self.logger.info(f"Found {len(collections)} collections")
            return collections
        
        except requests.exceptions.RequestException as e:
            handle_error(self.logger, e, "list_collections")
            raise Exception(f"Failed to list collections: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if Endee is reachable.
        
        Returns:
            True if healthy, False otherwise
        """
        url = f"{self.base_url}/health"
        
        try:
            response = requests.get(url, timeout=5)
            is_healthy = response.status_code == 200
            
            if is_healthy:
                self.logger.info("Endee health check: OK")
            else:
                self.logger.warning(f"Endee health check failed: {response.status_code}")
            
            return is_healthy
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Endee health check failed: {str(e)}")
            return False
