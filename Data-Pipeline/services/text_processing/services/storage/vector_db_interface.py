from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations"""

    @abstractmethod
    def add_documents(self, embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents with embeddings to the vector database
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries corresponding to embeddings
            
        Returns:
            Dictionary with operation result (message, count, etc.)
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[
        Dict[str, Any]]:
        """Search for similar vectors
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            
        Returns:
            List of results with metadata and scores
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics
        
        Returns:
            Dictionary with stats (status, count, etc.)
        """
        pass

    @abstractmethod
    def verify_connection(self) -> bool:
        """Verify connection to the vector database
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
